# -*- coding: utf-8 -*-
"""
Merge Floor Tiles — Script Editor (BBox mode)
Each tile is replaced by its AABB top quad, then all quads are
vertex-welded and triangulated into a single UsdGeom.Mesh.
Holes are intentionally omitted — output is a solid floor plane.

Usage:
    1. Select a component prim in USD Composer
    2. Run script

Config:
    FILTER_MODE        — "all_mesh" / "name" / "metadata"
    FLOOR_CATEGORIES   — Category values to match (FILTER_MODE="metadata")
    FLOOR_FAMILY_NAMES — FamilyName values to match (FILTER_MODE="metadata")
    RESULT_PRIM_NAME   — Output mesh prim name
    ORIGINAL_ACTION    — What to do with original tiles after merge:
                         "deactivate" — set active=false
                         "delete"     — remove prim from stage
                         "none"       — leave as-is
    PLANE_Z_OFFSET     — Additional Z offset for the result plane
"""

import omni.usd
import numpy as np
from pxr import Usd, UsdGeom, Gf, Vt

# ── Config ───────────────────────────────────────────────────────────────────
FILTER_MODE        = "metadata"

FLOOR_CATEGORIES   = {"Curtain Panels", "Walls", "Floors"}
FLOOR_FAMILY_NAMES = {"System Panel", "Access Floor Panel", "Basic Wall", "Floor"}
ATTR_CATEGORY      = "omni:hoops:metadata:Other:Category"
ATTR_FAMILY_NAME   = "omni:hoops:metadata:Other:tn__FamilyName_mA"

TILE_MESH_NAMES    = {"polySurface1"}

RESULT_PRIM_NAME   = "FloorPlane_Merged"
ORIGINAL_ACTION    = "deactivate"   # "deactivate" / "delete" / "none"
PLANE_Z_OFFSET     = 0.0
WELD_TOLERANCE     = 1e-3
# ─────────────────────────────────────────────────────────────────────────────


def _get_attr(prim, attr_name):
    attr = prim.GetAttribute(attr_name)
    if attr and attr.HasValue():
        v = attr.Get()
        return str(v) if v is not None else None
    return None


def _is_floor_tile(prim):
    cat = _get_attr(prim, ATTR_CATEGORY)
    fam = _get_attr(prim, ATTR_FAMILY_NAME)
    return (cat and cat in FLOOR_CATEGORIES) or (fam and fam in FLOOR_FAMILY_NAMES)


def _collect_tile_meshes(root_prim):
    result = []
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsActive() or not prim.IsA(UsdGeom.Mesh):
            continue
        if FILTER_MODE == "all_mesh":
            result.append(prim)
        elif FILTER_MODE == "name":
            if prim.GetName() in TILE_MESH_NAMES:
                result.append(prim)
        elif FILTER_MODE == "metadata":
            if _is_floor_tile(prim):
                result.append(prim)
                continue
            cur = prim.GetParent()
            while cur and cur.IsValid():
                if _is_floor_tile(cur):
                    result.append(prim)
                    break
                if cur.GetPath() == root_prim.GetPath():
                    break
                cur = cur.GetParent()
    return result


def _world_bbox_top_quad(mesh_prim):
    """Compute world-space AABB and return the top face as a CCW quad."""
    mesh = UsdGeom.Mesh(mesh_prim)
    pts_attr = mesh.GetPointsAttr()
    if not (pts_attr and pts_attr.HasValue()):
        return None

    pts = np.array(pts_attr.Get(), dtype=np.float64)
    mat = UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    m   = np.array(mat)
    pts = np.hstack([pts, np.ones((len(pts), 1))]) @ m   # USD row-vector convention
    pts = pts[:, :3]

    min_x, max_x = float(pts[:, 0].min()), float(pts[:, 0].max())
    min_y, max_y = float(pts[:, 1].min()), float(pts[:, 1].max())
    max_z        = float(pts[:, 2].max())

    # CCW winding (normal = +Z): BL → BR → TR → TL
    return np.array([
        [min_x, min_y, max_z],
        [max_x, min_y, max_z],
        [max_x, max_y, max_z],
        [min_x, max_y, max_z],
    ], dtype=np.float64)


def _merge_vertices(pts_list, tol=WELD_TOLERANCE):
    all_pts = np.vstack(pts_list)
    mapping = np.arange(len(all_pts))
    unique  = []
    seen    = {}
    for i, p in enumerate(all_pts):
        key = tuple(np.round(p / tol).astype(int))
        if key in seen:
            mapping[i] = seen[key]
        else:
            seen[key] = len(unique)
            mapping[i] = len(unique)
            unique.append(p)
    return np.array(unique), mapping


# ── Main ─────────────────────────────────────────────────────────────────────

def run():
    ctx   = omni.usd.get_context()
    stage = ctx.get_stage()

    selection = ctx.get_selection().get_selected_prim_paths()
    if not selection:
        print("[MergeFloorTiles] Please select a component prim first.")
        return

    for root_path in selection:
        root_prim = stage.GetPrimAtPath(root_path)
        if not root_prim or not root_prim.IsValid():
            print(f"[MergeFloorTiles] Invalid prim: {root_path}")
            continue

        print(f"[MergeFloorTiles] Processing: {root_path}")

        tile_meshes = _collect_tile_meshes(root_prim)
        if not tile_meshes:
            print(f"[MergeFloorTiles] No tile meshes found under: {root_path}")
            continue

        print(f"[MergeFloorTiles] Found {len(tile_meshes)} tile mesh(es)")

        # Step 1: compute AABB top quad for each tile (world space)
        quads = []
        valid_meshes = []
        for m in tile_meshes:
            q = _world_bbox_top_quad(m)
            if q is not None:
                quads.append(q)
                valid_meshes.append(m)

        if not quads:
            print("[MergeFloorTiles] No valid mesh data found.")
            continue

        # Step 2: weld vertices across all quads
        merged_pts, global_mapping = _merge_vertices(quads)
        print(f"[MergeFloorTiles] BBox quads: {len(quads)}, Welded vertices: {len(merged_pts)}")

        # Step 3: triangulate each quad (2 triangles each, CCW)
        all_triangles = []
        for qi, q in enumerate(quads):
            base = qi * 4
            v0 = int(global_mapping[base + 0])
            v1 = int(global_mapping[base + 1])
            v2 = int(global_mapping[base + 2])
            v3 = int(global_mapping[base + 3])
            all_triangles.append((v0, v1, v2))
            all_triangles.append((v0, v2, v3))

        print(f"[MergeFloorTiles] Triangles: {len(all_triangles)}")

        # Step 4: compact vertices (only referenced ones)
        used_set   = sorted({v for tri in all_triangles for v in tri})
        vert_remap = {old: new for new, old in enumerate(used_set)}
        compact_pts = merged_pts[used_set]
        remapped_tris = [(vert_remap[a], vert_remap[b], vert_remap[c])
                         for a, b, c in all_triangles]

        z_final      = float(compact_pts[:, 2].max()) + PLANE_Z_OFFSET
        pts3d_result = np.hstack([compact_pts[:, :2],
                                   np.full((len(compact_pts), 1), z_final)])

        # Step 5: create USD Mesh
        result_path = f"{root_path}/{RESULT_PRIM_NAME}"
        if stage.GetPrimAtPath(result_path):
            stage.RemovePrim(result_path)

        result_mesh = UsdGeom.Mesh.Define(stage, result_path)
        result_mesh.GetPointsAttr().Set(
            Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts3d_result]))
        result_mesh.GetFaceVertexCountsAttr().Set(
            Vt.IntArray([3] * len(remapped_tris)))
        result_mesh.GetFaceVertexIndicesAttr().Set(
            Vt.IntArray([v for tri in remapped_tris for v in tri]))
        result_mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        result_mesh.CreateDoubleSidedAttr().Set(True)

        lo = Gf.Vec3f(*pts3d_result.min(axis=0))
        hi = Gf.Vec3f(*pts3d_result.max(axis=0))
        result_mesh.GetExtentAttr().Set(Vt.Vec3fArray([lo, hi]))

        if ORIGINAL_ACTION == "deactivate":
            for m in valid_meshes:
                m.SetActive(False)
            print(f"[MergeFloorTiles] Deactivated {len(valid_meshes)} original tile(s)")
        elif ORIGINAL_ACTION == "delete":
            for m in valid_meshes:
                stage.RemovePrim(m.GetPath())
            print(f"[MergeFloorTiles] Deleted {len(valid_meshes)} original tile(s)")
        else:
            print(f"[MergeFloorTiles] Original tiles unchanged (ORIGINAL_ACTION='none')")

        stage.GetRootLayer().Save()
        print(f"[MergeFloorTiles] Done -> {result_path}")
        print(f"  Vertices: {len(pts3d_result)}, Triangles: {len(remapped_tris)}")


run()
