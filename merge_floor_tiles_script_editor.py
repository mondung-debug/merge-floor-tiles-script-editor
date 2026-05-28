# -*- coding: utf-8 -*-
"""
Merge Floor Tiles — Script Editor
Triangulates each tile's top face individually instead of extracting
a combined boundary polygon, which avoids O(n^3) ear-clip on thousands of vertices.

Usage:
    1. Select a component prim in USD Composer
    2. Run script

Config:
    FLOOR_CATEGORIES   — Category metadata values to match
    FLOOR_FAMILY_NAMES — FamilyName metadata values to match
    RESULT_PRIM_NAME   — Output mesh prim name
    DEACTIVATE_ORIGINAL — Deactivate original tile meshes after merge
    PLANE_Z_OFFSET     — Z offset for result plane (0 = keep original Z)
"""

import omni.usd
import numpy as np
from pxr import Usd, UsdGeom, Gf, Vt, Sdf
from collections import defaultdict

# ── Config ───────────────────────────────────────────────────────────────────
# FILTER_MODE: "all_mesh"  — all meshes under selection
#              "name"      — mesh prim name matches TILE_MESH_NAMES
#              "metadata"  — Category/FamilyName attribute match
FILTER_MODE        = "all_mesh"

FLOOR_CATEGORIES   = {"Floors", "Curtain Panels", "Access Floors"}
FLOOR_FAMILY_NAMES = {"Access Floor Panel", "Floor", "System Panel", "Access Floor"}
ATTR_CATEGORY      = "omni:hoops:metadata:Other:Category"
ATTR_FAMILY_NAME   = "omni:hoops:metadata:Other:tn__FamilyName_mA"

TILE_MESH_NAMES    = {"polySurface1"}

RESULT_PRIM_NAME   = "FloorPlane_Merged"
DEACTIVATE_ORIGINAL = True
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
            while cur and cur.IsValid() and cur.GetPath() != root_prim.GetPath():
                if _is_floor_tile(cur):
                    result.append(prim)
                    break
                cur = cur.GetParent()
    return result


def _world_mesh(mesh_prim):
    mesh = UsdGeom.Mesh(mesh_prim)
    pts_attr = mesh.GetPointsAttr()
    fvc_attr = mesh.GetFaceVertexCountsAttr()
    fvi_attr = mesh.GetFaceVertexIndicesAttr()
    if not (pts_attr and pts_attr.HasValue() and
            fvc_attr and fvc_attr.HasValue() and
            fvi_attr and fvi_attr.HasValue()):
        return None, None, None
    pts = np.array(pts_attr.Get(), dtype=np.float64)
    fvc = list(fvc_attr.Get())
    fvi = list(fvi_attr.Get())
    mat = UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    m   = np.array(mat)
    pts = np.hstack([pts, np.ones((len(pts), 1))]) @ m   # USD row-vector convention
    return pts[:, :3], fvc, fvi


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


def _ray_cast_2d(pt, poly_pts):
    """Point-in-polygon via ray casting."""
    x, y = float(pt[0]), float(pt[1])
    inside = False
    n = len(poly_pts)
    j = n - 1
    for i in range(n):
        xi, yi = float(poly_pts[i][0]), float(poly_pts[i][1])
        xj, yj = float(poly_pts[j][0]), float(poly_pts[j][1])
        if ((yi > y) != (yj > y)):
            denom = yj - yi
            if abs(denom) > 1e-300 and x < (xj - xi) * (y - yi) / denom + xi:
                inside = not inside
        j = i
    return inside


def _triangulate_delaunay(face_global, pts2d):
    """Delaunay triangulation filtered by point-in-polygon. Robust for bridged polygons."""
    try:
        from scipy.spatial import Delaunay as _Delaunay
    except ImportError:
        return []
    if len(face_global) < 3:
        return []
    local_pts = pts2d[face_global]
    try:
        tri = _Delaunay(local_pts)
    except Exception:
        return []
    result = []
    for simp in tri.simplices:
        centroid = local_pts[simp].mean(axis=0)
        if _ray_cast_2d(centroid, local_pts):
            result.append((face_global[simp[0]], face_global[simp[1]], face_global[simp[2]]))
    return result


def _signed_area_2d_list(face_global, pts2d):
    n = len(face_global)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        p0 = pts2d[face_global[i]]
        p1 = pts2d[face_global[j]]
        area += p0[0] * p1[1] - p1[0] * p0[1]
    return area * 0.5


def _triangulate_top_faces(meshes_data, global_mapping, offset_list, pts2d):
    """Per-tile top face triangulation using Delaunay + ray-cast filter."""
    all_triangles = []
    top_face_count = 0

    for (pts_w, fvc, fvi), off in zip(meshes_data, offset_list):
        if pts_w is None:
            continue

        max_z = float(pts_w[:, 2].max())
        min_z = float(pts_w[:, 2].min())
        z_tol = max((max_z - min_z) * 0.05, 1e-6)

        vi_idx = 0
        for fc in fvc:
            face_local = [fvi[vi_idx + k] for k in range(fc)]
            face_z_vals = [float(pts_w[v][2]) for v in face_local]

            is_top = all(z >= max_z - z_tol for z in face_z_vals)
            if is_top and fc >= 3:
                face_global = [global_mapping[off + v] for v in face_local]
                top_face_count += 1

                if fc == 3:
                    all_triangles.append(tuple(face_global))
                elif fc == 4:
                    all_triangles.append((face_global[0], face_global[1], face_global[2]))
                    all_triangles.append((face_global[0], face_global[2], face_global[3]))
                else:
                    # Delaunay + ray-cast: robust for bridged polygons with holes
                    tris = _triangulate_delaunay(face_global, pts2d)
                    if not tris:
                        # Fallback: fan from vertex 0 (may have errors for concave faces)
                        tris = [(face_global[0], face_global[i], face_global[i+1])
                                for i in range(1, fc - 1)]
                    all_triangles.extend(tris)

            vi_idx += fc

    print(f"[MergeFloorTiles] Top faces triangulated: {top_face_count}")
    return all_triangles


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

        # Collect world-space mesh data
        meshes_data = []
        for m in tile_meshes:
            d = _world_mesh(m)
            meshes_data.append(d)

        valid_pts = [d[0] for d in meshes_data if d[0] is not None]
        if not valid_pts:
            print("[MergeFloorTiles] No valid mesh data found.")
            continue

        # Merge vertices across all tiles so shared boundary verts are unified
        merged_pts, global_mapping = _merge_vertices(valid_pts)
        print(f"[MergeFloorTiles] Merged vertices: {len(merged_pts)}")

        # Build per-mesh offset into merged array
        offsets = []
        off     = 0
        for d in meshes_data:
            offsets.append(off)
            if d[0] is not None:
                off += len(d[0])

        # Project to 2D (XY plane) for triangulation
        pts2d = merged_pts[:, :2]

        # Per-tile top face triangulation (fast)
        all_triangles = _triangulate_top_faces(
            meshes_data, global_mapping, offsets, pts2d)

        if not all_triangles:
            print("[MergeFloorTiles] ERROR: Triangulation produced no triangles. "
                  "Check FILTER_MODE or mesh orientation.")
            continue

        print(f"[MergeFloorTiles] Triangles: {len(all_triangles)}")

        # Create USD Mesh
        result_path = f"{root_path}/{RESULT_PRIM_NAME}"
        if stage.GetPrimAtPath(result_path):
            stage.RemovePrim(result_path)

        result_mesh = UsdGeom.Mesh.Define(stage, result_path)

        # Compact: only include vertices actually referenced by triangles
        used_set  = sorted({v for tri in all_triangles for v in tri})
        vert_remap = {old: new for new, old in enumerate(used_set)}
        compact_pts = merged_pts[used_set]
        remapped_tris = [(vert_remap[a], vert_remap[b], vert_remap[c])
                         for a, b, c in all_triangles]

        z_final      = float(compact_pts[:, 2].max()) + PLANE_Z_OFFSET
        pts3d_result = np.hstack([compact_pts[:, :2],
                                   np.full((len(compact_pts), 1), z_final)])

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

        if DEACTIVATE_ORIGINAL:
            for m in tile_meshes:
                m.SetActive(False)
            print(f"[MergeFloorTiles] Deactivated {len(tile_meshes)} original tile(s)")

        stage.GetRootLayer().Save()
        print(f"[MergeFloorTiles] Done -> {result_path}")
        print(f"  Vertices: {len(pts3d_result)}, Triangles: {len(all_triangles)}")


run()
