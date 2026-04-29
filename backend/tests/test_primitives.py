import math
import pytest
from lumenforge.primitives import Hit, Plane, Triangle, AABB, Sphere, Scene
from lumenforge.vec3 import Vec3
from lumenforge.ray import Ray
from lumenforge.sphere import EPSILON

_INF = float("inf")


def _approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def _vec3_approx(a: Vec3, b: Vec3, tol: float = 1e-9) -> bool:
    return abs(a.x - b.x) <= tol and abs(a.y - b.y) <= tol and abs(a.z - b.z) <= tol


# ---------------------------------------------------------------------------
# F-003-C-001: Plane hit and parallel miss
# ---------------------------------------------------------------------------

def test_c001_plane_hit():
    """Plane hit: ray (0,1,0)->(0,-1,0) vs plane origin=(0,-1,0) normal=(0,1,0) -> t=2.0."""
    ray = Ray(Vec3(0, 1, 0), Vec3(0, -1, 0))
    plane = Plane(origin=Vec3(0, -1, 0), normal=Vec3(0, 1, 0))
    result = plane.intersect(ray, EPSILON, _INF)
    assert result is not None, "Expected hit, got None"
    assert _approx(result.t, 2.0), f"Expected t=2.0, got {result.t}"


def test_c001_plane_parallel_miss():
    """Parallel ray (horizontal) against a horizontal plane returns None."""
    ray = Ray(Vec3(0, 1, 0), Vec3(1, 0, 0))
    plane = Plane(origin=Vec3(0, -1, 0), normal=Vec3(0, 1, 0))
    result = plane.intersect(ray, EPSILON, _INF)
    assert result is None, f"Expected None for parallel ray, got hit at t={getattr(result, 't', '?')}"


# ---------------------------------------------------------------------------
# F-003-C-002: Triangle Möller-Trumbore hit
# ---------------------------------------------------------------------------

def test_c002_triangle_hit():
    """Triangle (0,0,0)(1,0,0)(0,1,0) hit by Ray(0.25,0.25,1)(0,0,-1) -> t=1.0, bary=(0.5,0.25,0.25)."""
    tri = Triangle(Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0))
    ray = Ray(Vec3(0.25, 0.25, 1), Vec3(0, 0, -1))
    result = tri.intersect(ray, EPSILON, _INF)
    assert result is not None, "Expected hit, got None"
    assert _approx(result.t, 1.0), f"Expected t=1.0, got {result.t}"
    assert result.barycentric is not None, "Expected barycentric coords, got None"
    w, u, v = result.barycentric
    assert _approx(w, 0.5), f"Expected w=0.5, got {w}"
    assert _approx(u, 0.25), f"Expected u=0.25, got {u}"
    assert _approx(v, 0.25), f"Expected v=0.25, got {v}"


# ---------------------------------------------------------------------------
# F-003-C-003: Triangle miss (outside triangle)
# ---------------------------------------------------------------------------

def test_c003_triangle_miss():
    """Ray(2,2,1)(0,0,-1) against triangle (0,0,0)(1,0,0)(0,1,0) -> None."""
    tri = Triangle(Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0))
    ray = Ray(Vec3(2, 2, 1), Vec3(0, 0, -1))
    result = tri.intersect(ray, EPSILON, _INF)
    assert result is None, f"Expected None (miss), got hit at t={getattr(result, 't', '?')}"


# ---------------------------------------------------------------------------
# F-003-C-004: AABB slab hit — t_near and t_far
# ---------------------------------------------------------------------------

def test_c004_aabb_hit():
    """Box (-1,-1,-1)(1,1,1) hit by Ray(0,0,-3)(0,0,1) -> t_near=2.0, t_far=4.0."""
    box = AABB(Vec3(-1, -1, -1), Vec3(1, 1, 1))
    ray = Ray(Vec3(0, 0, -3), Vec3(0, 0, 1))
    result = box.intersect(ray, EPSILON, _INF)
    assert result is not None, "Expected hit, got None"
    t_near = result.t
    t_far = result.barycentric[1]
    assert _approx(t_near, 2.0), f"Expected t_near=2.0, got {t_near}"
    assert _approx(t_far, 4.0), f"Expected t_far=4.0, got {t_far}"


# ---------------------------------------------------------------------------
# F-003-C-005: AABB miss
# ---------------------------------------------------------------------------

def test_c005_aabb_miss():
    """Ray(2,2,-3)(0,0,1) against box (-1,-1,-1)(1,1,1) -> None."""
    box = AABB(Vec3(-1, -1, -1), Vec3(1, 1, 1))
    ray = Ray(Vec3(2, 2, -3), Vec3(0, 0, 1))
    result = box.intersect(ray, EPSILON, _INF)
    assert result is None, f"Expected None (miss), got hit at t={getattr(result, 't', '?')}"


# ---------------------------------------------------------------------------
# F-003-C-006: Scene returns nearest hit across all primitive types
# ---------------------------------------------------------------------------

def test_c006_scene_nearest_hit():
    """Scene([Sphere,Plane,Triangle]) returns nearest hit (smallest t) across all primitives."""
    # Ray: origin=(0,0,10), direction=(0,0,-1) — shooting in -Z
    ray = Ray(Vec3(0, 0, 10), Vec3(0, 0, -1))

    # Triangle at z=8 -> t=2.0
    tri = Triangle(
        Vec3(-1, -1, 8), Vec3(1, -1, 8), Vec3(0, 1, 8), material_id="triangle"
    )

    # Sphere center=(0,0,5), radius=0.5 -> t=4.5
    sphere = Sphere(Vec3(0, 0, 5), 0.5, material_id="sphere")

    # Plane at z=2 (normal pointing +Z) -> t=8.0
    plane = Plane(Vec3(0, 0, 2), Vec3(0, 0, 1), material_id="plane")

    scene = Scene([sphere, plane, tri])
    result = scene.intersect(ray, EPSILON, _INF)

    assert result is not None, "Expected a hit from scene, got None"
    # Triangle is nearest at t=2.0
    assert result.material_id == "triangle", (
        f"Expected nearest hit to be triangle (t=2.0), got '{result.material_id}' at t={result.t}"
    )
    assert _approx(result.t, 2.0, tol=1e-6), f"Expected t≈2.0, got {result.t}"
