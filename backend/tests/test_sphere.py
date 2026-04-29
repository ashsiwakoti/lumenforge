import pytest
from lumenforge.sphere import intersect_sphere, Hit
from lumenforge.vec3 import Vec3
from lumenforge.ray import Ray

EPSILON = 1e-4


def _vec3_approx(a: Vec3, b: Vec3, tol: float) -> bool:
    return (
        abs(a.x - b.x) <= tol
        and abs(a.y - b.y) <= tol
        and abs(a.z - b.z) <= tol
    )


def test_c001_direct_front_hit():
    """F-002-C-001: Ray from origin toward -Z hits sphere at z=-5, r=1."""
    ray = Ray(Vec3(0, 0, 0), Vec3(0, 0, -1))
    result = intersect_sphere(ray, Vec3(0, 0, -5), 1.0)
    assert result is not None, "Expected a hit, got None"
    assert abs(result.t - 4.0) <= 1e-9, f"Expected t=4.0, got {result.t}"
    assert _vec3_approx(result.normal, Vec3(0, 0, 1), 1e-9), (
        f"Expected normal=(0,0,1), got {result.normal}"
    )


def test_c002_miss():
    """F-002-C-002: Ray pointing away from sphere returns None."""
    ray = Ray(Vec3(0, 0, 0), Vec3(0, 0, 1))
    result = intersect_sphere(ray, Vec3(0, 0, -5), 1.0)
    assert result is None, f"Expected None (miss), got hit with t={getattr(result, 't', '?')}"


def test_c003_tangent():
    """F-002-C-003: Ray grazing sphere returns single hit at t=5.0."""
    ray = Ray(Vec3(0, 1, 0), Vec3(0, 0, -1))
    result = intersect_sphere(ray, Vec3(0, 0, -5), 1.0)
    assert result is not None, "Expected a tangent hit, got None"
    assert abs(result.t - 5.0) <= 1e-6, f"Expected t=5.0, got {result.t}"


def test_c004_inside_sphere():
    """F-002-C-004: Ray origin inside sphere returns back-face exit with inward normal."""
    ray = Ray(Vec3(0, 0, -5), Vec3(0, 0, -1))
    result = intersect_sphere(ray, Vec3(0, 0, -5), 1.0)
    assert result is not None, "Expected a hit from inside sphere, got None"
    assert abs(result.t - 1.0) <= 1e-9, f"Expected t=1.0, got {result.t}"
    dot = ray.direction.dot(result.normal)
    assert dot < 0, (
        f"Inside-sphere normal should point inward (ray·normal < 0), got dot={dot}"
    )


def test_c005_self_intersection_guard():
    """F-002-C-005: Origin offset 1e-5 (< EPSILON=1e-4) from surface returns no spurious hit."""
    center = Vec3(0, 0, -5)
    radius = 1.0
    outward_normal = Vec3(0, 0, 1)
    surface_point = Vec3(
        center.x + outward_normal.x * radius,
        center.y + outward_normal.y * radius,
        center.z + outward_normal.z * radius,
    )
    offset = 1e-5
    origin = Vec3(
        surface_point.x + outward_normal.x * offset,
        surface_point.y + outward_normal.y * offset,
        surface_point.z + outward_normal.z * offset,
    )
    ray = Ray(origin, outward_normal)
    result = intersect_sphere(ray, center, radius)
    assert result is None, (
        f"Self-intersection guard failed: got spurious hit at t={getattr(result, 't', '?')}"
    )


def test_c006_normal_unit_length():
    """F-002-C-006: Hit normal has unit length within 1e-9 for any valid hit."""
    test_cases = [
        (Ray(Vec3(0, 0, 0), Vec3(0, 0, -1)), Vec3(0, 0, -5), 1.0),
        (Ray(Vec3(0, 1, 0), Vec3(0, 0, -1)), Vec3(0, 0, -5), 1.0),
        (Ray(Vec3(1, 0, 0), Vec3(-1, 0, -1).normalize() if hasattr(Vec3(0,0,-1), 'normalize') else Vec3(-1, 0, -1)), Vec3(0, 0, -5), 2.0),
    ]
    for ray, center, radius in test_cases:
        result = intersect_sphere(ray, center, radius)
        if result is not None:
            length = result.normal.length()
            assert abs(length - 1.0) <= 1e-9, (
                f"Normal length should be 1.0, got {length} for ray={ray}, center={center}"
            )
