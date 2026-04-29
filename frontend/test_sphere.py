import pytest
from .vec3 import Vec3
from .ray import Ray
from .sphere import intersect_sphere, EPSILON


def test_c001_direct_front_hit():
    ray = Ray(Vec3(0, 0, 0), Vec3(0, 0, -1))
    hit = intersect_sphere(ray, center=Vec3(0, 0, -5), radius=1.0)
    assert hit is not None
    assert abs(hit.t - 4.0) <= 1e-9
    assert abs(hit.normal.x - 0.0) <= 1e-9
    assert abs(hit.normal.y - 0.0) <= 1e-9
    assert abs(hit.normal.z - 1.0) <= 1e-9


def test_c002_miss_ray_pointing_away():
    ray = Ray(Vec3(0, 0, 0), Vec3(0, 0, 1))
    hit = intersect_sphere(ray, center=Vec3(0, 0, -5), radius=1.0)
    assert hit is None


def test_c003_tangent_ray():
    ray = Ray(Vec3(0, 1, 0), Vec3(0, 0, -1))
    hit = intersect_sphere(ray, center=Vec3(0, 0, -5), radius=1.0)
    assert hit is not None
    assert abs(hit.t - 5.0) <= 1e-6


def test_c004_inside_sphere_ray():
    ray = Ray(Vec3(0, 0, -5), Vec3(0, 0, -1))
    hit = intersect_sphere(ray, center=Vec3(0, 0, -5), radius=1.0)
    assert hit is not None
    assert abs(hit.t - 1.0) <= 1e-9
    # Normal must point toward ray origin (inward-pointing), i.e. Vec3(0,0,1)
    assert abs(hit.normal.x - 0.0) <= 1e-9
    assert abs(hit.normal.y - 0.0) <= 1e-9
    assert abs(hit.normal.z - 1.0) <= 1e-9


def test_c005_self_intersection_guard():
    # Origin 1e-5 inside the sphere surface (< EPSILON=1e-4), shooting outward.
    # Without the EPSILON guard the spurious t≈1e-5 hit would be returned.
    center = Vec3(0, 0, -5)
    radius = 1.0
    # Surface point (0,0,-4), outward normal (0,0,1).  Shift 1e-5 inward.
    origin = Vec3(0, 0, -4 - 1e-5)
    direction = Vec3(0, 0, 1)
    ray = Ray(origin, direction)
    hit = intersect_sphere(ray, center=center, radius=radius)
    assert hit is None


def test_c006_normal_unit_length():
    ray = Ray(Vec3(0, 0, 0), Vec3(0, 0, -1))
    hit = intersect_sphere(ray, center=Vec3(0, 0, -5), radius=1.0)
    assert hit is not None
    assert abs(hit.normal.length() - 1.0) <= 1e-9
