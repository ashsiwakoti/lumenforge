"""Pytest suite for F-001: Vector and Ray Primitives."""
import pytest
from .vec3 import Vec3
from .ray import Ray


# F-001-C-001: Vec3.dot exact equality (no tolerance)
def test_dot_exact():
    result = Vec3(1, 2, 3).dot(Vec3(4, 5, 6))
    assert result == 32.0


# F-001-C-002: Vec3.cross orthogonality — exact equality
def test_cross_unit_axes():
    result = Vec3(1, 0, 0).cross(Vec3(0, 1, 0))
    assert result == Vec3(0, 0, 1)


def test_cross_general():
    result = Vec3(2, 3, 4).cross(Vec3(5, 6, 7))
    assert result == Vec3(-3, 6, -3)


# F-001-C-003: Vec3.normalize unit length within 1e-12
def test_normalize_unit_length():
    v = Vec3(3, 4, 5)
    n = v.normalize()
    assert abs(n.length() - 1.0) < 1e-12


def test_normalize_known_values():
    n = Vec3(3, 4, 0).normalize()
    assert abs(n.x - 0.6) < 1e-12
    assert abs(n.y - 0.8) < 1e-12
    assert abs(n.z - 0.0) < 1e-12


def test_normalize_zero_raises():
    with pytest.raises(ValueError):
        Vec3(0, 0, 0).normalize()


# F-001-C-004: Ray.at correctness — exact
def test_ray_at_half():
    r = Ray(Vec3(1, 2, 3), Vec3(4, 5, 6))
    result = r.at(0.5)
    assert result == Vec3(3.0, 4.5, 6.0)


def test_ray_at_zero_returns_origin():
    origin = Vec3(1, 2, 3)
    r = Ray(origin, Vec3(4, 5, 6))
    result = r.at(0)
    assert result == origin


# F-001-C-005: Purity — id() unchanged after Vec3.add
def test_add_purity():
    v1 = Vec3(1, 2, 3)
    v2 = Vec3(4, 5, 6)
    id_v1_before = id(v1)
    id_v2_before = id(v2)
    _ = v1.add(v2)
    assert id(v1) == id_v1_before
    assert id(v2) == id_v2_before
    # Verify values are also unchanged
    assert v1 == Vec3(1, 2, 3)
    assert v2 == Vec3(4, 5, 6)


def test_sub_purity():
    v1 = Vec3(5, 6, 7)
    v2 = Vec3(1, 2, 3)
    id_v1_before = id(v1)
    _ = v1.sub(v2)
    assert id(v1) == id_v1_before
    assert v1 == Vec3(5, 6, 7)


# F-001-C-006: Type stability — all binary ops return Vec3
def test_type_stability_add():
    v1, v2 = Vec3(1, 2, 3), Vec3(4, 5, 6)
    assert isinstance(v1 + v2, Vec3)


def test_type_stability_sub():
    v1, v2 = Vec3(5, 6, 7), Vec3(1, 2, 3)
    assert isinstance(v1 - v2, Vec3)


def test_type_stability_neg():
    assert isinstance(-Vec3(1, 2, 3), Vec3)


def test_type_stability_mul():
    assert isinstance(Vec3(1, 2, 3) * 2.0, Vec3)
    assert isinstance(2.0 * Vec3(1, 2, 3), Vec3)


def test_type_stability_div():
    assert isinstance(Vec3(2, 4, 6) / 2.0, Vec3)


def test_type_stability_cross():
    assert isinstance(Vec3(1, 0, 0).cross(Vec3(0, 1, 0)), Vec3)


def test_type_stability_normalize():
    assert isinstance(Vec3(1, 2, 3).normalize(), Vec3)


def test_type_stability_lerp():
    assert isinstance(Vec3(0, 0, 0).lerp(Vec3(1, 1, 1), 0.5), Vec3)


# Additional correctness checks
def test_length_squared():
    assert Vec3(3, 4, 0).length_squared() == 25.0


def test_length():
    assert Vec3(3, 4, 0).length() == 5.0


def test_lerp_unclamped():
    v0 = Vec3(0, 0, 0)
    v1 = Vec3(1, 1, 1)
    # t=2 should extrapolate beyond v1
    result = v0.lerp(v1, 2.0)
    assert result == Vec3(2.0, 2.0, 2.0)


def test_component_access():
    v = Vec3(7, 8, 9)
    assert v[0] == 7.0
    assert v[1] == 8.0
    assert v[2] == 9.0


def test_component_access_out_of_range():
    with pytest.raises(IndexError):
        _ = Vec3(1, 2, 3)[3]
