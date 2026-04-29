from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Protocol
from .vec3 import Vec3
from .ray import Ray

EPSILON = 1e-4


@dataclass
class Hit:
    t: float
    point: Vec3
    normal: Vec3
    material_id: str = field(default="")
    barycentric: Optional[Tuple[float, float, float]] = field(default=None)


class Hittable(Protocol):
    def intersect(self, ray: Ray, t_min: float, t_max: float) -> Optional[Hit]: ...


# ---------------------------------------------------------------------------
# Standalone intersection functions
# ---------------------------------------------------------------------------

def intersect_plane(
    ray: Ray, origin: Vec3, normal: Vec3, material_id: str = ""
) -> Optional[Hit]:
    denom = ray.direction.dot(normal)
    if abs(denom) < EPSILON:
        return None
    t = origin.sub(ray.origin).dot(normal) / denom
    if t <= EPSILON:
        return None
    face_normal = normal.neg() if denom > 0 else normal
    return Hit(t=t, point=ray.at(t), normal=face_normal, material_id=material_id)


def intersect_triangle(
    ray: Ray, v0: Vec3, v1: Vec3, v2: Vec3, material_id: str = ""
) -> Optional[Hit]:
    edge1 = v1.sub(v0)
    edge2 = v2.sub(v0)
    h = ray.direction.cross(edge2)
    a = edge1.dot(h)
    if abs(a) < EPSILON:
        return None
    f = 1.0 / a
    s = ray.origin.sub(v0)
    u = f * s.dot(h)
    if u < 0.0 or u > 1.0:
        return None
    q = s.cross(edge1)
    v = f * ray.direction.dot(q)
    if v < 0.0 or u + v > 1.0:
        return None
    t = f * edge2.dot(q)
    if t <= EPSILON:
        return None
    outward_normal = edge1.cross(edge2).normalize()
    if ray.direction.dot(outward_normal) > 0:
        outward_normal = outward_normal.neg()
    return Hit(
        t=t,
        point=ray.at(t),
        normal=outward_normal,
        material_id=material_id,
        barycentric=(1.0 - u - v, u, v),
    )


def intersect_aabb(
    ray: Ray, box_min: Vec3, box_max: Vec3
) -> Optional[Tuple[float, float]]:
    t_near = -math.inf
    t_far = math.inf
    for i in range(3):
        d = ray.direction[i]
        o = ray.origin[i]
        mn = box_min[i]
        mx = box_max[i]
        if abs(d) < 1e-12:
            if o < mn or o > mx:
                return None
        else:
            t1 = (mn - o) / d
            t2 = (mx - o) / d
            if t1 > t2:
                t1, t2 = t2, t1
            t_near = max(t_near, t1)
            t_far = min(t_far, t2)
    if t_near > t_far or t_far < EPSILON:
        return None
    return (t_near, t_far)


# ---------------------------------------------------------------------------
# Hittable classes
# ---------------------------------------------------------------------------

class Sphere:
    def __init__(self, center: Vec3, radius: float, material_id: str = ""):
        self.center = center
        self.radius = radius
        self.material_id = material_id

    def intersect(self, ray: Ray, t_min: float, t_max: float) -> Optional[Hit]:
        oc = ray.origin.sub(self.center)
        a = ray.direction.dot(ray.direction)
        b_half = ray.direction.dot(oc)
        c = oc.dot(oc) - self.radius * self.radius
        discriminant = b_half * b_half - a * c
        if discriminant < 0:
            return None
        sqrt_disc = math.sqrt(discriminant)
        t = (-b_half - sqrt_disc) / a
        if t <= EPSILON:
            t = (-b_half + sqrt_disc) / a
            if t <= EPSILON:
                return None
        if t < t_min or t > t_max:
            return None
        hit_point = ray.at(t)
        outward_normal = hit_point.sub(self.center).div(self.radius)
        if ray.direction.dot(outward_normal) > 0:
            normal = outward_normal.neg()
        else:
            normal = outward_normal
        return Hit(t=t, point=hit_point, normal=normal, material_id=self.material_id)


class Plane:
    def __init__(self, origin: Vec3, normal: Vec3, material_id: str = ""):
        self.origin = origin
        self.normal = normal
        self.material_id = material_id

    def intersect(self, ray: Ray, t_min: float, t_max: float) -> Optional[Hit]:
        hit = intersect_plane(ray, self.origin, self.normal, self.material_id)
        if hit is None or hit.t < t_min or hit.t > t_max:
            return None
        return hit


class Triangle:
    def __init__(self, v0: Vec3, v1: Vec3, v2: Vec3, material_id: str = ""):
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        self.material_id = material_id

    def intersect(self, ray: Ray, t_min: float, t_max: float) -> Optional[Hit]:
        hit = intersect_triangle(ray, self.v0, self.v1, self.v2, self.material_id)
        if hit is None or hit.t < t_min or hit.t > t_max:
            return None
        return hit


class AABB:
    def __init__(self, min_pt: Vec3, max_pt: Vec3, material_id: str = ""):
        self.min_pt = min_pt
        self.max_pt = max_pt
        self.material_id = material_id

    def intersect(self, ray: Ray, t_min: float, t_max: float) -> Optional[Hit]:
        t_near = t_min
        t_far = t_max
        hit_axis = -1
        for i in range(3):
            d = ray.direction[i]
            if d == 0.0:
                if ray.origin[i] < self.min_pt[i] or ray.origin[i] > self.max_pt[i]:
                    return None
                continue
            inv_d = 1.0 / d
            t0 = (self.min_pt[i] - ray.origin[i]) * inv_d
            t1 = (self.max_pt[i] - ray.origin[i]) * inv_d
            if inv_d < 0.0:
                t0, t1 = t1, t0
            if t0 > t_near:
                t_near = t0
                hit_axis = i
            t_far = min(t_far, t1)
            if t_near > t_far:
                return None

        normal_components = [0.0, 0.0, 0.0]
        if hit_axis >= 0:
            normal_components[hit_axis] = -1.0 if ray.direction[hit_axis] > 0.0 else 1.0
        normal = Vec3(*normal_components)

        return Hit(
            t=t_near,
            point=ray.at(t_near),
            normal=normal,
            material_id=self.material_id,
            barycentric=(t_near, t_far, 0.0),
        )


class Scene:
    def __init__(self, hittables: List):
        self.hittables = hittables

    def intersect(self, ray: Ray, t_min: float, t_max: float) -> Optional[Hit]:
        closest: Optional[Hit] = None
        current_max = t_max
        for obj in self.hittables:
            hit = obj.intersect(ray, t_min, current_max)
            if hit is not None:
                closest = hit
                current_max = hit.t
        return closest
