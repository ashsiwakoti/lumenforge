from __future__ import annotations
import math
from typing import Optional
from .vec3 import Vec3
from .ray import Ray
from .primitives import Hit

EPSILON = 1e-4


def intersect_sphere(ray: Ray, center: Vec3, radius: float) -> Optional[Hit]:
    oc = ray.origin.sub(center)
    a = ray.direction.dot(ray.direction)
    b_half = ray.direction.dot(oc)
    c = oc.dot(oc) - radius * radius
    discriminant = b_half * b_half - a * c

    if discriminant < 0:
        return None

    sqrt_disc = math.sqrt(discriminant)

    t = (-b_half - sqrt_disc) / a
    if t <= EPSILON:
        t = (-b_half + sqrt_disc) / a
        if t <= EPSILON:
            return None

    hit_point = ray.at(t)
    outward_normal = hit_point.sub(center).div(radius)

    if ray.direction.dot(outward_normal) > 0:
        normal = outward_normal.neg()
    else:
        normal = outward_normal

    return Hit(t=t, point=hit_point, normal=normal)


class Sphere:
    def __init__(self, center: Vec3, radius: float, material_id: str = ""):
        self.center = center
        self.radius = radius
        self.material_id = material_id

    def intersect(self, ray: Ray, t_min: float, t_max: float) -> Optional[Hit]:
        hit = intersect_sphere(ray, self.center, self.radius)
        if hit is None or hit.t < t_min or hit.t > t_max:
            return None
        return Hit(
            t=hit.t,
            point=hit.point,
            normal=hit.normal,
            material_id=self.material_id,
        )
