from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional
from .vec3 import Vec3
from .ray import Ray

EPSILON = 1e-4


@dataclass
class Hit:
    t: float
    normal: Vec3


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

    return Hit(t=t, normal=normal)
