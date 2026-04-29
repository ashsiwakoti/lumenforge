from __future__ import annotations
from dataclasses import dataclass
from .vec3 import Vec3


@dataclass
class Ray:
    origin: Vec3
    direction: Vec3

    def at(self, t: float) -> Vec3:
        return self.origin.add(self.direction.mul(float(t)))
