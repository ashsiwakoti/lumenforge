from __future__ import annotations
import math
from dataclasses import dataclass


@dataclass
class Vec3:
    x: float
    y: float
    z: float

    def __post_init__(self):
        self.x = float(self.x)
        self.y = float(self.y)
        self.z = float(self.z)

    # Component access
    def __getitem__(self, i: int) -> float:
        if i == 0:
            return self.x
        if i == 1:
            return self.y
        if i == 2:
            return self.z
        raise IndexError(f"Vec3 index {i} out of range")

    # Arithmetic — all pure, return new Vec3
    def add(self, other: Vec3) -> Vec3:
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def sub(self, other: Vec3) -> Vec3:
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def neg(self) -> Vec3:
        return Vec3(-self.x, -self.y, -self.z)

    def mul(self, scalar: float) -> Vec3:
        s = float(scalar)
        return Vec3(self.x * s, self.y * s, self.z * s)

    def div(self, scalar: float) -> Vec3:
        s = float(scalar)
        return Vec3(self.x / s, self.y / s, self.z / s)

    # Operator overloads
    def __add__(self, other: Vec3) -> Vec3:
        return self.add(other)

    def __sub__(self, other: Vec3) -> Vec3:
        return self.sub(other)

    def __neg__(self) -> Vec3:
        return self.neg()

    def __mul__(self, scalar: float) -> Vec3:
        return self.mul(scalar)

    def __rmul__(self, scalar: float) -> Vec3:
        return self.mul(scalar)

    def __truediv__(self, scalar: float) -> Vec3:
        return self.div(scalar)

    # Dot and cross
    def dot(self, other: Vec3) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vec3) -> Vec3:
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    # Length
    def length_squared(self) -> float:
        return self.x * self.x + self.y * self.y + self.z * self.z

    def length(self) -> float:
        return math.sqrt(self.length_squared())

    def normalize(self) -> Vec3:
        l = self.length()
        if l == 0.0:
            raise ValueError("Cannot normalize the zero vector")
        return self.div(l)

    def lerp(self, other: Vec3, t: float) -> Vec3:
        return self.add(other.sub(self).mul(t))

    # Equality uses exact float comparison (dataclass default)
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vec3):
            return NotImplemented
        return self.x == other.x and self.y == other.y and self.z == other.z
