"""
2126 - Destroying Asteroids [Medium]
Greedy | Sorting | Array
Time: O(n*log(n)) | Space: O(1)

You are given an integer mass, which represents the original mass of a planet. You are further given an integer 
array asteroids, where asteroids[i] is the mass of the ith asteroid.

You can arrange for the planet to collide with the asteroids in any arbitrary order. If the mass of the planet 
is greater than or equal to the mass of the asteroid, the asteroid is destroyed and the planet gains the mass of 
the asteroid. Otherwise, the planet is destroyed.

Return true if all asteroids can be destroyed. Otherwise, return false.
"""

from typing import List


class Solution:
    def asteroidsDestroyed(self, mass: int, asteroids: List[int]) -> bool:
        asteroids.sort()

        for a in asteroids:
            if a > mass:
                return False
            mass += a

        return True
