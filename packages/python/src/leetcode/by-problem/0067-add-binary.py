"""
67 - Add Binary [Easy]
Bit Manipulation | Math | String
Time: O(m + n) where m is the length of a and n is the length of b | Space: O(1)

Given two binary strings a and b, return their sum as a binary string.
"""


class Solution:
    def addBinary(self, a: str, b: str) -> str:
        return "{0:b}".format(int(a, 2) + int(b, 2))
