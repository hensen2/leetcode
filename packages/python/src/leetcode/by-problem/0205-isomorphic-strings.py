"""
205 - Isomorphic Strings [Easy]
Hash Table | Check Existence | String
Time: O(n) | Space: O(k) or O(1)
Where k is the space of available characters in the alphabet, which is technically O(26) and could be considered constant space.

Given two strings s and t, determine if they are isomorphic.

Two strings s and t are isomorphic if the characters in s can be replaced to get t.

All occurrences of a character must be replaced with another character while preserving the order of characters. No
two characters may map to the same character, but a character may map to itself.
"""


class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        s_map = {}
        t_map = {}

        for c1, c2 in zip(s, t):
            if c1 not in s_map and c2 not in t_map:
                s_map[c1] = c2
                t_map[c2] = c1
            elif s_map.get(c1) != c2 or t_map.get(c2) != c1:
                return False

        return True
