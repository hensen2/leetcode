"""
28 - Find the Index of the First Occurrence in a String [Easy]
Sliding Window | String Matching | String
Time: O(n*m) where n is the length of 'haystack' and m is the length of 'needle' | Space: O(1)

Given two strings needle and haystack, return the index of the first occurrence of needle in haystack,
or -1 if needle is not part of haystack.
"""


class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        n = len(haystack)
        m = len(needle)

        for i in range(n - m + 1):
            for j in range(m):
                if needle[j] != haystack[i + j]:
                    break
                if j == m - 1:
                    return i

        return -1
