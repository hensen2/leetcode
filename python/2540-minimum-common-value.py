"""
2540 - Minimum Common Value [Easy]
Two Pointers | Two Iterators | Array
Time: O(n) or O(n+m) | Space: O(1)

Given two integer arrays nums1 and nums2, sorted in non-decreasing order, return the minimum integer common to both arrays. 
If there is no common integer amongst nums1 and nums2, return -1.

Note that an integer is said to be common to nums1 and nums2 if both arrays have at least one occurrence of that integer.
"""

from typing import List


class Solution:
    def getCommon(self, nums1: List[int], nums2: List[int]) -> int:
        p1 = 0
        p2 = 0
        n1 = len(nums1)
        n2 = len(nums2)

        while p1 < n1 and p2 < n2:
            if nums1[p1] == nums2[p2]:
                return nums1[p1]
            elif nums1[p1] > nums2[p2]:
                p2 += 1
            else:
                p1 += 1

        return -1
