"""
414 - Third Maximum Number [Easy]
Array
Time: O(n) | Space: O(1)

Given an integer array nums, return the third distinct maximum number in this array. If the third maximum
does not exist, return the maximum number.
"""

from math import inf
from typing import List


class Solution:
    def thirdMax(self, nums: List[int]) -> int:
        first = nums[0]
        second = -inf
        third = -inf

        for i in range(1, len(nums)):
            if nums[i] == first or nums[i] == second or nums[i] == third:
                continue

            if nums[i] > first:
                third = second
                second = first
                first = nums[i]
            elif nums[i] > second:
                third = second
                second = nums[i]
            elif nums[i] > third:
                third = nums[i]

        return first if third == -inf else third
