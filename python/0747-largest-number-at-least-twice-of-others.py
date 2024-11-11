"""
747 - Largest Number At Least Twice of Others [Easy]
Array
Time: O(n) | Space: O(1)

You are given an integer array nums where the largest integer is unique.

Determine whether the largest element in the array is at least twice as much as every other number in 
the array. If it is, return the index of the largest element, or return -1 otherwise.
"""

from math import inf
from typing import List


class Solution:
    def dominantIndex(self, nums: List[int]) -> int:
        first = -inf
        second = -inf
        ans = -1

        for i in range(len(nums)):
            if nums[i] > first:
                second = first
                first = nums[i]
                ans = i
            elif nums[i] > second:
                second = nums[i]

        return ans if first >= 2 * second else -1
