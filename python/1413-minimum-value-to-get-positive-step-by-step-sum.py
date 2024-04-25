"""
1413 - Minimum Value to Get Positive Step by Step Sum
Prefix Sum | Greedy
Time: O(n) | Space: O(1)

Given an array of integers nums, you start with an initial positive value startValue.

In each iteration, you calculate the step by step sum of startValue plus elements in nums (from left to right).

Return the minimum positive value of startValue such that the step by step sum is never less than 1.
"""

from typing import List

class Solution:
    def minStartValue(self, nums: List[int]) -> int:
        curr = 0
        minSum = 0

        for i in range(len(nums)):
            curr += nums[i]
            minSum = min(minSum, curr)

        return 1 - minSum # minSum + x = 1