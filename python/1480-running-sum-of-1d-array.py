"""
1480 - Running Sum of 1D Array [Easy]
Prefix Sum | Array
Time: O(n) | Space: O(1)

Given an array nums. We define a running sum of an array as runningSum[i] = sum(nums[0]…nums[i]).

Return the running sum of nums.
"""

from typing import List


class Solution:
    def runningSum(self, nums: List[int]) -> List[int]:
        for i in range(1, len(nums)):
            nums[i] += nums[i - 1]

        return nums
