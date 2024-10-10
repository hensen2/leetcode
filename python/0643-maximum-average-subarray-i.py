"""
643 - Maximum Average Subarray I [Easy]
Sliding Window | Fixed Window Size | Array
Time: O(n) | Space: O(1)

You are given an integer array nums consisting of n elements, and an integer k.

Find a contiguous subarray whose length is equal to k that has the maximum average value and return this value.
Any answer with a calculation error less than 10^-5 will be accepted.
"""

from typing import List


class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        curr = 0

        for i in range(k):
            curr += nums[i]

        res = curr

        for i in range(k, len(nums)):
            curr += nums[i] - nums[i - k]
            res = max(res, curr)

        return res / k
