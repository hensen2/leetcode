"""
209 - Minimum Size Subarray Sum [Medium]
Sliding Window | Array
Time: O(n) | Space: O(1)

Given an array of positive integers nums and a positive integer target, return the minimal length of a 
subarray whose sum is greater than or equal to target. If there is no such subarray, return 0 instead.
"""

from typing import List


class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        left = curr = 0
        ans = float("inf")

        for right in range(len(nums)):
            curr += nums[right]

            while curr >= target:
                ans = min(ans, right - left + 1)
                curr -= nums[left]
                left += 1

        return ans if ans != float("inf") else 0
