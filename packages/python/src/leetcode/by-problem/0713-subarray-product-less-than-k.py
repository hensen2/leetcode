"""
713 - Subarray Product Less Than K [Medium]
Sliding Window | Array
Time: O(n) | Space: O(1)

Given an array of integers nums and an integer k, return the number of contiguous subarrays where the
product of all the elements in the subarray is strictly less than k.
"""

from typing import List


class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        if k <= 1:
            return 0

        ans = left = 0
        curr = 1

        for right in range(len(nums)):
            curr *= nums[right]
            while curr >= k:
                curr //= nums[left]
                left += 1

            ans += right - left + 1

        return ans
