"""
1004 - Max Consecutive Ones III [Medium]
Sliding Window | Array
Time: O(n) | Space: O(1)

Given a binary array nums and an integer k, return the maximum number of consecutive 1's in the array if you can flip at most k 0's.
"""

from typing import List


class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        left = zeros = res = 0

        for right in range(len(nums)):
            if nums[right] == 0:
                zeros += 1

            # Loop until at most k zeros
            while zeros > k:
                if nums[left] == 0:
                    zeros -= 1
                left += 1

            res = max(res, right - left + 1)

        return res
