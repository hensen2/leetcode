"""
1004 - Max Consecutive Ones III
Sliding Window
Time: O(n) | Space: O(1)

Given a binary array nums and an integer k, return the maximum number of consecutive 1's in the array if you can flip at most k 0's.
"""

from typing import List

class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        l = zeros = res = 0

        for r in range(len(nums)):
            if nums[r] == 0:
                zeros += 1

            # Loop until at most k zeros
            while zeros > k:
                if nums[l] == 0:
                    zeros -= 1
                l += 1

            res = max(res, r - l + 1)

        return res
