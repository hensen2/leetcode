"""
487 - Max Consecutive Ones II [Medium]
Sliding Window | Array
Time: O(n) | Space: O(1)

Given a binary array nums, return the maximum number of consecutive 1's in the array if you can flip at most one 0.
"""

from typing import List


class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        left = curr = ans = 0

        for right in range(len(nums)):
            if nums[right] == 0:
                curr += 1
            while curr > 1:
                if nums[left] == 0:
                    curr -= 1
                left += 1
            ans = max(ans, right - left + 1)

        return ans
