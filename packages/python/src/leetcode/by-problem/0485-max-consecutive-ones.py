"""
485 - Max Consecutive Ones [Easy]
Array | Counting
Time: O(n) | Space: O(1)

Given a binary array nums, return the maximum number of consecutive 1's in the array.
"""

from typing import List


class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        count = 0
        maximum = 0

        for num in nums:
            if num == 1:
                count += 1
            else:
                maximum = max(maximum, count)
                count = 0

        return max(maximum, count)
