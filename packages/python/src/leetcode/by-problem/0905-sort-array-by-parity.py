"""
905 - Sort Array By Parity [Easy]
Two Pointers | Array
Time: O(n) | Space: O(1)

Given an integer array nums, move all the even integers at the beginning of the array followed by all the odd integers.

Return any array that satisfies this condition.
"""

from typing import List


class Solution:
    def sortArrayByParity(self, nums: List[int]) -> List[int]:
        left = 0
        right = len(nums) - 1

        while left < right:
            while left < right and nums[left] % 2 == 0:
                left += 1

            while left < right and nums[right] % 2 == 1:
                right -= 1

            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1

        return nums
