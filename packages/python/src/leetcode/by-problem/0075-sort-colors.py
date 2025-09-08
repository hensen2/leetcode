"""
75 - Sort Colors [Medium]
Two Pointers | Sorting | Array
Time: O(n) | Space: O(1)

Given an array nums with n objects colored red, white, or blue, sort them in-place so that objects of
the same color are adjacent, with the colors in the order red, white, and blue.

We will use the integers 0, 1, and 2 to represent the color red, white, and blue, respectively.

You must solve this problem without using the library's sort function.
"""

from typing import List


class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        p0 = 0  # pointer for right boundary of zeros
        p1 = 0  # pointer for iterating
        p2 = len(nums) - 1  # pointer for left boundary of twos

        while p1 <= p2:
            if nums[p1] == 0:
                nums[p0], nums[p1] = nums[p1], nums[p0]
                p0 += 1
                p1 += 1
            elif nums[p1] == 2:
                nums[p1], nums[p2] = nums[p2], nums[p1]
                p2 -= 1
            else:
                p1 += 1
