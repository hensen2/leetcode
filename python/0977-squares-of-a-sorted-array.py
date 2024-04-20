"""
977 - Squares of a Sorted Array
Two pointers | Middle convergance
Time: O(n) | Space: O(1)

Given an integer array nums sorted in non-decreasing order, return an array of the squares of each number sorted in non-decreasing order.
"""

from typing import List

class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        n = len(nums)
        l, r = 0, n - 1
        res = [0] * n

        while l <= r:
            left, right = abs(nums[l]), abs(nums[r])

            if left > right:
                res[r - l] = nums[l] ** 2
                l += 1
            else:
                res[r - l] = nums[r] ** 2
                r -= 1

        return res
