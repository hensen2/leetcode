"""
977 - Squares of a Sorted Array [Easy]
Two Pointers | Middle Convergence | Array | Sorting
Time: O(n) | Space: O(1)

Given an integer array nums sorted in non-decreasing order, return an array of the squares of each number sorted in non-decreasing order.
"""

from typing import List


class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        n = len(nums)
        i = 0
        j = n - 1
        res = [0] * n

        while i <= j:
            left = abs(nums[i])
            right = abs(nums[j])

            if left > right:
                res[j - i] = nums[i] ** 2
                i += 1
            else:
                res[j - i] = nums[j] ** 2
                j -= 1

        return res
