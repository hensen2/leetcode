"""
724 - Find Pivot Index [Easy]
Prefix Sum | Array
Time: O(n) | Space: O(1)

Given an array of integers nums, calculate the pivot index of this array.

The pivot index is the index where the sum of all the numbers strictly to the left of the index is equal 
to the sum of all the numbers strictly to the index's right.

If the index is on the left edge of the array, then the left sum is 0 because there are no elements to the 
left. This also applies to the right edge of the array.

Return the leftmost pivot index. If no such index exists, return -1.
"""

from typing import List


class Solution:
    def pivotIndex(self, nums: List[int]) -> int:
        prefix = sum(nums)
        left_sum = 0

        for i in range(len(nums)):
            if prefix - left_sum - nums[i] == left_sum:
                return i
            left_sum += nums[i]

        return -1
