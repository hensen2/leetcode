"""
561 - Array Partition [Easy]
Sorting | Greedy | Array
Time: O(n*log(n)) | Space: O(n)
Python uses Timsort for sorting under the hood, which has the above time and space complexities.

Given an integer array nums of 2n integers, group these integers into n pairs (a1, b1), (a2, b2), ..., (an, bn) 
such that the sum of min(ai, bi) for all i is maximized. Return the maximized sum.
"""

from typing import List


class Solution:
    def arrayPairSum(self, nums: List[int]) -> int:
        nums.sort()
        maximum = 0

        for i in range(0, len(nums), 2):
            maximum += nums[i]

        return maximum
