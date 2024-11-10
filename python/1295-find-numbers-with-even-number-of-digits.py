"""
1295 - Find Numbers with Even Number of Digits [Easy]
Array | Counting | Math
Time: O(n) | Space: O(1)

Given an array nums of integers, return how many of them contain an even number of digits.
"""

from typing import List


class Solution:
    def findNumbers(self, nums: List[int]) -> int:
        count = 0

        for num in nums:
            if len(str(num)) % 2 == 0:
                count += 1

        return count
