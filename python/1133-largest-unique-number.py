"""
1133 - Largest Unique Number [Easy]
Hash Table | Counting | Array
Time: O(n) | Space: O(n)

Given an integer array nums, return the largest integer that only occurs once. If no integer occurs once, return -1.
"""

from collections import defaultdict
from typing import List


class Solution:
    def largestUniqueNumber(self, nums: List[int]) -> int:
        counts = defaultdict(int)
        res = -1

        for num in nums:
            counts[num] += 1

        for num, count in counts.items():
            if count == 1:
                res = max(res, num)

        return res
