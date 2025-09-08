"""
2248 - Intersection of Multiple Arrays [Easy]
Hash Table | Counting | Sorting | Array
Time: O(m*(n+log m)) | Space: O(n*m)
n = number of arrays, m = average number of elements in each array

Given a 2D integer array nums where nums[i] is a non-empty array of distinct positive integers, return
the list of integers that are present in each array of nums sorted in ascending order.
"""

from typing import List
from collections import defaultdict


class Solution:
    def intersection(self, nums: List[List[int]]) -> List[int]:
        counts = defaultdict(int)
        for arr in nums:
            for x in arr:
                counts[x] += 1

        n = len(nums)
        ans = []
        for key in counts:
            if counts[key] == n:
                ans.append(key)

        return sorted(ans)
