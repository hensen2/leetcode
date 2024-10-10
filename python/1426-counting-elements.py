"""
1426 - Counting Elements [Easy]
Hash Set | Check Existence | Array
Time: O(n) | Space: O(n)

Given an integer array arr, count how many elements x there are, such that x + 1 is also in arr. If there are duplicates in arr, count them separately.
"""

from typing import List


class Solution:
    def countElements(self, arr: List[int]) -> int:
        res = 0
        num_set = set(arr)

        for i in arr:
            if i + 1 in num_set:
                res += 1

        return res
