"""
1346 - Check If N and Its Double Exist [Easy]
Hash Set | Check Existence | Array
Time: O(n) | Space: O(n)

Given an array arr of integers, check if there exist two indices i and j such that :
- i != j
- 0 <= i, j < arr.length
- arr[i] == 2 * arr[j]
"""

from typing import List


class Solution:
    def checkIfExist(self, arr: List[int]) -> bool:
        seen = set()

        for num in arr:
            if num * 2 in seen or num / 2 in seen:
                return True
            if num not in seen:
                seen.add(num)

        return False
