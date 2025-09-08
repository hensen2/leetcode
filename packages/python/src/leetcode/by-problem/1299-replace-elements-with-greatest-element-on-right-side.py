"""
1299 - Replace Elements with Greatest Element on Right Side [Easy]
Array
Time: O(n) | Space: O(1)

Given an array arr, replace every element in that array with the greatest element among the elements
to its right, and replace the last element with -1.

After doing so, return the array.
"""

from typing import List


class Solution:
    def replaceElements(self, arr: List[int]) -> List[int]:
        maximum = arr[-1]
        arr[-1] = -1

        for i in range(len(arr) - 2, -1, -1):
            arr[i], maximum = maximum, max(maximum, arr[i])

        return arr
