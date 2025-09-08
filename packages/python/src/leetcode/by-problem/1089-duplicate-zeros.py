"""
1089 - Duplicate Zeros [Easy]
Two Pointers | Array
Time: O(n) | Space: O(1)

Given a fixed-length integer array arr, duplicate each occurrence of zero, shifting the remaining elements
to the right.

Note that elements beyond the length of the original array are not written. Do the above modifications to
the input array in place and do not return anything.
"""

from typing import List


class Solution:
    def duplicateZeros(self, arr: List[int]) -> None:
        """
        Do not return anything, modify arr in-place instead.
        """
        count = 0
        right = len(arr) - 1

        for left in range(right + 1):
            if left > right - count:
                break

            if arr[left] == 0:
                if left == right - count:
                    arr[right] = 0
                    right -= 1
                    break

                count += 1

        end = right - count

        for i in range(end, -1, -1):
            if arr[i] == 0:
                arr[i + count] = 0
                count -= 1
                arr[i + count] = 0
            else:
                arr[i + count] = arr[i]
