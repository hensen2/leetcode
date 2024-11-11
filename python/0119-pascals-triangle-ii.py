"""
119 - Pascal's Triangle II [Easy]
Dynamic Programming | Array
Time: O(rowIndex^2) | Space: O(1)
The last row does (rowIndex^2 + rowIndex) / 2 = O(rowIndex^2) operations, which is Gauss' formula.

Given an integer rowIndex, return the rowIndexth (0-indexed) row of the Pascal's triangle.

In Pascal's triangle, each number is the sum of the two numbers directly above it as shown in the problem.
"""

from typing import List


class Solution:
    def getRow(self, rowIndex: int) -> List[int]:
        ans = [1] * (rowIndex + 1)

        for i in range(1, rowIndex):
            for j in range(i, 0, -1):
                ans[j] += ans[j - 1]

        return ans
