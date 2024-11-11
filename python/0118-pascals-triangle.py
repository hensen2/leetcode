"""
118 - Pascal's Triangle [Easy]
Dynamic Programming | Array
Time: O(numRows^2) | Space: O(1)
The last row does (numRows^2 + numRows) / 2 = O(numRows^2) operations, which is Gauss' formula.

Given an integer numRows, return the first numRows of Pascal's triangle.

In Pascal's triangle, each number is the sum of the two numbers directly above it as shown in the problem.
"""

from typing import List


class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        ans = []

        for row in range(numRows):
            arr = [1] * (row + 1)

            for col in range(1, row):
                arr[col] = ans[row - 1][col - 1] + ans[row - 1][col]

            ans.append(arr)

        return ans
