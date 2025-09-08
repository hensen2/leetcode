"""
54 - Spiral Matrix [Medium]
Many Pointers | Matrix | Array
Time: O(m*n) | Space: O(1)

Given an m x n matrix, return all elements of the matrix in spiral order.
"""

from typing import List


class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        right = len(matrix[0])
        bottom = len(matrix)
        left = 0
        top = 0

        final = right * bottom
        ans = []

        while len(ans) < final:
            for col in range(left, right):
                ans.append(matrix[top][col])

            top += 1

            for row in range(top, bottom):
                ans.append(matrix[row][right - 1])

            right -= 1

            if top != bottom:
                for col in range(right - 1, left - 1, -1):
                    ans.append(matrix[bottom - 1][col])

            bottom -= 1

            if left != right:
                for row in range(bottom - 1, top - 1, -1):
                    ans.append(matrix[row][left])

            left += 1

        return ans
