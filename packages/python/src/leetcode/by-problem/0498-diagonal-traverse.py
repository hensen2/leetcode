"""
498 - Diagonal Traverse [Medium]
Two Pointers | Matrix | Array
Time: O(m*n) | Space: O(1)

Given an m x n matrix mat, return an array of all the elements of the array in a diagonal order.
"""

from typing import List


class Solution:
    def findDiagonalOrder(self, mat: List[List[int]]) -> List[int]:
        m = len(mat)
        n = len(mat[0])
        final = m * n

        i = 0
        j = 0
        up_direction = True
        ans = []

        while len(ans) < final:
            if up_direction:
                while i >= 0 and j < n:
                    ans.append(mat[i][j])
                    i -= 1
                    j += 1

                up_direction = False
                if j >= n:
                    i += 2
                    j -= 1
                else:
                    i += 1
            else:
                while i < m and j >= 0:
                    ans.append(mat[i][j])
                    i += 1
                    j -= 1

                up_direction = True
                if i >= m:
                    j += 2
                    i -= 1
                else:
                    j += 1

        return ans
