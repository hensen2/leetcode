"""
1026 - Maximum Difference Between Node and Ancestor [Medium]
Binary Tree | DFS | Recursion
Time: O(n) | Space: O(n) because the tree could be a straight line

Given the root of a binary tree, find the maximum value v for which there exist different nodes a and b
where v = |a.val - b.val| and a is an ancestor of b.

A node a is an ancestor of b if either: any child of a is equal to b or any child of a is an ancestor of b.
"""

from typing import Optional


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def maxAncestorDiff(self, root: Optional[TreeNode]) -> int:
        ans = 0

        def dfs(node, maxVal, minVal):
            if not node:
                return

            nonlocal ans

            maxVal = max(maxVal, node.val)
            minVal = min(minVal, node.val)
            ans = max(ans, abs(maxVal - minVal))

            dfs(node.left, maxVal, minVal)
            dfs(node.right, maxVal, minVal)

        dfs(root, root.val, root.val)
        return ans
