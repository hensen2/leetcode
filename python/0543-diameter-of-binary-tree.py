"""
543 - Diameter of Binary Tree [Easy]
Binary Tree | DFS | Recursion
Time: O(n) | Space: O(n) because the tree could be a straight line

Given the root of a binary tree, return the length of the diameter of the tree.

The diameter of a binary tree is the length of the longest path between any two nodes in a tree. 
This path may or may not pass through the root.

The length of a path between two nodes is represented by the number of edges between them.
"""

from typing import Optional


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        def dfs(node):
            if not node:
                return 0

            nonlocal ans

            left = dfs(node.left)
            right = dfs(node.right)

            ans = max(ans, left + right)

            return max(left, right) + 1

        ans = 0
        dfs(root)
        return ans
