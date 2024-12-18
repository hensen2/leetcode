"""
112 - Path Sum [Easy]
Binary Tree | DFS | Recursion
Time: O(n) | Space: O(n) because the tree could be a straight line

Given the root of a binary tree and an integer targetSum, return true if the tree has a root-to-leaf 
path such that adding up all the values along the path equals targetSum.

A leaf is a node with no children.
"""

from typing import Optional


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        def dfs(node, curr):
            if not node:
                return False

            if not node.left and not node.right:
                return (curr + node.val) == targetSum

            curr += node.val
            return dfs(node.left, curr) or dfs(node.right, curr)

        return dfs(root, 0)
