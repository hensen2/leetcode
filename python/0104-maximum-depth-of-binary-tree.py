"""
104 - Maximum Depth of Binary Tree
Binary tree | DFS | Recursion
Time: O(n) | Space: O(n)
Space is O(n) because the tree could be a straight line.

Given the root of a binary tree, return its maximum depth.

A binary tree's maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.
"""

from typing import Optional

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        
        l = self.maxDepth(root.left)
        r = self.maxDepth(root.right)
        return max(l, r) + 1