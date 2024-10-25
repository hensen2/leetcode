"""
98 - Validate Binary Search Tree [Medium]
BST | DFS | Recursion
Time: O(n) | Space: O(n) because the tree could be a straight line

Given the root of a binary tree, determine if it is a valid binary search tree (BST).

A valid BST is defined as follows:
- The left subtree of a node contains only nodes with keys less than the node's key.
- The right subtree of a node contains only nodes with keys greater than the node's key.
- Both the left and right subtrees must also be binary search trees.
"""

from typing import Optional


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def dfs(node, small, large):
            if not node:
                return True

            if not (small < node.val < large):
                return False

            left = dfs(node.left, small, node.val)  # node.val is new upper bound
            right = dfs(node.right, node.val, large)  # node.val is new lower bound

            # tree is a BST if left and right subtrees are also BSTs
            return left and right

        return dfs(root, float("-inf"), float("inf"))
