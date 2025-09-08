"""
1448 - Count Good Nodes in Binary Tree [Medium]
Binary Tree | DFS | Recursion
Time: O(n) | Space: O(n) because the tree could be a straight line

Given a binary tree root, a node X in the tree is named good if in the path from root to X there are
no nodes with a value greater than X.

Return the number of good nodes in the binary tree.
"""


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def goodNodes(self, root: TreeNode) -> int:
        def dfs(node, maxVal):
            nonlocal ans

            if node.val >= maxVal:
                ans += 1
            if node.left:
                dfs(node.left, max(maxVal, node.val))
            if node.right:
                dfs(node.right, max(maxVal, node.val))

        ans = 0
        dfs(root, root.val)
        return ans
