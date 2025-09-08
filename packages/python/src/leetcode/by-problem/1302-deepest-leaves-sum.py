"""
1302 - Deepest Leaves Sum [Medium]
Binary Tree | BFS | Queue
Time: O(n) | Space: O(n) because the queue could hold up to n/2 nodes

Given the root of a binary tree, return the sum of values of its deepest leaves.
"""

from typing import Optional
from collections import deque


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def deepestLeavesSum(self, root: Optional[TreeNode]) -> int:
        ans = 0
        queue = deque([root])

        while queue:
            size = len(queue)
            ans = 0

            for _ in range(size):
                node = queue.popleft()
                ans += node.val

                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

        return ans
