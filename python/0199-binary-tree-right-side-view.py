"""
199 - Binary Tree Right Side View
Binary tree | BFS | Deque
Time: O(n) | Space: O(n)
Space is O(n) because the queue could hold up to n/2 nodes.

Given the root of a binary tree, imagine yourself standing on the right side of it, return the values of the nodes you can see ordered from top to bottom.
"""

from typing import Optional, List
from collections import deque

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        ans = []
        queue = deque([root])

        while queue:
            size = len(queue)
            ans.append(queue[-1].val)

            for _ in range(size):
                node = queue.popleft()

                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

        return ans
