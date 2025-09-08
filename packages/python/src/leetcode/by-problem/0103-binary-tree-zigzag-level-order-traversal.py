"""
103 - Binary Tree Zigzag Level Order Traversal [Medium]
Binary Tree | BFS | Queue
Time: O(n) | Space: O(n) because the queues could hold up to 2*(n/2) nodes

Given the root of a binary tree, return the zigzag level order traversal of its nodes' values. (i.e., from
left to right, then right to left for the next level and alternate between).
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
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []

        queue = deque([root])
        isLeft = True
        ans = []

        while queue:
            size = len(queue)
            vals = deque()

            for _ in range(size):
                node = queue.popleft()

                if isLeft:
                    vals.append(node.val)
                else:
                    vals.appendleft(node.val)

                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

            isLeft = not isLeft
            ans.append(vals)

        return ans
