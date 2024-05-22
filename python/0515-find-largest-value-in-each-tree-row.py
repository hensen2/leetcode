"""
515 - Find Largest Value in Each Tree Row
Binary tree | BFS | Deque
Time: O(n) | Space: O(n)
Space is O(n) because the queue could hold up to n/2 nodes.

Given the root of a binary tree, return an array of the largest value in each row of the tree (0-indexed).
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
            currMax = queue[0].val

            for _ in range(size):
                node = queue.popleft()
                currMax = max(currMax, node.val)

                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

            ans.append(currMax)

        return ans
