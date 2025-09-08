"""
239 - Sliding Window Maximum [Hard]
Queue | Monotonic Queue | Sliding Window | Array
Time: O(n) | Space: O(k)

You are given an array of integers nums, there is a sliding window of size k which is moving from the
very left of the array to the very right. You can only see the k numbers in the window. Each time the
sliding window moves right by one position.

Return the max sliding window.
"""

from collections import deque
from typing import List


class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        ans = []
        queue = deque()
        for i in range(len(nums)):
            # maintain monotonic decreasing.
            # all elements in the deque smaller than the current one
            # have no chance of being the maximum, so get rid of them
            while queue and nums[i] > nums[queue[-1]]:
                queue.pop()

            queue.append(i)

            # queue[0] is the index of the maximum element.
            # if queue[0] + k == i, then it is outside the window
            if queue[0] + k == i:
                queue.popleft()

            # only add to the answer once our window has reached size k
            if i >= k - 1:
                ans.append(nums[queue[0]])

        return ans
