"""
1438 - Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit [Medium]
Queue | Monotonic Queue | Sliding Window | Array
Time: O(n) | Space: O(n)

Given an array of integers nums and an integer limit, return the size of the longest non-empty subarray 
such that the absolute difference between any two elements of this subarray is less than or equal to limit.
"""

from collections import deque
from typing import List


class Solution:
    def longestSubarray(self, nums: List[int], limit: int) -> int:
        increasing = deque()
        decreasing = deque()
        left = ans = 0

        for right in range(len(nums)):
            # maintain the monotonic deques
            while increasing and increasing[-1] > nums[right]:
                increasing.pop()
            while decreasing and decreasing[-1] < nums[right]:
                decreasing.pop()

            increasing.append(nums[right])
            decreasing.append(nums[right])

            # maintain window property
            while decreasing[0] - increasing[0] > limit:
                if nums[left] == decreasing[0]:
                    decreasing.popleft()
                if nums[left] == increasing[0]:
                    increasing.popleft()
                left += 1

            ans = max(ans, right - left + 1)

        return ans
