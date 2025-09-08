"""
2090 - K Radius Subarray Averages [Medium]
Prefix Sum | Sliding Window | Fixed Window Size | Array
Time: O(n) | Space: O(1)

You are given a 0-indexed array nums of n integers, and an integer k.

The k-radius average for a subarray of nums centered at some index i with the radius k is the average of all elements in nums between the indices i - k and i + k (inclusive).
If there are less than k elements before or after the index i, then the k-radius average is -1.

Build and return an array avgs of length n where avgs[i] is the k-radius average for the subarray centered at index i.

The average of x elements is the sum of the x elements divided by x, using integer division. The integer division truncates toward zero, which means losing its fractional part.
"""

from typing import List


class Solution:
    def getAverages(self, nums: List[int], k: int) -> List[int]:
        # Edge case when k = 0, its average will be itself
        if k == 0:
            return nums

        windowSize = 2 * k + 1
        n = len(nums)
        avgs = [-1] * n

        # Edge case when any index won't have k elements on each side
        if windowSize > n:
            return avgs

        # First get sum of first window of nums
        windowSum = 0
        for i in range(windowSize):
            windowSum += nums[i]

        # The first k-radius center
        avgs[k] = windowSum // windowSize

        # Iterate from windowSize to n using sliding window method to update windowSum
        for i in range(windowSize, n):
            windowSum += nums[i] - nums[i - windowSize]
            avgs[i - k] = windowSum // windowSize

        return avgs
