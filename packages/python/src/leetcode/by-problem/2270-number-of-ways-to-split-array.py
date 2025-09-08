"""
2270 - Number of Ways to Split Array [Medium]
Prefix Sum | Array
Time: O(n) | Space: O(1)

You are given a 0-indexed integer array nums of length n.

nums contains a valid split at index i if the following are true:

- The sum of the first i + 1 elements is greater than or equal to the sum of the last n - i - 1 elements.
- There is at least one element to the right of i. That is, 0 <= i < n - 1.

Return the number of valid splits in nums.
"""

from typing import List


class Solution:
    def waysToSplitArray(self, nums: List[int]) -> int:
        ans = left_section = 0
        total = sum(nums)

        for i in range(len(nums) - 1):
            left_section += nums[i]
            right_section = total - left_section
            if left_section >= right_section:
                ans += 1

        return ans
