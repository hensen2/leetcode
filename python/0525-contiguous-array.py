"""
525 - Contiguous Array
Hashmap
Time: O(n) | Space: O(n)

Given a binary array nums, return the maximum length of a contiguous subarray with an equal number of 0 and 1.
"""

from collections import defaultdict
from typing import List

class Solution:
    def findMaxLength(self, nums: List[int]) -> int:
        hashMap = defaultdict(int)
        hashMap[0] = -1
        curr = 0
        ans = 0

        for i, val in enumerate(nums):
            if val == 0:
                curr -= 1
            if val == 1:
                curr += 1

            if curr not in hashMap:
                hashMap[curr] = i
            else:
                ans = max(ans, i - hashMap[curr])

        return ans