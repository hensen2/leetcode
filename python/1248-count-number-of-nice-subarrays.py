"""
1248 - Count Number of Nice Subarrays [Medium]
Hash Table | Counting | Prefix Sum | Array
Time: O(n) | Space: O(n)

Given an array of integers nums and an integer k. A continuous subarray is called nice if there are k odd numbers on it.

Return the number of nice sub-arrays.
"""

from typing import List
from collections import defaultdict


class Solution:
    def numberOfSubarrays(self, nums: List[int], k: int) -> int:
        counts = defaultdict(int)
        counts[0] = 1
        ans = curr = 0

        for num in nums:
            curr += num % 2
            ans += counts[curr - k]
            counts[curr] += 1

        return ans
