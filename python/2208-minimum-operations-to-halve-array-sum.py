"""
2208 - Minimum Operations to Halve Array Sum [Medium]
Max Heap | Priority Queue | Array | Greedy
Time: O(n*log(n)) | Space: O(n)

You are given an array nums of positive integers. In one operation, you can choose any number from nums and 
reduce it to exactly half the number. (Note that you may choose this reduced number in future operations.)

Return the minimum number of operations to reduce the sum of nums by at least half.
"""

import heapq
from typing import List


class Solution:
    def halveArray(self, nums: List[int]) -> int:
        target = sum(nums) / 2
        heap = [-num for num in nums]
        heapq.heapify(heap)

        ans = 0
        while target > 0:
            ans += 1
            halved = heapq.heappop(heap) / 2
            target += halved
            heapq.heappush(heap, halved)

        return ans
