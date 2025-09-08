"""
1962 - Remove Stones to Minimize the Total [Medium]
Max Heap | Priority Queue | Array | Greedy
Time: O((n+k)*log(n)) where k is the number of operations | Space: O(n)

You are given a 0-indexed integer array piles, where piles[i] represents the number of stones in the
ith pile, and an integer k. You should apply the following operation exactly k times:

- Choose any piles[i] and remove floor(piles[i] / 2) stones from it.

Notice that you can apply the operation on the same pile more than once.

Return the minimum possible total number of stones remaining after applying the k operations.

floor(x) is the greatest integer that is smaller than or equal to x (i.e., rounds x down).
"""

import heapq
from math import ceil
from typing import List


class Solution:
    def minStoneSum(self, piles: List[int], k: int) -> int:
        heap = [-pile for pile in piles]
        heapq.heapify(heap)

        for _ in range(k):
            curr = heapq.heappop(heap)
            dec = ceil(curr / 2)
            heapq.heappush(heap, (curr - dec))

        return -sum(heap)
