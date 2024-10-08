"""
1167 - Minimum Cost to Connect Sticks [Medium]
Min Heap | Priority Queue | Array | Greedy
Time: O(n*log(n)) | Space: O(n)

You have some number of sticks with positive integer lengths. These lengths are given as an array 
sticks, where sticks[i] is the length of the ith stick.

You can connect any two sticks of lengths x and y into one stick by paying a cost of x + y. You must 
connect all the sticks until there is only one stick remaining.

Return the minimum cost of connecting all the given sticks into one stick in this way.
"""

import heapq
from typing import List


class Solution:
    def connectSticks(self, sticks: List[int]) -> int:
        cost = 0
        heapq.heapify(sticks)

        while len(sticks) > 1:
            first = heapq.heappop(sticks)
            second = heapq.heappop(sticks)
            new_stick = first + second
            cost += new_stick
            heapq.heappush(sticks, new_stick)

        return cost
