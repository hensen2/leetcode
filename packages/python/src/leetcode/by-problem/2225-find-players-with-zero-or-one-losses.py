"""
2225 - Find Players With Zero or One Losses [Medium]
Hash Table | Counting | Sorting | Array
Time: O(n*log(n)) | Space: O(n)
Time complexity is O(n*log(n)) for sorting

You are given an integer array matches where matches[i] = [winneri, loseri] indicates that the player winneri defeated player loseri in a match.

Return a list answer of size 2 where:
- answer[0] is a list of all players that have not lost any matches.
- answer[1] is a list of all players that have lost exactly one match.

The values in the two lists should be returned in increasing order.
"""

from typing import List


class Solution:
    def findWinners(self, matches: List[List[int]]) -> List[List[int]]:
        losses_count = {}

        for winner, loser in matches:
            losses_count[winner] = losses_count.get(winner, 0)
            losses_count[loser] = losses_count.get(loser, 0) + 1

        zero_lose, one_lose = [], []
        for player, count in losses_count.items():
            if count == 0:
                zero_lose.append(player)
            if count == 1:
                one_lose.append(player)

        return [sorted(zero_lose), sorted(one_lose)]
