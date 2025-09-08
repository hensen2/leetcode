"""
771 - Jewels and Stones
Hash set | Check existence
Time: O(n + m) | Space: O(n)
Where n is the length of jewels and m is the length of stones.

You're given strings jewels representing the types of stones that are jewels, and stones representing the stones you have. Each character in stones is a type of stone you have. You want to know how many of the stones you have are also jewels.

Letters are case sensitive, so "a" is considered a different type of stone from "A".
"""


class Solution:
    def numJewelsInStones(self, jewels: str, stones: str) -> int:
        ans = 0
        jewels_set = set(jewels)

        for s in stones:
            if s in jewels_set:
                ans += 1

        return ans
