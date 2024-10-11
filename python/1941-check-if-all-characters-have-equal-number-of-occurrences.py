"""
1941 - Check if All Characters Have Equal Number of Occurrences [Easy]
Hash Table | Counting | String
Time: O(n) | Space: O(1) because the input can only have characters from the aplhabet (26)

Given a string s, return true if s is a good string, or false otherwise.

A string s is good if all the characters that appear in s have the same number of occurrences (i.e., the same frequency).
"""

from collections import defaultdict


class Solution:
    def areOccurrencesEqual(self, s: str) -> bool:
        counts = defaultdict(int)
        for c in s:
            counts[c] += 1

        frequencies = counts.values()
        return len(set(frequencies)) == 1
