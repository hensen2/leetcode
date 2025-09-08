"""
3 - Longest Substring Without Repeating Characters
Hash map | Sliding window | Greedy
Time: O(n) | Space: O(min(n, m))
Where n is the length of str and m is the space of possible characters.

Given a string s, find the length of the longest substring without repeating characters.
"""


class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        letters = {}
        left = 0
        ans = 0

        for right in range(len(s)):
            if s[right] in letters:
                left = max(letters[s[right]], left)

            letters[s[right]] = right + 1
            ans = max(ans, right - left + 1)

        return ans
