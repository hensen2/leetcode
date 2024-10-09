"""
2000 - Reverse Prefix of Word [Easy]
Two Pointers | Middle Convergence | String
Time: O(n) | Space: O(n)

Given a 0-indexed string word and a character ch, reverse the segment of word that starts at index 0 
and ends at the index of the first occurrence of ch (inclusive). If the character ch does not exist 
in word, do nothing.

- For example, if word = "abcdefd" and ch = "d", then you should reverse the segment that starts at 0 
and ends at 3 (inclusive). The resulting string will be "dcbaefd".

Return the resulting string.
"""


class Solution:
    def reversePrefix(self, word: str, ch: str) -> str:
        chars = list(word)
        left = 0

        for right in range(len(word)):
            if chars[right] == ch:
                while left < right:
                    chars[left], chars[right] = chars[right], chars[left]
                    left += 1
                    right -= 1

                return "".join(chars)

        return word
