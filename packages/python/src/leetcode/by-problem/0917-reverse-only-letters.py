"""
917 - Reverse Only Letters [Easy]
Two Pointers | Middle Convergence | String
Time: O(n) | Space: O(n)

Given a string s, reverse the string according to the following rules:

- All the characters that are not English letters remain in the same position.
- All the English letters (lowercase or uppercase) should be reversed.

Return s after reversing it.
"""


class Solution:
    def reverseOnlyLetters(self, s: str) -> str:
        chars = list(s)
        left = 0
        right = len(s) - 1

        while left < right:
            if chars[left].isalpha() and chars[right].isalpha():
                chars[left], chars[right] = chars[right], chars[left]
                left += 1
                right -= 1
            elif not chars[left].isalpha() and chars[right].isalpha():
                left += 1
            else:
                right -= 1

        return "".join(chars)
