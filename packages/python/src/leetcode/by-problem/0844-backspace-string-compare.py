"""
844 - Backspace String Compare [Easy]
Stack | String
Time: O(n) | Space: O(n)

Given two strings s and t, return true if they are equal when both are typed into empty text editors.
'#' means a backspace character.

Note that after backspacing an empty text, the text will continue empty.
"""


class Solution:
    def backspaceCompare(self, s: str, t: str) -> bool:
        def build(string):
            stack = []

            for c in string:
                if c != "#":
                    stack.append(c)
                elif stack:
                    stack.pop()

            return "".join(stack)

        return build(s) == build(t)
