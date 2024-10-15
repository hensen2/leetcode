"""
1047 - Remove All Adjacent Duplicates In String [Easy]
Stack | String
Time: O(n) | Space: O(n)

You are given a string s consisting of lowercase English letters. A duplicate removal consists of choosing 
two adjacent and equal letters and removing them.

We repeatedly make duplicate removals on s until we no longer can.

Return the final string after all such duplicate removals have been made. It can be proven that the answer is unique.
"""


class Solution:
    def removeDuplicates(self, s: str) -> str:
        stack = []

        for c in s:
            if stack and c == stack[-1]:
                stack.pop()
            else:
                stack.append(c)

        return "".join(stack)
