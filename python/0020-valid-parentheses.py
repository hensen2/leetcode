"""
20 - Valid Parentheses [Easy]
Stack | String
Time: O(n) | Space: O(n)

Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:

1. Open brackets must be closed by the same type of brackets.
2. Open brackets must be closed in the correct order.
3. Every close bracket has a corresponding open bracket of the same type.
"""


class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        brackets = {"(": ")", "[": "]", "{": "}"}

        for c in s:
            if c in brackets:  # if opening bracket
                stack.append(c)  # then push onto stack
            else:
                if not stack:  # if closing bracket and stack empty
                    return False  # return False

                # check if brackets are a match
                top = stack.pop()
                if brackets[top] != c:
                    return False

        # return True if empty, otherwise False
        return not stack
