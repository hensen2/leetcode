"""
125 - Valid Palindrome
Two pointers | Middle convergance
Time: O(n) | Space: O(1)

Given a string s, return true if it is a palindrome, false otherwise.

A string is a palindrome if it reads the same forward as backward. 
That means, after reversing it, it is still the same string. For example: "abcdcba", or "racecar".
"""

class Solution:
    def isPalindrome(self, s: str) -> bool:
        l = 0
        r = len(s) - 1

        while l < r:
            if s[l] != s[r]:
                return False
            l += 1
            r -= 1
    
        return True
