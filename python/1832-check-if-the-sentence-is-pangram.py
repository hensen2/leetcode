"""
1832 - Check if the Sentence is Pangram
Sets | Check existence
Time: O(n) | Space: O(1) 
Space is O(1) because there can only be at most 26 unique characters, which is O(26).

A pangram is a sentence where every letter of the English alphabet appears at least once.

Given a string sentence containing only lowercase English letters, return true if sentence is a pangram, or false otherwise.
"""

class Solution:
    def checkIfPangram(self, sentence: str) -> bool:
        chars = set(sentence)

        return len(chars) == 26