"""
1832 - Check if the Sentence is Pangram [Easy]
Hash Set | Check Existence | String
Time: O(n) | Space: O(1) because the input can only have characters from the aplhabet (26)

A pangram is a sentence where every letter of the English alphabet appears at least once.

Given a string sentence containing only lowercase English letters, return true if sentence is a pangram, or false otherwise.
"""


class Solution:
    def checkIfPangram(self, sentence: str) -> bool:
        chars = set(sentence)

        return len(chars) == 26
