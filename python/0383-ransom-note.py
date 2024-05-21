"""
383 - Ransom Note
Hash map | Counting
Time: O(n) | Space: O(k) or O(1)
Where k is the space of available characters in the alphabet, which is technically O(26) and could be considered constant space.

Given two strings ransomNote and magazine, return true if ransomNote can be constructed by using the letters from magazine and false otherwise.

Each letter in magazine can only be used once in ransomNote.
"""

from collections import defaultdict

class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        letters = defaultdict(int)

        for i in magazine:
            letters[i] += 1

        for j in ransomNote:
            if letters[j] > 0:           
                letters[j] -= 1
            else:
                return False
        
        return True
