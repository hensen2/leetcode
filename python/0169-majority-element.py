"""
169 - Majority Element [Easy]
Array | Counting
Time: O(n) | Space: O(1)

Given an array nums of size n, return the majority element.

The majority element is the element that appears more than ⌊n / 2⌋ times. You may assume that the majority 
element always exists in the array.
"""

# Boyer-Moore Voting Algorithm

from typing import List


class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        majority = 0
        count = 0

        for n in nums:
            if count == 0:
                majority = n

            if n == majority:
                count += 1
            else:
                count -= 1

        return majority
