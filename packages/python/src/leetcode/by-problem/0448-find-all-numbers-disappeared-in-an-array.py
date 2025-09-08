"""
448 - Find All Numbers Disappeared in an Array [Easy]
Hash Set | Check Existence | Array
Time: O(n) | Space: O(n)

Given an array nums of n integers where nums[i] is in the range [1, n], return an array of all the integers
in the range [1, n] that do not appear in nums.
"""

from typing import List


class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        seen = set(nums)
        ans = []

        for i in range(1, len(nums) + 1):
            if i not in seen:
                ans.append(i)

        return ans


# Time: O(n) | Space: O(1)
# Slower on average than above solution, but uses constant extra space

# class Solution:
#     def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
#         for i in range(len(nums)):
#             j = abs(nums[i]) - 1

#             if nums[j] > 0:
#                 nums[j] *= -1

#         ans = []

#         for i in range(1, len(nums) + 1):
#             if nums[i - 1] > 0:
#                 ans.append(i)

#         return ans
