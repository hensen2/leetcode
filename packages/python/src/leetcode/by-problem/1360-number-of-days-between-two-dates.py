"""
1360 - Number of Days Between Two Dates [Easy]
Datetime | Math
Time: O(1) | Space: O(1)

Write a program to count the number of days between two dates.

The two dates are given as strings, their format is YYYY-MM-DD as shown in the examples.
"""

from datetime import date


class Solution:
    def daysBetweenDates(self, date1: str, date2: str) -> int:
        timedelta = abs(date.fromisoformat(date1) - date.fromisoformat(date2))
        return timedelta.days
