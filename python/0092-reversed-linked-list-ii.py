"""
92 - Reversed Linked List II
Linked list | Pointer manipulation
Time: O(n) | Space: O(1)

Given the head of a singly linked list and two integers left and right where left <= right, reverse the nodes of the list from position left to position right, and return the reversed list.
"""

from typing import Optional

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def reverseBetween(
        self, head: Optional[ListNode], left: int, right: int
    ) -> Optional[ListNode]:
        # edge case when left == right
        if left == right:
            return head

        prev = None
        curr = head

        # iterate until head points at left position
        for _ in range(left - 1):
            prev = curr
            curr = curr.next

        # two pointers that will fix the final connections
        tail, con = curr, prev

        for _ in range(right - left + 1):
            next_node = curr.next  # first, make sure we don't lose the next node
            curr.next = prev  # reverse the direction of the pointer
            prev = curr  # set the current node to prev for the next node
            curr = next_node

        # Adjust the final connections
        if con:
            con.next = prev
        else:
            head = prev

        tail.next = curr

        return head