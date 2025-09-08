"""
24 - Swap Nodes in Pairs [Medium]
Linked List | Reversing Nodes
Time: O(n) | Space: O(1)

Given a linked list, swap every two adjacent nodes and return its head. You must solve the problem
without modifying the values in the list's nodes (i.e., only nodes themselves may be changed.)
"""

from typing import Optional


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return head

        ptr = head.next
        prev = None
        while head and head.next:
            if prev:
                prev.next = head.next
            prev = head

            next_node = head.next.next
            head.next.next = head

            head.next = next_node
            head = next_node

        return ptr
