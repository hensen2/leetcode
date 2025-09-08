from collections.abc import Iterator
from typing import Any


class ListNode:
    def __init__(self, val, next=None):
        self.val = val
        self.next = next

    def __repr__(self):
        return f"Node({self.val}, {repr(self.next)})"


class SinglyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def __iter__(self) -> Iterator[Any]:
        """
        This function is intended for iterators to access
        and iterate through data inside linked list.
        >>> linked_list = LinkedList()
        >>> linked_list.insert_tail("tail")
        >>> linked_list.insert_tail("tail_1")
        >>> linked_list.insert_tail("tail_2")
        >>> for node in linked_list: # __iter__ used here.
        ...     node
        'tail'
        'tail_1'
        'tail_2'
        """
        node = self.head
        while node:
            yield node.val
            node = node.next

    def __repr__(self) -> str:
        """
        String representation/visualization of a Linked Lists
        >>> linked_list = LinkedList()
        >>> linked_list.insert_tail(1)
        >>> linked_list.insert_tail(3)
        >>> linked_list.__repr__()
        '1 -> 3'
        >>> repr(linked_list)
        '1 -> 3'
        >>> str(linked_list)
        '1 -> 3'
        >>> linked_list.insert_tail(5)
        >>> f"{linked_list}"
        '1 -> 3 -> 5'
        """
        return " -> ".join([str(item) for item in self])

    def add_first(self, val):
        self.head = ListNode(val, self.head)
        if self.tail is None:
            self.tail = self.head
        self.size += 1

    def add_last(self, val):
        self.tail.next = ListNode(val)
        self.tail = self.tail.next
        self.size += 1

    def remove_first(self):
        item = self.head.val
        self.head = self.head.next
        self.size -= 1
        return item

    def remove_last(self):
        if self.head is self.tail:
            return self.remove_first()
        else:
            curr = self.head
            while curr.next is not self.tail:
                curr = curr.next
            item = self.tail.val
            self.tail = curr
            self.tail.next = None
            return item

    def is_empty(self):
        return self.size == 0


one = ListNode(1)
two = ListNode(2)
three = ListNode(3)
one.next = two
two.next = three
head = one

print(head)
print(two)
print(three)

list = SinglyLinkedList()
list.add_first(1)
list.add_last(2)
print(list)

removed = list.remove_last()
print(removed)
print(list)
