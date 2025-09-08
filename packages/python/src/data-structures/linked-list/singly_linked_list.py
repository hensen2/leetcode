class ListNode:
    def __init__(self, data, next=None):
        self.data = data
        self.next = next

    def __repr__(self):
        return f"Node({self.data}, {repr(self.next)})"


class SinglyLinkedList:
    def __init__(self):
        self.head = self.Node(None)
        self.tail = self.head
        self.size = 0

    def add_first(self, e):
        new_node = self.Node(e)
        new_node.next = self.head.next
        self.head.next = new_node
        if self.size == 0:
            self.tail = new_node
        self.size += 1

    def add_last(self, e):
        new_node = self.Node(e)
        self.tail.next = new_node
        self.tail = new_node
        self.size += 1

    def add(self, index, element):
        self.check_position_index(index)

        if index == self.size:
            self.add_last(element)
            return

        prev = self.head
        for i in range(index):
            prev = prev.next
        new_node = self.Node(element)
        new_node.next = prev.next
        prev.next = new_node
        self.size += 1

    def remove_first(self):
        if self.is_empty():
            raise Exception("NoSuchElementException")
        first = self.head.next
        self.head.next = first.next
        if self.size == 1:
            self.tail = self.head
        self.size -= 1
        return first.val

    def remove_last(self):
        if self.is_empty():
            raise Exception("NoSuchElementException")

        prev = self.head
        while prev.next != self.tail:
            prev = prev.next
        val = self.tail.val
        prev.next = None
        self.tail = prev
        self.size -= 1
        return val

    def remove(self, index):
        self.check_element_index(index)

        prev = self.head
        for i in range(index):
            prev = prev.next

        node_to_remove = prev.next
        prev.next = node_to_remove.next
        # deleting the last element
        if index == self.size - 1:
            self.tail = prev
        self.size -= 1
        return node_to_remove.val

    # ***** Retrieve *****

    def get_first(self):
        if self.is_empty():
            raise Exception("NoSuchElementException")
        return self.head.next.val

    def get_last(self):
        if self.is_empty():
            raise Exception("NoSuchElementException")
        return self.tail.val

    def get(self, index):
        self.check_element_index(index)
        p = self.get_node(index)
        return p.val

    # ***** Update *****

    def set(self, index, element):
        self.check_element_index(index)
        p = self.get_node(index)

        old_val = p.val
        p.val = element

        return old_val

    # ***** Other Utility Functions *****
    def get_size(self):
        return self.size

    def is_empty(self):
        return self.size == 0

    def is_element_index(self, index):
        return 0 <= index < self.size

    def is_position_index(self, index):
        return 0 <= index <= self.size

    # Check if the index position can have an element
    def check_element_index(self, index):
        if not self.is_element_index(index):
            raise IndexError(f"Index: {index}, Size: {self.size}")

    # Check if the index position can add an element
    def check_position_index(self, index):
        if not self.is_position_index(index):
            raise IndexError(f"Index: {index}, Size: {self.size}")

    # Return the Node corresponding to the index
    # Note: Please ensure that the passed index is valid
    def get_node(self, index):
        p = self.head.next
        for i in range(index):
            p = p.next
        return p


if __name__ == "__main__":
    list = SinglyLinkedList()
    list.add_first(1)
    list.add_first(2)
    list.add_last(3)
    list.add_last(4)
    list.add(2, 5)

    print(list.remove_first())  # 2
    print(list.remove_last())  # 4
    print(list.remove(1))  # 5

    print(list.get_first())  # 1
    print(list.get_last())  # 3
    print(list.get(1))  # 3
