"""
146 - LRU Cache [Medium]
OOP | Design | Hash Table
Time: O(1) for both get and put | Space: O(n) where n is the cache capacity

Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.

Implement the LRUCache class:
- LRUCache(int capacity): Initialize the LRU cache with positive size capacity.
- int get(int key): Return the value of the key if the key exists, otherwise return -1.
- void put(int key, int value): Update the value of the key if the key exists. Otherwise, add the key-value pair to the cache. If the number of keys exceeds the capacity from this operation, evict the least recently used key.

The functions get and put must each run in O(1) average time complexity.
"""

"""
According to Python docs:

Dictionaries preserve insertion order, meaning that keys will be produced in the 
same order they were added sequentially over the dictionary. Replacing an existing 
key does not change the order, however removing a key and re-inserting it will add 
it to the end instead of keeping its old place (since version 3.7).
"""


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}

    def get(self, key: int) -> int:
        if key in self.cache:
            val = self.cache.pop(key)
            self.cache[key] = val
            return val
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            del self.cache[key]

        elif len(self.cache) == self.capacity:
            self.cache.pop(next(iter(self.cache)))

        self.cache[key] = value
