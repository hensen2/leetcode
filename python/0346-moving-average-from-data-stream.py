"""
346 - Moving Average from Data Stream
Deque | OOP
Time: O(1) | Space: O(n)

Given a stream of integers and a window size, calculate the moving average of all integers in the sliding window.

Implement the MovingAverage class:

MovingAverage(int size) Initializes the object with the size of the window size.
double next(int val) Returns the moving average of the last size values of the stream.
"""

from collections import deque

class MovingAverage:

    def __init__(self, size: int):
        self.queue = deque(maxlen=size)
        self.sum = 0

    def next(self, val: int) -> float:
        if len(self.queue) == self.queue.maxlen:
            self.sum -= self.queue[0]

        self.queue.append(val)
        self.sum += val

        return self.sum / len(self.queue)


# Your MovingAverage object will be instantiated and called as such:
# obj = MovingAverage(size)
# param_1 = obj.next(val)