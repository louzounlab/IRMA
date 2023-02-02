# -*- coding: utf-8 -*-
#
# priorityq: An object-oriented priority queue with updatable priorities.
#
# Copyright 2018 Edward L. Platt
#
# This software is released under multiple licenses. See the LICENSE file for
# more information.
#
# Authors:
#   Edward L. Platt <ed@elplatt.com>
"""Priority queue class with updatable priorities.
"""

import time

__all__ = ['MappedQueue', 'MaxQueue']

class MaxQueue(object):
    def __init__(self, data=None, max_size=-1):
        """Priority queue class with updatable priorities.
        The queue can be with fixed size, according to the parameter max_size.
        In case the length of the queue grow beyond max_size,
        the smallest elements (20% of all the elements) in the queue got deleted.
        For usual max-queue (without fixed size) you might use max_size = -1.
        """
        if data is None:
            data = []
        for i in range(len(data)):
            temp = data[i]
            data[i] = (-1*temp[0], temp[1])
        self.mapped_queue = MappedQueue(data)
        self.max_size = max_size

    def __len__(self):
        return len(self.mapped_queue)

    def isEmpty(self):
        return self.mapped_queue.isEmpty()

    def contains(self, key):
        return self.mapped_queue.contains(key)

    def addToPriority(self, add, key):
        self.mapped_queue.addToPriority(-1 * add, key)

    def value(self, key):
        temp = self.mapped_queue.value(key)
        return_value = (-1 * temp[0], temp[1])
        return return_value

    def push(self, priority, key):
        if self.max_size != -1 and len(self.mapped_queue) >= self.max_size:
            self.mapped_queue.remove_biggest_elements(int(self.max_size * 0.2))

        return self.mapped_queue.push(-1 * priority, key)

    def top(self):
        elt = self.mapped_queue.top()
        return_elt = (-elt[0], elt[1])
        return return_elt

    def pop(self):
        elt = self.mapped_queue.pop()
        return_elt = (-elt[0], elt[1])
        return return_elt

    def update(self, priority, key):
        self.mapped_queue.update(-1 * priority, key)

    def remove(self, key):
        self.mapped_queue.remove(key)

class MappedQueue(object):
    """The MappedQueue class implements an efficient minimum heap. The
    smallest element can be popped in O(1) time, new elements can be pushed
    in O(log n) time, and any element can be removed or updated in O(log n)
    time. The queue cannot contain duplicate elements and an attempt to push an
    element already in the queue will have no effect.
    MappedQueue complements the heapq package from the python standard
    library. While MappedQueue is designed for maximum compatibility with
    heapq, it has slightly different functionality.
    Examples
    --------
    A `MappedQueue` can be created empty or optionally given an array of
    initial elements. Calling `push()` will add an element and calling `pop()`
    will remove and return the smallest element.
    >>> q = MappedQueue([916, 50, 4609, 493, 237])
    >>> q.push(1310)
    True
    >>> x = [q.pop() for i in range(len(q.h))]
    >>> x
    [50, 237, 493, 916, 1310, 4609]
    Elements can also be updated or removed from anywhere in the queue.
    >>> q = MappedQueue([916, 50, 4609, 493, 237])
    >>> q.remove(493)
    >>> q.update(237, 1117)
    >>> x = [q.pop() for i in range(len(q.h))]
    >>> x
    [50, 916, 1117, 4609]
    References
    ----------
    .. [1] Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2001).
       Introduction to algorithms second edition.
    .. [2] Knuth, D. E. (1997). The art of computer programming (Vol. 3).
       Pearson Education.
    """

    def __init__(self, data=None):
        """Priority queue class with updatable priorities.
        """
        if data is None:
            data = []
        self.h = list(data)
        self.d = dict()
        self._heapify()

    def __len__(self):
        return len(self.h)

    def _heapify(self):
        """Restore heap invariant and recalculate map."""
        self.heapify()
        self.d = dict([(elt[1], pos) for pos, elt in enumerate(self.h)])
        if len(self.h) != len(self.d):
            raise AssertionError("Heap contains duplicate elements")

    def isEmpty(self):
        return len(self.h) == 0

    def contains(self, key):
        return key in self.d

    def addToPriority(self, add, key):
        if key not in self.d:
            self.push(add, key)
        else:
            priority = self.h[self.d[key]][0]
            newPriority = add + priority # logicly it should be (add - priority) but priority multiplied by -1 when pushed.
            self.update(newPriority, key)

    def value(self, key):
        return self.h[self.d[key]]

    def push(self, priority, key):
        # priority = -priority
        elt = (priority, key)
        """Add an element to the queue."""
        # If element is already in queue, do nothing
        if elt[1] in self.d:
            return False
        # Add element to heap and dict
        pos = len(self.h)
        self.h.append(elt)
        self.d[elt[1]] = pos
        # Restore invariant by sifting down
        self._siftdown(pos)
        return True

    def top(self):
        elt = self.h[0]
        # returnElt = (-elt[0], elt[1])
        return elt

    def pop(self):
        """Remove and return the smallest element in the queue."""
        # Remove smallest element
        elt = self.h[0]
        # returnElt = (-elt[0], elt[1])
        del self.d[elt[1]]
        # If elt is last item, remove and return
        if len(self.h) == 1:
            self.h.pop()
            return elt   # reverse multiply priority by -1
        # Replace root with last element
        last = self.h.pop()
        self.h[0] = last
        self.d[last[1]] = 0
        # Restore invariant by sifting up, then down
        pos = self._siftup(0)
        self._siftdown(pos)
        # Return smallest element
        return elt    # reverse multiply priority by -1

    def update(self, priority, key):
        if key not in self.d:
            self.push(priority, key)
        else:
            # Mpriority = -priority
            new = (priority, key)
            """Replace an element in the queue with a new one."""
            # Replace
            pos = self.d[new[1]]
            self.h[pos] = new
            self.d[new[1]] = pos
            # Restore invariant by sifting up, then down
            pos = self._siftup(pos)
            self._siftdown(pos)

    def remove(self, key):
        """Remove an element from the queue."""
        # Find and remove element
        try:
            pos = self.d[key]
            del self.d[key]
        except KeyError:
            # Not in queue
            raise
        # If elt is last item, remove and return
        if pos == len(self.h) - 1:
            self.h.pop()
            return
        # Replace elt with last element
        last = self.h.pop()
        self.h[pos] = last
        self.d[last[1]] = pos
        # Restore invariant by sifting up, then down
        pos = self._siftup(pos)
        self._siftdown(pos)

    def remove_biggest_elements(self, num: int):
        start = time.time()
        self.h.sort(key=lambda x: x[0])
        pos = len(self.h) - num
        if pos > 0:
            del self.h[pos:]

        del self.d
        self.d = {key: index for index, (_, key) in enumerate(self.h)}
        print(f"--- clean heap: {time.time() - start}")

    def heapify(self):
        """Transform list into a heap, in-place, in O(len(x)) time."""
        n = len(self.h)
        # Transform bottom-up.  The largest index there's any point to looking at
        # is the largest with a child index in-range, so must have 2*i + 1 < n,
        # or i < (n-1)/2.  If n is even = 2*j, this is (2*j-1)/2 = j-1/2 so
        # j-1 is the largest, which is n//2 - 1.  If n is odd = 2*j+1, this is
        # (2*j+1-1)/2 = j so j-1 is the largest, and that's again n//2-1.
        for i in reversed(range(n // 2)):
            self._siftup_(i)

    def _siftup(self, pos):
        """Move element at pos down to a leaf by repeatedly moving the smaller
        child up."""
        h, d = self.h, self.d
        elt = h[pos]
        # Continue until element is in a leaf
        end_pos = len(h)
        left_pos = (pos << 1) + 1
        while left_pos < end_pos:
            # Left child is guaranteed to exist by loop predicate
            left = h[left_pos]
            try:
                right_pos = left_pos + 1
                right = h[right_pos]
                # Out-of-place, swap with left unless right is smaller
                if right[0] < left[0]:
                    h[pos], h[right_pos] = right, elt
                    pos, right_pos = right_pos, pos
                    d[elt[1]], d[right[1]] = pos, right_pos
                else:
                    h[pos], h[left_pos] = left, elt
                    pos, left_pos = left_pos, pos
                    d[elt[1]], d[left[1]] = pos, left_pos
            except IndexError:
                # Left leaf is the end of the heap, swap
                h[pos], h[left_pos] = left, elt
                pos, left_pos = left_pos, pos
                d[elt[1]], d[left[1]] = pos, left_pos
            # Update left_pos
            left_pos = (pos << 1) + 1
        return pos

    def _siftdown(self, pos):
        """Restore invariant by repeatedly replacing out-of-place element with
        its parent."""
        h, d = self.h, self.d
        elt = h[pos]
        # Continue until element is at root
        while pos > 0:
            parent_pos = (pos - 1) >> 1
            parent = h[parent_pos]
            if parent[0] > elt[0]:
                # Swap out-of-place element with parent
                h[parent_pos], h[pos] = elt, parent
                parent_pos, pos = pos, parent_pos
                d[elt[1]] = pos
                d[parent[1]] = parent_pos
            else:
                # Invariant is satisfied
                break
        return pos

    def _siftup_(self, pos):
        """Restore invariant by repeatedly replacing out-of-place element with
        its parent."""
        h, d = self.h, self.d
        elt = h[pos]
        # Continue until element is at root
        while pos < len(h)-1:
            son1_pos = pos*2+1
            son2_pos = pos*2+2
            min_son_pos = 0
            if son1_pos >= len(h):
                break
            if son2_pos >= len(h) or h[son2_pos][0] > h[son1_pos][0]:
                min_son_pos = son1_pos
            else:
                min_son_pos = son2_pos

            if h[min_son_pos][0] < h[pos][0]:
                son = h[min_son_pos]
                h[pos], h[min_son_pos] = h[min_son_pos], h[pos]
                d[elt[1]], d[son[1]] = min_son_pos, pos
                pos = min_son_pos
                elt = h[pos]
            else:
                # Invariant is satisfied
                break
        return pos


