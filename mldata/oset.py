import collections

SLICE_ALL = slice(None)


class OrderedSet(collections.MutableSet):

    def __init__(self, iterable=None):
        self.end = end = []
        end += [None, end, end]  # sentinel node for doubly linked list
        self.map = {}            # key --> [key, prev, next]
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def discard(self, key):
        if key in self.map:
            key, prev, next = self.map.pop(key)
            prev[2] = next
            next[1] = prev

    def __iter__(self):
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __hash__(self):
        return hash(tuple(self))

    def __reversed__(self):
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def pop(self, last=True):
        if not self:
            raise KeyError('set is empty')
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)

    def __getitem__(self, index):
        """Get the item at a given index.

        If `index` is a slice, you will get back that slice of items. If it's
        the slice [:], a deep copy of the object is returned.

        If `index` is an iterable, you'll get the OrderedSet of items
        corresponding to those indices. This is similar to NumPy's
        "fancy indexing".
        """
        if isinstance(index, int):
            return list(self)[index]  # inefficient, but works for now
        elif index == SLICE_ALL:
            return OrderedSet(self.end)
        elif hasattr(index, '__index__') or isinstance(index, slice):
            return OrderedSet(self.end[index])
        elif is_iterable(index):
            return OrderedSet([self.end[i] for i in index])
        else:
            raise TypeError(
                "Don't know how to index an OrderedSet by %r" % index)

    @staticmethod
    def union_all(*sets):
        union = OrderedSet()
        union.union(*sets)
        return union

    def union(self, *sets):
        for set in sets:
            self |= set


if __name__ == '__main__':
    s = OrderedSet('abracadaba')
    t = OrderedSet('simsalabim')
    print(s | t)
    print(s & t)
    print(s - t)
