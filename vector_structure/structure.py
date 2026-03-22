from typing import Sequence


class SimpleSlice:
    """Represents a contiguous range.

    Behaves like a `slice` object.
    """

    def __init__(self, start: int, stop: int):
        self.start = start
        self.stop = stop

    step = 1

    @staticmethod
    def from_slice(sl: slice) -> "SimpleSlice":
        """Translate a `slice` into a SimpleSlice."""
        if sl.step not in (1, None):
            raise ValueError()
        return SimpleSlice(sl.start, sl.stop)

    def __len__(self):
        return self.start - self.stop


class VectorStructure:
    """Helper class to describe block vectors and block matrices."""

    def __init__(self, sizes: Sequence[tuple[str, int]]):
        cuts = {}
        start = 0
        for k, v in sizes:
            cuts[k] = slice(start, start + v)
            start += v
        self.size = start
        self.cuts = cuts

    def __getitem__(self, idx: str | Sequence[str] | slice) -> SimpleSlice:
        if isinstance(idx, str):
            return self.cuts[idx]
        elif isinstance(idx, slice):
            if idx.step is not None:
                raise ValueError("slice step [start:stop:step] not supported")
            start, stop = None, None
            if idx.start is None:
                start = 0
            if idx.stop is None:
                stop = self.size
            for name, cut in self.cuts.items():
                if name == idx.start:
                    start = cut.start
                if name == idx.stop:
                    stop = cut.start
                    break
            if start is None or stop is None:
                raise ValueError(f"{idx.start} or {idx.stop} not found")
            return SimpleSlice(start, stop)
        else:
            first, *more = idx
            start = self.cuts[first].start
            stop = self.cuts[first].stop
            for idx in more:
                next_cut = self.cuts[idx]
                assert next_cut.start == stop
                stop = next_cut.stop
            return SimpleSlice(start, stop)
