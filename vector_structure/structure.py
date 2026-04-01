from typing import Generic, Protocol, Sequence, SupportsIndex, TypeVar, overload


def simple_slice_len(sl: slice):
    """Compute the size of a 'simple' slice.

    A simple slice is one with step 1 and no negative indices.
    VectorStructure[...] always returns simple slices.
    This function is just syntactic sugar for sl.stop - sl.start.
    """
    if sl.step not in (None, 1):
        raise ValueError("Only slices with step 1 supported")
    if sl.start < 0 or sl.stop < 0:
        raise ValueError("Only slices with positive indices supported")
    return sl.stop - sl.start


T = TypeVar("T", covariant=True)


class ArrayLike(Protocol, Generic[T]):
    """A list-like container that can be used with VectorStructure.

    At least `list` and `np.ndarray` follow this protocol.
    """

    @overload
    def __getitem__(self, idxs: SupportsIndex, /) -> T: ...
    @overload
    def __getitem__(self, idxs: slice, /) -> "ArrayLike[T]": ...
    def __len__(self) -> int: ...


BlockName = TypeVar("BlockName")
SpecificArrayLike = TypeVar("SpecificArrayLike", bound=ArrayLike)


class VectorStructure(Generic[BlockName]):
    """Helper class to describe block vectors and block matrices."""

    def __init__(self, sizes: Sequence[tuple[BlockName, int]]):
        cuts = {}
        start = 0
        for k, v in sizes:
            cuts[k] = slice(start, start + v)
            start += v
        self.size = start
        self.cuts = cuts

    def as_dict(self, arr: SpecificArrayLike) -> dict[BlockName, SpecificArrayLike]:
        """Split the array into pieces according to the vector structure."""
        if len(arr) != self.size:
            raise ValueError(
                f"Size mismatch: Array structure is {self.size}, array is {len(arr)}"
            )
        return {key: arr[sl] for key, sl in self.cuts.items()}

    def block_size(self, idx: BlockName | tuple[BlockName, ...] | slice) -> int:
        """Return the size of one or more blocks."""
        return simple_slice_len(self[idx])

    def __getitem__(self, idx: BlockName | tuple[BlockName, ...] | slice) -> slice:
        """Get the indices for one or multiple blocks.

        Note: when indexing by slice, both the start and the stop are included,
        i.e. a["a":"c"] = a["a", "b", "c"].
        This is different from the Python convention, but similar to Pandas.
        """
        if isinstance(idx, slice):
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
                    stop = cut.stop
                    break
            if start is None or stop is None:
                raise ValueError(f"{idx.start} or {idx.stop} not found")
            return slice(start, stop)
        elif isinstance(idx, tuple):
            first, *more = idx
            start = self.cuts[first].start
            stop = self.cuts[first].stop
            for idx in more:
                next_cut = self.cuts[idx]
                assert next_cut.start == stop
                stop = next_cut.stop
            return slice(start, stop)
        else:
            return self.cuts[idx]
