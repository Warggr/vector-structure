import pytest

from vector_structure import VectorStructure

# --- basic construction -------------------------------------------------------


def test_size_and_slices():
    vs = VectorStructure([("x", 3), ("u", 2), ("s", 3)])

    assert vs.size == 8
    assert vs["x"] == slice(0, 3)
    assert vs["u"] == slice(3, 5)
    assert vs["s"] == slice(5, 8)


# --- indexing -----------------------------------------------------------------


def test_single_block_access():
    vs = VectorStructure([("x", 3), ("u", 2)])

    z = list(range(vs.size))

    assert z[vs["x"]] == [0, 1, 2]
    assert z[vs["u"]] == [3, 4]


def test_multiple_block_access_contiguous():
    vs = VectorStructure([("x", 3), ("u", 2), ("s", 3)])

    s = vs["x", "u"]
    assert s == slice(0, 5)


def test_multiple_block_access_noncontiguous_raises():
    vs = VectorStructure([("x", 3), ("u", 2), ("s", 3)])

    with pytest.raises(AssertionError):
        _ = vs["x", "s"]


# --- slice syntax -------------------------------------------------------------


def test_slice_syntax():
    vs = VectorStructure([("x", 3), ("u", 2), ("s", 3)])

    s = vs["x":"u"]
    assert s == slice(0, 5)


# --- invalid access -----------------------------------------------------------


def test_invalid_key_raises():
    vs = VectorStructure([("x", 3), ("u", 2)])

    with pytest.raises(KeyError):
        _ = vs["invalid"]


def test_invalid_sequence_type_raises():
    vs = VectorStructure([("x", 3), ("u", 2)])

    with pytest.raises(KeyError):
        _ = vs[123]  # not a valid key or sequence


def test_as_dict():
    vs = VectorStructure([("x", 3), ("u", 2)])
    li = [1, 2, 3, 4, 5]
    cuts = vs.as_dict(li)
    assert cuts["x"] == [1, 2, 3]
    assert cuts["u"] == [4, 5]


def test_block_size():
    vs = VectorStructure([("x", 3), ("u", 2)])
    assert vs.block_size("x") == 3
    assert vs.block_size("u") == 2
    assert vs.block_size(("x", "u")) == 5


# --- numpy integration --------------------------------------------------------


def test_numpy_assignment():
    np = pytest.importorskip("numpy")
    vs = VectorStructure([("x", 3), ("u", 2)])

    z = np.zeros(vs.size)

    z[vs["x"]] = np.array([1, 2, 3])
    z[vs["u"]] = 5

    assert np.all(z == np.array([1, 2, 3, 5, 5]))


# --- dynamic structure --------------------------------------------------------


def test_optional_block():
    base = [("x", 3), ("u", 2)]

    vs1 = VectorStructure(base)
    vs2 = VectorStructure(base + [("s", 3)])

    assert vs1.size == 5
    assert vs2.size == 8

    assert "s" not in vs1.cuts
    assert vs2["s"] == slice(5, 8)


# --- edge cases ---------------------------------------------------------------


def test_empty_structure():
    vs = VectorStructure([])

    assert vs.size == 0
    assert vs.cuts == {}


def test_single_element_blocks():
    vs = VectorStructure([("a", 1), ("b", 1), ("c", 1)])

    assert vs["a"] == slice(0, 1)
    assert vs["b"] == slice(1, 2)
    assert vs["c"] == slice(2, 3)


def test_order_is_respected():
    vs = VectorStructure([("u", 2), ("x", 3)])

    assert vs["u"] == slice(0, 2)
    assert vs["x"] == slice(2, 5)


def get_separators(st, colon="|"):
    return map(len, st.split(colon))


def test_repr_is_sensibly_formatted():
    for make_key in [
        lambda name: name,
        lambda name: (name, 1),
    ]:
        vs = VectorStructure(
            [
                (make_key("a very long name!"), 2),
                (make_key("z"), 2),
                (make_key("x"), 100_000),
            ]
        )
        r = repr(vs)
        header, nums = r.split("\n")
        header, nums = tuple(get_separators(header)), tuple(get_separators(nums))
        assert header == nums
