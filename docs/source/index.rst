Installation
------------

Can be installed using pip:

.. code-block:: console

   (.venv) $ pip install vector-structure

Usage
-----

Create a :py:class:`~vector_structure.VectorStructure` object:

>>> from vector_structure import VectorStructure
>>> x = VectorStructure([("x", 3), ("y", 1), ("z", 2)])

Index an indexable object (list, NumPy array, or similar) with it:

>>> li = [
...  1, 2, 3,  # x
...  4,        # y
...  5, 6      # z
... ]
>>> li[x["y"]]
[4]
>>> li[x[:"y"]]
[1, 2, 3, 4]
>>> li[x["y":"z"]]
[4, 5, 6]

Contents
--------

.. toctree::
   api
