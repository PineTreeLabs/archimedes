"""
This file contains code modified from pybaum
Original Copyright (c) 2022 Janoś Gabler, Tobias Raabe
https://github.com/OpenSourceEconomics/pybaum


The original code has been modified throughout. While the original code remains
under its MIT license, all modifications and additions are:
Copyright (c) 2025 Jared Callaham
Licensed under the GNU General Public License v3.0

As a combined work, use of this code requires compliance with the GNU GPL v3.0.
The original MIT license terms are included below for attribution:

MIT LICENSE

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
"""

from .registry import (
    register_pytree_node,
    register_dataclass,
)

from .tree_util import (
    tree_flatten as flatten,
    tree_unflatten as unflatten,
    tree_structure as structure,
    tree_leaves as leaves,
    tree_map as map,
    tree_all as all,
    tree_reduce as reduce,
)

from .flatten_util import ravel_pytree as ravel

from . import struct

__all__ = [
    "register_pytree_node",
    "register_dataclass",
    "flatten",
    "unflatten",
    "structure",
    "leaves",
    "map",
    "all",
    "reduce",
    "ravel",
    "struct",
]