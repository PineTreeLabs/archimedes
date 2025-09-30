---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: archimedes
---

```{code-cell} python
:tags: [hide-cell]
# ruff: noqa: N802, N803, N806, N815, N816
import os

from utils import display_text

import archimedes as arc
```

# Quickstart

This section will walk through a "Hello, World!" example of C code generation and usage without getting into the details of the structure of the generated code.
The remaining sections will expand on this same basic workflow so that you can build a deeper understanding of what is generated, why, and how to use it effectively.

## Python implementation

To keep this as simple as possible, we'll work with a Python implementation of the classic Fibonnaci sequence:


```{code-cell} python
def fib(a, b):
    return b, a + b


# Generate the first 10 Fibonacci numbers
a, b = 0, 1
for _ in range(10):
    a, b = fib(a, b)
    print(a)
```

## Converting to C code

Next we generate a C implementation of the Python logic using the [`codegen`](#archimedes.codegen) function, including initial values of the inputs.  Note that we have to provide names for the output variables so that the generated code can use meaningful names.


```{code-cell} python
# Create "template" arguments for type inference
# and initialization
a, b = 0, 1

arc.codegen(fib, (a, b), return_names=("a_new", "b_new"))
```

This will generate several files.  For our purposes in this quick start, the only one we need to look at is `fib.h`.


```{code-cell} python
with open("fib.h", "r") as f:
    c_code = f.read()

display_text(c_code)
```

## Using the generated code

The basic structure of this API, which is the same for _all_ generated code, is that there are specific structs to hold the input data (`arg`), output data (`res`) and preallocated working memory (`work`).  There are also two functions: an `_init` function to initialize the data (this will also use the "template" values we provided earlier), and a `_step` function that executes the code and stores the results in the `res` struct.

Here's a minimal `main.c` "application" that shows how to use this API to generate the same results we saw from Python:


```{code-cell} python
with open("main.c", "r") as f:
    c_code = f.read()

display_text(c_code)
```


```{code-cell} python
# Compile and run the C application
os.system("gcc -o main main.c fib.c fib_kernel.c")
os.system("./main > output.txt")

with open("output.txt", "r") as f:
    output = f.read()

print(output)
```

That's all there is to it.  The generated C code is self-contained, portable, and efficient, so it can be used in a standalone C application like this, called with Cython bindings for speed, or (most importantly) deployed to a wide variety of embedded controllers.

Next we will delve into some of the details of the generated code and work through a more practical example: digital filtering.
