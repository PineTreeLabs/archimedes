"""Simple utility functions for demonstrating code generation"""

import os
import re

from IPython.display import Markdown, display


def cleanup():
    for filename in (
        "main",
        "iir_filter.c",
        "iir_filter.h",
        "iir_filter_kernel.c",
        "iir_filter_kernel.h",
        "fib.c",
        "fib.h",
        "fib_kernel.c",
        "fib_kernel.h",
        "output.txt",
        "compound_filter.c",
        "compound_filter.h",
        "compound_filter_kernel.c",
        "compound_filter_kernel.h",
    ):
        if os.path.exists(filename):
            os.remove(filename)


def display_text(content, show=True, language="c"):
    md = Markdown(f"```{language}\n{content}\n```")

    if show:
        display(md)
    else:
        return md


def extract_c_function(filename, function_name, show=True):
    """
    Extract a specific function from a C file using regex

    Args:
        filename: Path to the C file
        function_name: Name of the function to extract

    Returns:
        The extracted function as a string, or None if not found
    """
    try:
        with open(filename, "r") as f:
            source = f.read()

        # Pattern to match function definition
        pattern = re.compile(
            r".*" + re.escape(function_name) + r"\s*\([^)]*\)\s*{", re.MULTILINE
        )

        match = pattern.search(source)
        if not match:
            print(f"Function '{function_name}' not found in '{filename}'")
            return None

        # Find the start position of the function
        start_pos = match.start()

        # Count braces to find the end of the function
        brace_count = 0
        in_function = False
        end_pos = start_pos

        for i in range(start_pos, len(source)):
            if source[i] == "{":
                in_function = True
                brace_count += 1
            elif source[i] == "}":
                brace_count -= 1
                if in_function and brace_count == 0:
                    end_pos = i + 1
                    break

        # Extract the function code
        extracted_code = source[start_pos:end_pos]

        # Display in a markdown cell
        return display_text(extracted_code, show=show, language="c")

    except FileNotFoundError:
        print(f"File '{filename}' not found")
        return None
