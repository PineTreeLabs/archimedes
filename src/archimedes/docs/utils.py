import ast
import re

try:
    from IPython.display import Markdown, display
except ImportError:
    raise ImportError("IPython is required to use these utilities")


__all__ = ["display_text", "extract_c_function"]


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


def extract_py_function(
    filename: str,
    function_name: str,
    show: bool = True,
    include_decorators: bool = True,
) -> str | None:
    """
    Extract a specific function from a Python file using AST parsing

    Args:
        filename: Path to the Python file
        function_name: Name of the function to extract
        show: Whether to display the result
        include_decorators: Whether to include decorators in the extraction

    Returns:
        The extracted function as a string, or None if not found
    """
    return _extract_py_definition(
        filename, function_name, "function", show, include_decorators
    )


def extract_py_class(
    filename: str, class_name: str, show: bool = True, include_decorators: bool = True
) -> str | None:
    """
    Extract a specific class from a Python file using AST parsing

    Args:
        filename: Path to the Python file
        class_name: Name of the class to extract
        show: Whether to display the result
        include_decorators: Whether to include decorators in the extraction

    Returns:
        The extracted class as a string, or None if not found
    """
    return _extract_py_definition(
        filename, class_name, "class", show, include_decorators
    )


def _extract_py_definition(
    filename: str,
    name: str,
    definition_type: str = "function",
    show: bool = True,
    include_decorators: bool = True,
) -> str | None:
    """
    Extract a function or class from a Python file using AST parsing

    Args:
        filename: Path to the Python file
        name: Name of the function/class to extract
        definition_type: Either "function" or "class"
        show: Whether to display the result
        include_decorators: Whether to include decorators in the extraction

    Returns:
        The extracted code as a string, or None if not found
    """
    try:
        with open(filename, "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source)
        lines = source.splitlines()

        target_node_type = (
            ast.FunctionDef if definition_type == "function" else ast.ClassDef
        )

        for node in ast.walk(tree):
            if isinstance(node, target_node_type) and node.name == name:
                start_line = node.lineno - 1  # AST uses 1-based indexing

                # Include decorators if requested
                if include_decorators and node.decorator_list:
                    decorator_start = min(d.lineno for d in node.decorator_list) - 1
                    start_line = decorator_start

                # Try to get end line (available in Python 3.8+)
                if hasattr(node, "end_lineno") and node.end_lineno is not None:
                    end_line = node.end_lineno
                else:
                    # Error for older Python versions
                    raise SyntaxError(
                        "Python version must be 3.8 or higher to extract definitions "
                        "without end_lineno."
                    )

                extracted_code = "\n".join(lines[start_line:end_line])
                return display_text(extracted_code, show=show, language="python")

        print(f"{definition_type.capitalize()} '{name}' not found in '{filename}'")
        return None

    except FileNotFoundError:
        print(f"File '{filename}' not found")
        return None
    except SyntaxError as e:
        print(f"Syntax error in '{filename}': {e}")
        return None
