import os
import shutil
import tempfile
from typing import NamedTuple

import numpy as np
import pytest

import archimedes as arc
from archimedes._core._codegen._renderer import (
    ArduinoRenderer,
    _extract_protected_regions,
    _render_template,
)

# TODO:
# - Test data type explicit specification
# - Test data type inference
# - Test with static args
# - Test re-importing with casadi extern


@pytest.fixture
def temp_dir():
    """Create a temporary directory and file path for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def func(x, y):
    return x, np.sin(y) * x


@pytest.fixture()
def scalar_func():
    return func


# Create arrays of the right shape and dtype
x_type = np.array([1, 2], dtype=float)
y_type = np.array(3, dtype=float)


def compare_files(expected_file, output_dir):
    expected_output = os.path.join(
        os.path.dirname(__file__),
        f"fixtures/{expected_file}",
    )

    # Load expected output
    with open(expected_output, "r") as f:
        expected = f.read()

    # Load actual output
    with open(output_dir, "r") as f:
        actual = f.read()

    print("#### ACTUAL ####")
    print(actual)

    # Compare (normalize whitespace to handle line endings)
    assert expected.strip() == actual.strip()


def check_in_file(file, pattern):
    with open(file, "r") as f:
        content = f.read()
        assert pattern in content


class TestCodegen:
    def _check_files(self, temp_dir, func_name):
        # Check that the files were created
        assert os.path.exists(f"{temp_dir}/{func_name}_kernel.c")
        assert os.path.exists(f"{temp_dir}/{func_name}_kernel.h")
        assert os.path.exists(f"{temp_dir}/{func_name}.c")
        assert os.path.exists(f"{temp_dir}/{func_name}.h")

        # Compare to expected output
        compare_files(f"{func_name}.h", f"{temp_dir}/{func_name}.h")
        compare_files(f"{func_name}.c", f"{temp_dir}/{func_name}.c")

    def test_basic_codegen(self, temp_dir, scalar_func):
        kwargs = {
            "float_type": np.float32,
            "output_dir": temp_dir,
        }

        # Generate code for the function.  This iteration will also compile
        arc.codegen(
            scalar_func, (x_type, y_type), return_names=("x_new", "z"), **kwargs
        )
        self._check_files(temp_dir, "func")

        # Pre-compile the function for the remaining tests
        func = arc.compile(scalar_func, name="func", return_names=("x_new", "z"))

        # Check that the kernel header includes a proper function signature
        check_in_file(f"{temp_dir}/{func.name}_kernel.h", "int func")

        # Check that the API header includes correct functions
        check_in_file(f"{temp_dir}/{func.name}.h", "int func_init")
        check_in_file(f"{temp_dir}/{func.name}.h", "int func_step")

        c_file = f"{func.name}.c"
        h_file = f"{func.name}.h"

        # Run with integer arguments to check type conversion
        # Because CasADi treats everything as a float, this will get
        # converted to float in the generated code.
        y_type_int = np.array(3, dtype=int)
        arc.codegen(func, (x_type, y_type_int), **kwargs)
        check_in_file(f"{temp_dir}/{h_file}", "float y;")
        check_in_file(f"{temp_dir}/{c_file}", "arg->y = 3.000000f;")

        # Run with a specified kwarg, also test for double support
        kwargs["kwargs"] = {"y": 5.0}
        kwargs["float_type"] = np.float64
        arc.codegen(func, (x_type,), **kwargs)
        check_in_file(f"{temp_dir}/{h_file}", "double y;")
        check_in_file(f"{temp_dir}/{c_file}", "arg->y = 5.0;")

    def test_nested_codegen(self, temp_dir):
        kwargs = {
            "float_type": np.float32,
            "output_dir": temp_dir,
        }

        @arc.struct
        class EmptyStruct:
            arr: np.ndarray  # Empty array
            static_field: str = arc.field(static=True, default="static")

        class Point(NamedTuple):
            x: float
            y: float

        @arc.struct
        class Cluster:
            center: Point
            points: list[Point]  # Array of structs
            weights: np.ndarray  # Simple array

        def nested_func(
            scalar: float,
            arr: np.ndarray,
            clusters: list[
                Cluster
            ],  # array of structs containing arrays of named tuples
            empty_struct: EmptyStruct,
            static_str: str = "default",  # Static arg should be ignored
        ) -> float:
            return scalar + clusters[0].points[1].x

        # Pre-compile the function
        func = arc.compile(
            nested_func,
            name=nested_func.__name__,
            return_names=("z",),
            static_argnames=("static_str",),
        )

        # Initial arguments
        args = (
            42.0,  # scalar
            np.array([1.0, 2.0, 3.0]),  # arr
            [
                Cluster(
                    center=Point(1.0, 2.0),
                    points=[
                        Point(1.0, 2.0),
                        Point(2.0, 3.0),
                        Point(3.0, 4.0),
                    ],
                    weights=np.array([0.1, 0.2, 0.3]),
                ),
                Cluster(
                    center=Point(4.0, 5.0),
                    points=[
                        Point(4.0, 5.0),
                        Point(5.0, 6.0),
                    ],
                    weights=np.array([0.4, 0.5, 0.6]),
                ),
            ],
            EmptyStruct(arr=np.array([])),  # empty struct
        )
        # First, raise an error (second Cluster only has two points)
        with pytest.raises(
            ValueError,
            match="All items in list 'clusters' must have the same structure.",
        ):
            arc.codegen(func, args, **kwargs)

        # Fix the error and generate code
        args[2][1].points.append(Point(6.0, 7.0))  # Fix by adding a point
        arc.codegen(func, args, **kwargs, debug=True)
        self._check_files(temp_dir, func.name)

    def test_dict_codegen(self, temp_dir):
        kwargs = {
            "float_type": np.float32,
            "output_dir": temp_dir,
        }

        def dict_func(
            config: dict[str, float],  # {"lr": 0.01, "momentum": 0.9}
            bounds: tuple[float, float],  # (0.0, 1.0)
            empty_dict: dict[str, float],  # {} - empty
            empty_list: list[float],  # [] - empty
            single_tuple: tuple[float],  # (42.0,) - single element
            none_arg: None,  # empty
        ) -> dict[str, float | None]:
            return {"result": config["lr"] + bounds[0], "none_res": None}

        # Pre-compile the function
        func = arc.compile(dict_func, name=dict_func.__name__, return_names=("output",))

        args = ({"lr": 0.01, "momentum": 0.9}, (0.0, 1.0), {}, [], (42.0,), None)

        arc.codegen(func, args, **kwargs, debug=True)
        self._check_files(temp_dir, func.name)

    def test_array_codegen(self, temp_dir):
        kwargs = {
            "float_type": np.float32,
            "output_dir": temp_dir,
        }

        @arc.struct
        class EdgeCase:
            empty: np.ndarray  # []
            single: np.ndarray  # [1]

        def array_func(
            zero_d: float,  # 0D array that becomes scalar
            one_d_single: np.ndarray,  # [1] array
            one_d_normal: np.ndarray,  # [5] array
            two_d_normal: np.ndarray,  # [[1,2],[3,4]] array
            edge_case: EdgeCase,
        ) -> float:
            return np.sum(two_d_normal), zero_d + one_d_single[0], edge_case

        # Pre-compile the function
        func = arc.compile(
            array_func,
            name=array_func.__name__,
            return_names=("sum", "z", "edge_out"),
        )

        args = (
            1.0,
            np.array([1.0]),
            np.arange(5),
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            EdgeCase(
                empty=np.array([]),
                single=np.array([1.0]),
            ),
        )

        arc.codegen(func, args, **kwargs, debug=True)
        self._check_files(temp_dir, func.name)

    def test_error_handling(self, scalar_func, temp_dir):
        # Test with unknown return names.  By design choice, this raises an error
        # in order to help generate more readable code.
        with pytest.raises(arc.CodegenError, match=r"Return names must be provided"):
            arc.codegen(scalar_func, (x_type, y_type))

        func = arc.compile(scalar_func, name="func", return_names=("x_new", "z"))

        # Can't use non-numeric inputs
        with pytest.raises(TypeError, match="data type 'string' not understood"):
            arc.codegen(func, (x_type, "string"))

        # Should be impossible to do this anyway, but for completeness, check that
        # the low-level codegen function raises an error if the target path is not
        # the CWD.
        specialized_func, _ = func._specialize(x_type, y_type)
        with pytest.raises(RuntimeError):
            specialized_func.codegen("invalid/func.c", {})

        # Invalid arg name (collision with `_t`)
        def invalid_arg_func(x_t):
            return x_t

        with pytest.raises(
            arc.CodegenError, match=r"Argument name 'x_t' cannot end with '_t'"
        ):
            arc.codegen(invalid_arg_func, (x_type,), return_names=("y",))

        os.remove("invalid_arg_func_kernel.c")
        os.remove("invalid_arg_func_kernel.h")


class TestExtractProtectedRegions:
    def test_basic_extraction(self):
        # Create a temporary file with test content
        content = """// Some code
    // PROTECTED-REGION-START: imports
    #include <stdlib.h>
    // PROTECTED-REGION-END
    // More code
    // PROTECTED-REGION-START: main
    printf("Hello World");
    // PROTECTED-REGION-END
    """
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp:
            temp.write(content)
            temp_path = temp.name

        try:
            # Test extraction functionality
            regions = _extract_protected_regions(temp_path)

            # Check results
            assert len(regions) == 2
            assert "imports" in regions
            assert "main" in regions
            assert "#include <stdlib.h>" in regions["imports"]
            assert 'printf("Hello World");' in regions["main"]

        finally:
            # Clean up
            os.unlink(temp_path)

    def test_nonexistent(self):
        filename = "nonexistent_file.c"
        regions = _extract_protected_regions(filename)
        assert regions == {}


@pytest.fixture
def context():
    # Basic context for consistent driver template rendering
    # Note this needs to match the test_func above
    return {
        "filename": "gen",
        "app_name": "test_app",
        "function_name": "test_func",
        "sample_rate": 0.01,
        "float_type": "float",
        "int_type": "int",
        "inputs": [
            {
                "type": "float",
                "name": "x",
                "dims": "2",
                "initial_value": "{1.0, 2.0}",
                "is_addr": False,
            },
            {
                "type": "float",
                "name": "y",
                "dims": None,
                "initial_value": "3.0",
                "is_addr": True,
            },
        ],
        "outputs": [
            {
                "type": "float",
                "name": "x_new",
                "dims": "2",
                "is_addr": False,
            },
            {
                "type": "float",
                "name": "z",
                "dims": "2",
                "is_addr": False,
            },
        ],
    }


class TestRender:
    @pytest.mark.parametrize(
        "template,expected_file",
        [
            ("c_app", "expected_c_app.c"),
            ("arduino", "expected_arduino.ino"),
        ],
    )
    def test_initial_render(self, template, expected_file, temp_dir, context):
        extension = expected_file.split(".")[-1]
        filename = f"{context['app_name']}.{extension}"
        output_path = os.path.join(temp_dir, filename)

        # Render the template
        _render_template(template, context, output_path=output_path)
        compare_files(expected_file, output_path)

    # @pytest.mark.parametrize("driver_type", ["c", "arduino"])
    # def test_default_output_path(self, driver_type, context):
    #     renderer = _render_template(driver_type, context, output_path=None)

    #     # Check that the file exists with the default name
    #     assert os.path.exists(renderer.default_output_path)
    #     os.remove(renderer.default_output_path)

    def test_invalid_renderer(self, context):
        with pytest.raises(ValueError, match=r"Template .* not found."):
            _render_template("invalid", context)

        with pytest.raises(ValueError, match=r"Template must be a .*"):
            _render_template(type(self), context)

    def test_direct_renderer(self, temp_dir, context):
        # Pass the Arduino renderer directly
        output_path = os.path.join(temp_dir, "sketch.ino")
        _render_template(ArduinoRenderer, context, output_path=output_path)

    def test_preserve_protected(self, temp_dir, context):
        output_path = os.path.join(temp_dir, f"{context['app_name']}.c")

        # Initial render
        _render_template("c_app", context, output_path=output_path)

        # Modify a protected region
        with open(output_path, "r") as f:
            content = f.read()

        # Insert custom code into a protected region
        modified_content = content.replace(
            "// PROTECTED-REGION-START: main\n",
            '// PROTECTED-REGION-START: main\n    printf("Custom code\\n");\n',
        )

        with open(output_path, "w") as f:
            f.write(modified_content)

        # Re-render with same context
        _render_template("c_app", context, output_path=output_path)

        # Verify protected region was preserved
        with open(output_path, "r") as f:
            final_content = f.read()

        assert 'printf("Custom code\\n");' in final_content

    def test_context_changes(self, temp_dir, context):
        output_path = os.path.join(temp_dir, f"{context['app_name']}.c")

        # Render with initial context
        _render_template("c_app", context, output_path=output_path)

        # Modify a protected region
        with open(output_path, "r") as f:
            content = f.read()

        modified_content = content.replace(
            "// PROTECTED-REGION-START: main\n",
            '// PROTECTED-REGION-START: main\n    printf("Custom code\\n");\n',
        )

        with open(output_path, "w") as f:
            f.write(modified_content)

        # Updated context with different function name
        context["function_name"] = "func2"

        # Re-render with new context
        _render_template("c_app", context, output_path=output_path)

        # Check results
        with open(output_path, "r") as f:
            final_content = f.read()

        # Protected region should be preserved
        assert 'printf("Custom code\\n");' in final_content

        # Function name should be updated
        assert "func2" in final_content
        assert "func1" not in final_content


if __name__ == "__main__":
    tmp_dir = "tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    # TestCodegen().test_dict_codegen(tmp_dir)
    # TestCodegen().test_basic_codegen(tmp_dir, func)
    # TestCodegen().test_array_codegen(tmp_dir)
    TestCodegen().test_nested_codegen(tmp_dir)
