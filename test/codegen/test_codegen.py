import os
import shutil
import tempfile

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


@pytest.fixture()
def myfunc():
    def func(x, y):
        return x, np.sin(y) * x

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

    # Compare (normalize whitespace to handle line endings)
    assert expected.strip() == actual.strip()


def check_in_file(file, pattern):
    with open(file, "r") as f:
        content = f.read()
        assert pattern in content


class TestCodegen:
    def test_codegen(self, temp_dir, myfunc):
        kwargs = {
            "float_type": np.float32,
            "output_dir": temp_dir,
        }

        # Generate code for the function.  This iteration will also compile
        arc.codegen(myfunc, (x_type, y_type), return_names=("x_new", "z"), **kwargs)

        # Pre-compile the function for the remaining tests
        func = arc.compile(myfunc, name="func", return_names=("x_new", "z"))

        # Check that the files were created
        assert os.path.exists(f"{temp_dir}/{func.name}_kernel.c")
        assert os.path.exists(f"{temp_dir}/{func.name}_kernel.h")
        assert os.path.exists(f"{temp_dir}/{func.name}.c")
        assert os.path.exists(f"{temp_dir}/{func.name}.h")

        # Check that the kernel header includes a proper function signature
        check_in_file(f"{temp_dir}/{func.name}_kernel.h", "int func")

        # Check that the API header includes correct functions
        check_in_file(f"{temp_dir}/{func.name}.h", "int func_init")
        check_in_file(f"{temp_dir}/{func.name}.h", "int func_step")

        # Compare to expected output
        compare_files(f"expected_{func.name}.h", f"{temp_dir}/{func.name}.h")
        compare_files(f"expected_{func.name}.c", f"{temp_dir}/{func.name}.c")

        c_file = f"{func.name}.c"
        h_file = f"{func.name}.h"

        # Run with integer arguments to check type conversion
        # Because CasADi treats everything as a float, this will get
        # converted to float in the generated code.
        y_type_int = np.array(3, dtype=int)
        arc.codegen(func, (x_type, y_type_int), **kwargs)
        check_in_file(f"{temp_dir}/{h_file}", "float y;")
        check_in_file(f"{temp_dir}/{c_file}", "arg->y = 3;")

        # Run with a specified kwarg
        kwargs["kwargs"] = {"y": 5.0}
        arc.codegen(func, (x_type,), **kwargs)
        check_in_file(f"{temp_dir}/{h_file}", "float y;")
        check_in_file(f"{temp_dir}/{c_file}", "arg->y = 5.0;")

    def test_error_handling(self, myfunc):
        # Test with unknown return names.  By design choice, this raises an error
        # in order to help generate more readable code.
        with pytest.raises(ValueError, match=r"Return names must be provided"):
            arc.codegen(myfunc, (x_type, y_type))

        func = arc.compile(myfunc, name="func", return_names=("x_new", "z"))

        # Can't use non-numeric inputs
        with pytest.raises(TypeError, match=r"Argument .* is not numeric.*"):
            arc.codegen(func, (x_type, "string"))

        # Should be impossible to do this anyway, but for completeness, check that
        # the low-level codegen function raises an error if the target path is not
        # the CWD.
        specialized_func, _ = func._specialize(x_type, y_type)
        with pytest.raises(RuntimeError):
            specialized_func.codegen("invalid/func.c", {})


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
