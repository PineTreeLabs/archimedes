import pytest
import os

import numpy as np
import archimedes as arc


# TODO:
# - Test data type explicit specification
# - Test data type inference
# - Test with static args
# - Test re-importing with casadi extern

class TestCodegen:
    def _gen_code(self, file):
        # https://web.casadi.org/docs/#syntax-for-generating-code
        
        def myfunc(x, y):
            return x, np.sin(y) * x
        
        # Create arrays of the right shape and dtype
        x_type = np.empty((2,), dtype=float)
        y_type = np.empty((), dtype=float)

        # Can't get code as a string yet
        with pytest.raises(ValueError):
            arc.codegen(myfunc, None, (x_type, y_type), header=True)

        arc.codegen(myfunc, f"{file}.c", (x_type, y_type), header=True)

        # Check that the file was created
        assert os.path.exists(f"{file}.c")
        assert os.path.exists(f"{file}.h")

        # Check that the header includes a proper function signature
        with open(f"{file}.h") as f:
            lines = f.readlines()
            for line in lines:
                if "int myfunc" in line:
                    break
            else:
                assert False

        # Clean up
        os.remove(f"{file}.c")
        os.remove(f"{file}.h")

    def test_codegen(self):
        self._gen_code("gen")

    def test_error_handling(self):
        # Create temporary directory
        if not os.path.exists("tmp"):
            os.mkdir("tmp")
    
        with pytest.raises(RuntimeError):
            self._gen_code("tmp/gen")

        # Clean up
        os.rmdir("tmp")

