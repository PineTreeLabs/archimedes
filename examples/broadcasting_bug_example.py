"""
Example verifying broadcasting correctness in archimedes.

This example demonstrates that broadcasting between matrix (2, 3) and vector (3,)
works correctly for non-commutative operations like subtraction.
"""

import numpy as np
import casadi as ca
import archimedes as arc


def demonstrate_broadcasting_bug():
    """Show the broadcasting bug with a simple example."""

    # Create test values
    # Matrix A of shape (2, 3)
    A_val = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ])

    # Vector b of shape (3,)
    b_val = np.array([0.1, 0.2, 0.3])

    print("=" * 60)
    print("Broadcasting Bug Example: Matrix (2,3) - Vector (3,)")
    print("=" * 60)
    print()
    print(f"Matrix A (shape {A_val.shape}):")
    print(A_val)
    print()
    print(f"Vector b (shape {b_val.shape}):")
    print(b_val)
    print()

    # Expected result with numpy
    np_result = A_val - b_val
    print("Expected result (numpy: A - b):")
    print(np_result)
    print()

    # Create symbolic arrays with archimedes
    A_sym = arc.sym("A", (2, 3), kind="SX")
    b_sym = arc.sym("b", (3,), kind="SX")

    # Perform the subtraction
    result_sym = A_sym - b_sym

    print(f"Archimedes result shape: {result_sym.shape}")
    print()

    # Build CasADi function to evaluate
    result_func = ca.Function(
        "broadcast_sub",
        [A_sym._sym, b_sym._sym],
        [result_sym._sym],
        ["A", "b"],
        ["result"]
    )

    # Evaluate with concrete values
    cs_result = result_func.call({"A": A_val, "b": b_val})
    arc_result = cs_result["result"].__array__()

    print("Actual result (archimedes):")
    print(arc_result)
    print()

    # Verify results match
    print("-" * 60)
    print("VERIFICATION:")
    print("-" * 60)
    print()

    if np.allclose(arc_result, np_result):
        print("[OK] Archimedes result matches numpy!")
    else:
        print("[FAIL] Results don't match")
        print("Difference:")
        print(arc_result - np_result)


if __name__ == "__main__":
    demonstrate_broadcasting_bug()
