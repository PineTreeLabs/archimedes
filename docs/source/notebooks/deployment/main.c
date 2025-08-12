// gcc -o main main.c fib.c fib_kernel.c

#include <stdio.h>
#include "fib.h"


// Declare the argument, result, and workspace structures
fib_arg_t arg;
fib_res_t res;
fib_workspace_t workspace;


int main() {

    // Initialize the structs
    fib_init(&arg, &res, &workspace);

    for (int i = 0; i < 10; i++) {
        // Perform a step in the Fibonacci sequence
        fib_step(&arg, &res, &workspace);

        // Update the arguments for the next step
        arg.a = res.a_new;
        arg.b = res.b_new;

        // Print the current Fibonacci number
        printf("%d\n", (int)arg.a);
    }

    return 0;
}