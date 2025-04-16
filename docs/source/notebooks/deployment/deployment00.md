# Deploying to Hardware

A major goal of Archimedes is to provide a "path to hardware" from Python.  Importantly though, this does _not_ mean "running Python on a microcontroller".  For a variety of reasons, this is often not a viable approach, especially for safety- and mission-critical systems that require precise and deterministic memory and timing management.

Instead, the workflow in Archimedes is to develop your _algorithms_ in Python, test them in simulation, then automatically translate them to efficient C implementations that are ready for deployment.

This example walks through a simple example of this process.  We will develop an IIR filter using SciPy's signal processing tools and generate code for deployment to an Arduino.  Although this is a simplified example, it illustrates the key steps of the Archimedes hardware deployment model.

The core workflow involves three pieces of code:

1. An Archimedes-compatible Python function (written by you)
2. A C implementation of the same function (generated automatically)
3. A platform-specific "driver" code in C (generated semi-automatically)

We'll get into the details later on.  For now, the key idea is that only the C code gets deployed to the target hardware, but you can modify the Python code to quickly make changes that automatically propagate to the C source code. This unlocks streamlined workflows for quickly moving from modeling, simulation, and analysis to deployment and testing.

Before we get started, one last comment: the full "path to hardware" includes not only code generation, but data acquisition and hardware-in-the-loop (HIL) testing workflows.  These are central to the [development roadmap](../../roadmap.md) - stay tuned for more on those capabilities soon.


```{toctree}
:maxdepth: 1
../../generated/notebooks/deployment/deployment01
../../generated/notebooks/deployment/deployment02
../../generated/notebooks/deployment/deployment03
   
```