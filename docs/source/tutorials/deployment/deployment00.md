# Hardware Deployment

Controller development can be a messy process involving a patchwork of software tools, hardware configurations, and algorithm parameters - but it doesn't need to be.
Pure software development often follows a highly structured process designed to ensure both reliability of the output _and_ engineering efficiency.
While hardware development by its nature will always be a different beast than pure software, we can take inspiration from workflows incorporating continuous integration, automated testing, and version control to imagine a process with much less friction.

In this tutorial we will go through one such workflow to see how Python and Archimedes can act as a unifying high-level layer enabling a logical development progression.
The result is a straightforward development cycle with the potential for faster iteration, improved validation, and reduced hardware risks.

```{image} _static/dev_workflow.png
:class: only-light
```

```{image} _static/dev_workflow_dark.png
:class: only-dark
```

This workflow assumes that the hardware itself is fixed, though of course the controls development could be embedded in a broader design loop for hardware iteration.

## Tutorial overview

Our example application and control algorithm will be as simple as possible: a brushed DC motor controlled by PI feedback.
This will let us focus on the development _process_, which can scale to more complex systems, without getting bogged down in the application-specific physics and algorithm details.

As mapped out in the figure above, we will start by constructing a first-principles model of the system and collecting some characterization data.
We can then apply parameter estimation to calibrate the physics model using the test data, resulting in a "plant" model.
We will design a controller based on the plant model and simulate its performance - all in Python.

In principle, we could then simply generate C code corresponding to the Python control algorithm, deploy to hardware, and evaluate performance on a test system.
However, here we will explore incorporating an additional stage of the controller development: hardware-in-the-loop (HIL) testing.
Using the same code generation mechanisms, we can construct a real-time simulation of the plant model and connect this to the controller board - as far as the controller knows, it is sensing and actuating the real system.

```{image} _static/hil_diagram.png
:class: only-light
```

```{image} _static/hil_diagram_dark.png
:class: only-dark
```

While HIL testing is often relegated to late-stage validation for compliance requirements, it can also be a valuable testing stage to catch costly or difficult-to-debug errors before deploying to the real hardware.

This end-to-end workflow tutorial provides an example of how Archimedes can be used as the backbone of a modern, structured approach to rapid development iteration.
While simplified (and applied to a simple physical system), every aspect of the process can scale naturally to more complex workflows and systems.

## Outline

1. [**Physical System**](deployment01.md)
    - Brushed DC motor and physics model
    - Motor driver circuit
    - The STM32 controller board
    - Bill of materials (if you want to build it yourself)

2. [**Characterization**](deployment02.md)
    - Configuring the STM32
    - Collecting step response data
    - Implementing the physics model
    - Calibration via parameter estimation

3. [**Controller Design**](deployment03.md)
    - Implementing a simple PI controller
    - Classical control systems analysis
    - C code generation

4. [**HIL Testing**](deployment04.md)
    - Setting up a real-time simulator
    - The analog communication circuit
    - Generating code for the real-time model
    - Evaluating the controller

5. [**Deployment**](deployment05.md)
    - Running the same controller on the physical system
    - Comparing to HIL testing results
    - Key takeaways


## Prerequisites

This tutorial integrates several Archimedes concepts, including [structured data types](../../trees.md), [C code generation](#archimedes.codegen), and [system identification](#archimedes.sysid).
We'll introduce them as needed, but it will be easier to follow if you are already comfortable with these concepts - the following documentation pages are a good place to start:

* [**Structured Data Types**](../../trees.md)
* [**Hierarchical Systems Modeling**](../hierarchical/hierarchical00.md)
* [**Parameter Estimation**](../sysid/parameter-estimation.md)
* [**C Code Generation**](../codegen/codegen00.md)

Beyond Archimedes specifics, the tutorial only assumes basic physics and control systems knowledge (an RL circuit and proportional-integral control), though familiarity with the [Python Control System Library](https://python-control.readthedocs.io/) may be helpful in the controller design section.

```{toctree}
:maxdepth: 1
deployment01
deployment02
deployment03
deployment04
deployment05
   
```
