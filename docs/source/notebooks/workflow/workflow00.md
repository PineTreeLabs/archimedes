# End-to-End Controller Development

Controller development can be a messy process involving a patchwork of software tools, hardware configurations, and algorithm parameters - but it doesn't need to be.
Pure software development often follows a highly structured process designed to ensure both reliability of the output _and_ engineering efficiency.
While hardware development by its nature will always be a harder beast than pure software, we can take inspiration from these workflows to imagine a process with much less friction.

In this tutorial we will go through one such workflow to see how Python and Archimedes can act as a unifying high-level layer enabling a logical development progression.

```{image} _static/dev_workflow.png
:class: only-light
```

```{image} _static/dev_workflow_dark.png
:class: only-dark
```

## Tutorial overview

Our example application and control algorithm will be as simple as possible: a brushed DC motor controlled by PI feedback.
This will let us focus on the development _process_, which can scale to more complex systems, without getting bogged down in the application-specific physics and algorithm details.

As mapped out in the figure above, we will start by constructing a first-principles model of the system and collecting some characterization data.
We can then apply parameter estimation to calibrate the physics model using the test data, resulting in a "plant" model.
We will design a controller based on the plant model and simulate its performance - all in Python.

Following a more traditional or simplified development cycle, we can then simply generate C code corresponding to the Python control algorithm, deploy to hardware, and evaluate performance on a test system.
Here we will explore incorporating an additional stage of the controller development: hardware-in-the-loop (HIL) testing.
Using the same code generation mechanisms, we can construct a real-time simulation of the plant model and connect this to the controller board.
As far as the controller knows, it is sensing and actuating the real system, making this a valuable testing stage that can catch costly or difficult-to-debug errors before deploying to the real hardware.

## Prerequisites

This is an advanced tutorial that will integrate a number of the key features of Archimedes, including [PyTrees](../../pytrees.md), [C code generation](#archimedes.codegen), and [system identification](#archimedes.sysid).
As a result, it will be much easier to follow if you are already comfortable with these concepts - the following documentation pages are a good place to start:

* [**Working with PyTrees**](../../pytrees.md)
* [**Hierarchical Design Patterns**](../../generated/notebooks/modular-design.md)
* [**Parameter Estimation**](../../generated/notebooks/sysid/parameter-estimation.md)
* [**Deploying to Hardware**](../deployment/deployment00.md)

Beyond Archimedes specifics, the tutorial only assumes basic physics and control systems knowledge (an RL circuit and proportional-integral control), though familiarity with the [Python Control System Library](https://python-control.readthedocs.io/) may be helpful in the controller design section.

## Outline

1. [**Physical System**](../../generated/notebooks/workflow/workflow01.md)
    - Brushed DC motor and physics model
    - Motor driver circuit
    - The STM32 controller board
    - Bill of materials (if you want to build it yourself)

2. [**Characterization**](../../generated/notebooks/workflow/workflow02.md)
    - Configuring the STM32
    - Collecting step response data

3. [**Parameter Estimation**](../../generated/notebooks/workflow/workflow03.md)
    - Implementing the physics model
    - Calibrating with test data

4. [**Controller Design**](../../generated/notebooks/workflow/workflow04.md)
    - Implementing a simple PI controller
    - Classical control systems analysis
    - C code generation

5. [**HIL Testing**](../../generated/notebooks/workflow/workflow05.md)
    - Setting up a real-time simulator
    - The analog communication circuit
    - Generating code for the real-time model
    - Evaluating the controller

6. [**Deployment**](../../generated/notebooks/workflow/workflow06.md)
    - Running the same controller on the physical system
    - Comparing to HIL testing results

7. [**Conclusions**](workflow07.md)
    - Overview of the workflow
    - Key takeaways

<!--
Lessons:
- Complex controller failed
- Missing volatile flat
-->

```{toctree}
:maxdepth: 1
../../generated/notebooks/workflow/workflow01
../../generated/notebooks/workflow/workflow02
../../generated/notebooks/workflow/workflow03
../../generated/notebooks/workflow/workflow04
../../generated/notebooks/workflow/workflow05
../../generated/notebooks/workflow/workflow06
workflow07
   
```