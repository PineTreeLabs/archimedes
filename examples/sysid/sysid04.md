
# Conclusion

In this tutorial we've covered basic workflows for system identification in Archimedes, including:

- How to use the high-level system ID interface in Archimedes
- Using PyTrees to organize parameters and system models
- How to add bounds and constraints to enforce physical realism
- Customizing objectives and constraints using a lower-level optimization interface

However, as we noted at the beginning, there are a number of crucially important topics that we _didn't_ cover, such as:

- **Data wrangling:** How should you clean up and preprocess you data?
- **Model parameterization:** How should the system be modeled mathematically and implemented numerically?
- **Noise modeling:** How do you estimate the noise covariance matrices needed for the predictive filters?
- **Validation:** How do you make sure the model is correct?

Instead, all of these examples used cases where we had enough good-quality data, known model structures, etc.
This was largely to keep the focus on the algorithm and workflow rather than a comprehensive guide to best practices in system ID.
The "best practices" are also frequently domain- or problem-specific.

That said, we can share some general advice for data-driven modeling.
The following is based on a wide variety of system ID work across a range of disciplines.
However, it's not one-size-fits-all; trust your engineering intuition above hard and fast rules.

## Practical tips

**Data:**

- You don't necessarily need "big data" for system ID. You need to capture parametric dependence, intrinsic dynamics, and I/O relationships - that's it.
- Design different experiment types to capture these different relationships: step/staircase response, chirp response, closed-loop control (for unstable systems).
- Think in terms of capturing the slowest time scales of the system.  A good timeseries might be 3-10x the longest relevant time scales, but doesn't need to be 100x unless the system is intrinsically chaotic or otherwise pathological.
- Clearly define the ranges of parametric variation and plan experiments to cover this space. Sampling density depends on parameterization: the more complex the parameterization, the more data you need to ensure proper validation.

**Model structure:**

- "Inductive bias" usually helps: add known physics wherever you can. For example, a great combination is a higher-fidelity gray-box model calibrated to experimental data plus a faster, approximate black-box model (for real-time estimation and control) that is carefully anchored to the gray-box.
- Start with linear models first. They are the most easily interpretable, reliable, and amenable to analytic treatment. Look at coherence metrics to determine "how linear" the dynamics are.
- For more complex behavior, try including "static" nonlinearities to modify the inputs and/or outputs to the linear system (Hammerstein-Wiener form).  This is a powerful way to model a wide range stable nonlinear systems.
- For inherently nonlinear systems (e.g. bistability, hysteresis, limit cycle oscillations), try "weakly nonlinear" normal form models.  These are low-order polynomials that capture some canonical nonlinear behavior. For example, if your system has limit cycle oscillations, try the Hopf normal form. For bistability, try the pitchfork normal form.
- For more expressivity with simple nonlinear models, combine a weakly nonlinear model with static output nonlinearities.
- When using static nonlinearities, prefer simple parameterizations whenever possible. These are easier to analyze and understand and simplicity mitigates over-fitting. Add complexity if needed (see "neural network" recommendations) and be very careful about validation.
- For complex systems, think hierarchically: can you break the model down into smaller pieces that can be independently calibrated?  Keep components as simple and interpretable as possible.

**Noise modeling**

- The process noise (Q) and measurement noise (R) essentially play the role of "how much to trust the model" in the PEM algorithm.  For Q >> R (in orders of magnitude), the assumed measurement noise is small and the Kalman filter will give a lot of weight to measurements, correcting even very poor models.  The opposite is also true for Q << R.  Having Q be too large for deterministic systems can lead to poor convergence of the optimization.
- Many systems are approximately deterministic, but with measurement noise. Try modeling these with Q 2-4 orders of magnitude less than R.
- Estimate diagonal R based on variance of single-step differences in measurement time series (assuming sampling rate is ~10x larger than fastest frequencies).  See the code in [Part 1](
../../generated/notebooks/sysid/sysid01) for an example.
- More sophisticated noise estimation: iterative parameter/noise estimates with MAP or add Cholesky factorizations of Q, R as parameters in MLE formulation.

**Neural networks**

- These typically take orders of magnitude more data to train robust deep neural network models compared to linear, weakly nonlinear, or Hammerstein-Wiener forms.
- May require different optimization methods (e.g. mini-batch SGD or ADAM instead of second-order least-squares or NLP solvers)
- Consider for use as static nonlinearities when there are multiple interacting inputs or outputs (but ensure there is enough data to do proper cross-validation)
- Avoid "neural ODEs" for safety-critical applications: very hard to comprehensively characterize model behavior.

## Final thoughts

We started this tutorial with the question:

**You have a model of the dynamics of your system.  You have test data from your system.  How do you get them to match?**

By working through three increasingly complex examples, we've seen how to use Archimedes to solve this problem.
The combination of hierarchical PyTree modeling, automatic differentiation, and familiar NumPy syntax provides a flexible and powerful foundation for system identification.

The identified models are also immediately compatible with the rest of the Archimedes ecosystem, meaning that you can use them for trajectory optimization, sensitivity analysis, and more.
You can even build up an integrated workflow for system identification → controller design → [hardware deployment](../../tutorials/codegen/codegen00.md) that will take you from data to deployment all from the comfort of Python.

### Where to go from here

The best way to continue learning is by working through the system identification process on your own with real data.
For some widely used benchmark data sets to get started, see [Nonlinear Benchmarks](https://www.nonlinearbenchmark.org) and the [DaISy](https://homes.esat.kuleuven.be/~smc/daisy/daisydata.html) database.

If you haven't already, be sure to read the documentation pages on [Working with PyTrees](../../trees.md) and [Hierarchical Design Patterns](../../generated/notebooks/modular-design.md) to get a feel for some approaches to modeling complex systems.
You may also want to check out the [Multirotor Dynamics](../../tutorials/multirotor/multirotor00.md) series to see these patterns in action.

To explore Archimedes further, check out:

1. The [GitHub repository](https://github.com/pinetreelabs/archimedes) to access the full codebase
2. The [website](https://archimedes.sh/docs) site for tutorials and in-depth documentation
3. Join our [community forum](https://github.com/pinetreelabs/archimedes/discussions) to discuss your projects and get help from other users