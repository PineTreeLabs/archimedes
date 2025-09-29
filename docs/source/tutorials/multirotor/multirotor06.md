# Conclusion

In this tutorial, we've explored the capabilities of Archimedes through the lens of multirotor flight vehicle modeling. We've demonstrated how Archimedes can handle complex computational graphs efficiently, provided a proof-of-concept implementation for multirotor dynamics, and laid the groundwork for more advanced modeling and analysis techniques.

We covered the basics of multirotor flight vehicle modeling, including 6-dof rigid body dynamics and blade-element momentum theory for rotor aerodynamics. While the details of this model are fairly specific to this kind of flight vehicle, it represents a moderately complex physics model common in aerospace engineering applications.

Key points from this tutorial include:

* A straightforward implementation of the model equations in NumPy easily ran with Archimedes, yielding a ~5x speedup in simulation.
* Archimedes also enabled automatic differentiation for trim point identification and linear stability analysis, again with the same NumPy code
* We saw the power and simplicity of the implicit function abstraction in Archimedes when implementing the momentum theory inflow model.

Beyond illustrating multirotor flight vehicle modeling, this code (or a similar framework) can be the starting point for more advanced analysis and applications, such as:

* Trajectory optimization to design optimal flight paths and control sequences
* Parameter estimation for calibrating model parameters against test data
* Advanced aerodynamics modeling like dynamic inflow or vortex particle methods
* Design optimization with an XFOIL-type panel code to solve sectional aerodynamics for variable airfoil profiles
* State estimation (IMU) algorithms and discrete control system design with automatic C code generation for hardware deployment

We also discussed the design of a modular and extensible modeling framework, combining a hierarchy of classes (inheritance) with nested callable classes that implement well-defined abstract interfaces (composition). By thoughtfully structuring the framework, we were able to add a significantly more complex component model (the blade-element model) while fully reusing our existing code. Similar principles can be applied to a wide range of physics models.

To explore Archimedes further, check out:

1. The [GitHub repository](https://github.com/pinetreelabs/archimedes) to access the full codebase
2. The [website](https://pinetreelabs.github.io/archimedes) site for tutorials and in-depth documentation
3. Join our [community forum](https://github.com/pinetreelabs/archimedes/discussions) to discuss your projects and get help from other users
