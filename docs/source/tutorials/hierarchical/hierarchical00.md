# Hierarchical Modeling

This tutorial series covers the basics of _hierarchical modeling_ in Archimedes.
In this context, "hierarchical modeling" just means having nested or composite models of complex dynamical systems.

The design patterns described in [Part 1](hierarchical01.md) let you write code that maps to how you naturally think about engineering systems in terms of a hierarchy of subsystems and components.
[Part 2](hierarchical02.md) takes a deeper dive into some more advanced topics including defining interfaces for generic components and creating a scalable configuration management system.

:::{warning}
Part 2 is considerably more advanced, targeted at implementations of complex systems with lots of parameters, subsystems, and variant component models.
If you're new to Archimedes (or Python), you might still want to take a look to get an idea of what it's about, but if it seems complicated please feel free to completely ignore it.
You can absolutely use Archimedes with zero "configuration management" - this page just exists to document the scaffolding in case you do need it.
:::

## Outline

1. [**Design Patterns**](hierarchical01.md)
    - Decomposing a complex system into natural components
    - Recommended template for component models
    - Nesting modular components to create hierarchical models

2. [**Configuration Management**](hierarchical02.md)
    - Implementing `Protocol` types for component models
    - Defining configuration classes
    - Configuration unions for component variants
    - Reading from a configuration file

## Prerequisites

This series assumes only that you are familiar with the basics of Archimedes (for instance the content on the [Getting Started](../../getting-started.md) page) and the concept of [structured data types](../../trees.md).

[Part 2](hierarchical02.md) does use some more advanced Python concepts, but as noted above, it's not necessary to use these (or even read that page) to start applying hierarchical modeling principles to your own project.

```{toctree}
:maxdepth: 1
hierarchical01
hierarchical02
   
```