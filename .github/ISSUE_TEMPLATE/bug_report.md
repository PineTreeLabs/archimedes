---
name: Bug report
about: Report a bug or unexpected behavior
title: ''
labels: bug
assignees: ''

---

<!--
Thank you for taking the time to report an issue with Archimedes. 
To help us address your issue efficiently, please provide the information requested below.
The more details you provide, the faster we can help!
-->

**Describe the bug**
<!-- A clear and concise description of what the bug is -->

**Expected behavior**
<!-- Describe what you expected -->

**To Reproduce**
1. 
2. 
3. 

** Minimal Reproducible Example **
<!--
Please provide the smallest, complete piece of code that demonstrates the issue.
This makes it much easier for us to diagnose and fix the problem.

A good example looks like:
```python
import archimedes as arc
import numpy as np

@arc.compile
def problem_function(x):
    return np.array([x[1], x[0]])  # This fails with error X

x = np.array([1.0, 2.0])
problem_function(x)  # Error occurs here
```
>>>

```python
# Youe code example here
```

** Error Message/Stack Trace **
<!-- If applicable, copy the full error message and stack trace -->

```raw
Paste your error message here
```

** What have you tried? **
<!-- Describe any attempts you've made to solve or work around the issue -->

**Environment**
- Python version:
- Operating system:
- Installation method: [e.g., pip install, from source]
- Archimedes version:
- CasADi version:
- NumPy version:

**Additional context**
<!-- Any other information that might be helpful (screenshots, related issues, etc.) -->

** Checklist **
Checklist Before Submitting

- [] Searched for similar issues in the issue tracker
- [] Checked the documentation for information related to this issue
- [] Verified this issue with the latest version of Archimedes
- [] Provided a minimal reproducible example
- [] Included the full error message and stack trace (if applicable)
