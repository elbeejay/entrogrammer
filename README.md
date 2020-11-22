# entrogrammer
![build](https://github.com/elbeejay/entrogrammer/workflows/build/badge.svg)
[![codecov](https://codecov.io/gh/elbeejay/entrogrammer/branch/main/graph/badge.svg?token=VLEEKXSINN)](https://codecov.io/gh/elbeejay/entrogrammer)
[![PyPI version](https://badge.fury.io/py/entrogrammer.svg)](https://badge.fury.io/py/entrogrammer)

Python package to calculate "geologic entropy" and related metrics (e.g. entrogram, entropic scale, and local/global entropy values).

**NOTE:** Package still very much in alpha stages of development. This means that functionality and API are subject to change at any time. From v1.0.0 onwards the intent is for the package to stable and have settled on an API scheme.

## Table of Contents
- [Introduction and Quickstart](#introduction-and-quickstart)
- [Examples](#examples)
- [Background](#background)
- [Contributing Guidelines](#contributing-guidelines)
- [References](#references)

## Introduction and Quickstart
In a nutshell, entropy quantifies the "surprise" content of some set of data. By this, we mean that if data are assigned probabilities and randomly drawn, entropy quantifies the "surprise" you'd encounter in the random drawing (no surprise if all data is identical, high amount of surprise if data is all different). For a more thorough description of entropy, we recommend starting with Claude Shannon's seminal paper on the subject [[1]](#1). 

With `entrogrammer`, data can be classified into discrete categories, and entropy can be calculated at a variety of scales. This package is inspired by the works of Bianchi and Pedretti [[2](#2), [3](#3)] in which they related this idea of entropy to solute transport in porous media. 

A quick example of calculating the entrogram of a small 1-D series of data ([0, 1, 0]) can be performed using `entrogrammer` with the following Python code:

```
import numpy as np
from entrogrammer import classifier, core
C = classifier.BinaryClassifier(np.array([0, 1, 0]), 0.5)  # binary classification of data
C.classify()
HR, win_size = core.calculate_entrogram(C)  # calculate the entrogram of this vector
```

From this point, the entrogram for this data could be plotted by making a plot of `HR` vs `win_size` to get a visual understanding of how the entropy varies at different scales. For a definition of the entrogram, some examples and an explanation of the metric, see [[3]](#3) and the [background](#background) section.

## Examples
For example jupyter notebooks illustrating the functionality provided in `entrogrammer` please visit the [examples subdirectory](https://github.com/elbeejay/entrogrammer/tree/main/examples).

## Background
Herein the procedures used by `entrogrammer` are described...

## Contributing Guidelines
How to contribute...

## References
<a id="1">[1]</a> 
Shannon, Claude E. "A mathematical theory of communication." The Bell system technical journal 27.3 (1948): 379-423.

<a id="2">[2]</a> 
Bianchi, Marco, and Daniele Pedretti. "Geological entropy and solute transport in heterogeneous porous media." Water Resources Research 53.6 (2017): 4691-4708.

<a id="3">[3]</a>
Bianchi, Marco, and Daniele Pedretti. "An Entrogram‐Based Approach to Describe Spatial Heterogeneity With Applications to Solute Transport in Porous Media." Water Resources Research 54.7 (2018): 4432-4448.

