# MERAKit.jl
[![][docs-img]][docs-url] [![CI](https://github.com/mhauru/MERAKit.jl/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/mhauru/MERAKit.jl/actions/workflows/ci.yml) [![][codecov-img]][codecov-url]

MERAKit.jl provides Julia implementations of [Multiscale Entaglement Renormalization Ansatz](https://arxiv.org/abs/quant-ph/0610099) algorithms.
At the moment it only implements infinite, translation invariant MERA.
It has implementations of ternary, binary, and modified binary MERA, with functions for doing energy minimization, evaluating local expectation values, and computing scaling dimensions.
Energy can be minimised using either the classic [alternating energy minimization algorithm](https://arxiv.org/abs/0707.1454), that in the code is called the Evenbly-Vidal algorithm, or using [gradient-based optimization methods](https://arxiv.org/abs/2007.03638).
MERAKit.jl makes extensive use of [TensorKit](https://github.com/Jutho/TensorKit.jl), and uses it to support global internal group symmetries, both Abelian and non-Abelian, as well as anyonic MERAs.

## Installation
```
]add https://github.com/mhauru/MERAKit.jl
```
or if you also want the demo scripts discussed below,
```
git clone https://github.com/mhauru/MERAKit.jl
```

## Usage

The reference documentation can be found [here][docs-url].
However, in practice the best way to get started is to use the script `demo/demo.jl` as an example: `julia --project=. demo/demo.jl` should get you running.
It runs energy minimization on the Ising model, using a bond dimension 8 ternary MERA, and computes scaling dimensions and entanglement entropies for the resulting MERA.
It shows you how to initialize a random MERA, optimize it for a given Hamiltonian, and measure things from the MERA.
It also shows you how to gradually increase the bond dimension during an optimization, something that is often very helpful in aiding convergence.
Once you've gone through `demo.jl`, you can check out the reference docs for things like additional options for how to do energy minimization, etc.

If you have any questions, requests, or issues, feel free to open a GitHub issue or email [markus@mhauru.org](mailto:markus@mhauru.org).

## Structure of the package

The actual library is obviously in `src`.
The type system is based on an abstract type `GenericMERA{N, LT} where LT <: Layer`, and its concrete subtypes such as `TernaryMERA{N} = GenericMERA{N, TernaryLayer}` and `BinaryMERA{N} = GenericMERA{N, BinaryLayer}`.
Here's a rough summary of the contents of each file in `src`:
* `MERAKit.jl`: Exports, imports, and inclusion of the other files.
* `layer.jl`: Define the abstract type `Layer`, and empty functions for it that subtypes should implement.
* `genericmera.jl`: The `GenericMERA` type and all functions that are common to all types of MERAs.
* `meracache.jl`: A cache for things like ascended operators and environments, used by `GenericMERA`.
* `simplelayer.jl`: `SimpleLayer <: Layer`, an abstract type for layers that are made out of a collection of tensors and nothing else, and methods for it. All of the current concrete layer types are subtypes of `SimpleLayer`.
* `binarylayer.jl`, `ternarylayer.jl`, `modifiedbinarylayer.jl`: The concrete layer types, and all methods that depend on the specific type of MERA, e.g. diagrams for contraction of ascending and descending superoperators.
* `modifiedbinaryop.jl`: The `ModifiedBinaryOp` type, that's used for representing the alternating structure of operators ascended/descended through a `ModifiedBinaryMERA`.
* `tensortools.jl`: Utilities, mostly related to `TensorMap`s.

See also `test/runtests.jl` for the test suite.

[docs-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-url]: https://mhauru.github.io/MERAKit.jl/dev/
[travis-img]: https://travis-ci.org/mhauru/MERAKit.jl.svg?branch=master
[travis-url]: https://travis-ci.org/mhauru/MERAKit.jl
[codecov-img]: https://codecov.io/gh/mhauru/MERAKit.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/mhauru/MERAKit.jl
