# MERA.jl
[![][travis-img]][travis-url] [![][codecov-img]][codecov-url]

MERA.jl provides Julia implementations of some basic [Multiscale Entaglement Renormalization Ansatz](https://arxiv.org/abs/quant-ph/0610099) [algorithms](https://arxiv.org/abs/0707.1454). It only implements infinite, translation invariant MERAs. At the moment it has implementations of both ternary and binary MERAs, with functions for doing energy minimization, evaluating local expectation values, and computing scaling dimensions.

MERA.jl makes extensive use of [TensorKit](https://github.com/Jutho/TensorKit.jl), and uses it to support global internal symmetries, both Abelian and non-Abelian.

MERA.jl remains in active development as of January 2020. Some further plans include supporting modified binary MERAs, improving performance, and developing the optimization method.

## Usage

The folder `demo` has a script `demo.jl`, that runs energy minimization on either the Ising or the XXZ model (your choice), and computes scaling dimensions and entanglement entropies from the resulting MERA. The best way to get going is to clone this repo, navigate to its folder, open a Julia prompt and do
```
]activate .
include("demo/demo.jl")
```

`demo.jl` writes to disk the MERAs it creates, by default in a folder called `JLMdata`. Another script, `demo/demo_refine.jl`, can be used to load these files, optimize the MERA further for better convergence, and write them back to disk. If I would need high quality results for a given bond dimension, I would first generate a decent starting point using `demo.jl`, which builds the MERA up by slowly increasing bond dimension, and then use `demo_refine.jl` to push for proper convergence. Both of these scripts use `demo/demo_tools.jl`, which deals with creating Hamiltonians, writing to and reading from disk, and gradually increasing bond dimension during an optimization.

The actual library is obviously in `src`. The type system is based on a an abstract type `GenericMERA{T} where T <: Layer`, and its concrete subtypes `TernaryMERA = GenericMERA{TernaryLayer}` and `BinaryMERA = GenericMERA{BinaryLayer}`. The file `src/genericmera.jl` implements functions that are independent of the exact type of MERA. `src/ternarylayer.jl` and `src/binarylayer.jl` provide the details of things like ascending/descending superoperators, that depend on the specific MERA. `src/tensortools.jl` supplies some functions for TensorKit objects such as `TensorMap`s and vector spaces that the rest of the package needs.

[travis-img]: https://travis-ci.org/mhauru/MERA.jl.svg?branch=master
[travis-url]: https://travis-ci.org/mhauru/MERA.jl
[codecov-img]: https://codecov.io/gh/mhauru/MERA.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/mhauru/MERA.jl
