# MERA.jl

MERA.jl provides Julia implementations of some basic [Multiscale Entaglement Renormalization Ansatz](https://arxiv.org/abs/quant-ph/0610099) algorithms. It only implements infinite, translation invariant MERAs. At the moment it has implementations of both ternary and binary MERAs, with functions for doing energy minimization, evaluating expectation values, and computing scaling dimensions.

MERA.jl makes extensive use of [TensorKit](https://github.com/Jutho/TensorKit.jl), and uses it to support global internal symmetries, both Abelian and non-Abelian.

## Usage

The type system is based on `GenericMERA{T} where T <: Layer`, and `TernaryMERA = GenericMERA{TernaryLayer}` and `BinaryMERA = GenericMERA{BinaryLayer}`.

The file `src/genericmera.jl` implements functions that are independent of the exact type of MERA. `src/ternarylayer.jl` and `src/binarylayer.jl` provide the details of things like ascending/descending superoperators, that depend on the specific MERA. `src/tensortools.jl` supplies some functions for TensorKit objects such as `TensorMap`s and vector spaces that the rest of the package needs.

The folder `demo` has a script and supporting files for it, that runs energy minimization on either the Ising or the XXZ model, and computes scaling dimensions and entanglement entropies from the resulting MERA. The best way to get going is to clone this repo, navigate to its folder, open a Julia prompt and do
```
]activate .
include("demo/demo.jl")
```
