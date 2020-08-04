# MERA.jl
[![][travis-img]][travis-url] [![][codecov-img]][codecov-url]

MERA.jl provides Julia implementations of some basic [Multiscale Entaglement Renormalization Ansatz](https://arxiv.org/abs/quant-ph/0610099) algorithms. It only implements infinite, translation invariant MERAs. At the moment it has implementations of ternary, binary, and modified binary MERAs, with functions for doing energy minimization, evaluating local expectation values, and computing scaling dimensions. An implementation is provided of the classic [alternating energy minimization algorithm](https://arxiv.org/abs/0707.1454), that in the code is called the Evenbly-Vidal, or EV, algorithm. Work is also ongoing on gradient based optimization methods. MERA.jl makes extensive use of [TensorKit](https://github.com/Jutho/TensorKit.jl), and uses it to support global internal symmetries, both Abelian and non-Abelian.

MERA.jl remains in active development as of April 2020.

## Usage

The folder `demo` has a script `demo.jl`, that runs energy minimization on either the Ising or the XXZ model, and computes scaling dimensions and entanglement entropies from the resulting MERA. The best way to get going is to clone this repo, navigate to its folder, open a Julia prompt and do
```
]activate .
include("demo/demo.jl")
```

`demo.jl` writes to disk the MERAs it creates, by default in a folder called `JLMdata`. Another script, `demo/demo_refine.jl`, can be used to load these files, optimize the MERA further for better convergence, and write them back to disk. You can for instance first create a decent decent starting point for a MERA using `demo.jl`, since it builds the MERA up by slowly increasing bond dimension, and then use `demo_refine.jl` to push for proper convergence. Both of these scripts use `demo/demo_tools.jl`, which deals with creating Hamiltonians, writing to and reading from disk, and gradually increasing bond dimension during an optimization. Both `demo.jl` and `demo_refine.jl` take plenty of command line arguments, allowing things
```
julia --project=. demo/demo.jl --model=XXZ --meratype=binary --chi=5 --layers=4 --symmetry=none
```
See the source code for more details.

The actual library is obviously in `src`. The type system is based on an abstract type `GenericMERA{N, LT} where LT <: Layer`, and its concrete subtypes such as `TernaryMERA{N} = GenericMERA{N, TernaryLayer}` and `BinaryMERA{N} = GenericMERA{N, BinaryLayer}`. The file `src/genericmera.jl` implements functions that are independent of the exact type of MERA. `src/simplelayer.jl` implements methods for the abstract type `SimpleLayer` that all the concrete `Layer` types are subtypes of, that just assumes that each layer consists of a finite number of `TensorMap`s. `src/ternarylayer.jl`, `src/binarylayer.jl`, and `src/modifiedbinarylayer.jl` provide the details of things like ascending/descending superoperators, that depend on the specific MERA. `src/tensortools.jl` supplies some functions for TensorKit objects such as `TensorMap`s and vector spaces that the rest of the package needs.

[travis-img]: https://travis-ci.org/mhauru/MERA.jl.svg?branch=master
[travis-url]: https://travis-ci.org/mhauru/MERA.jl
[codecov-img]: https://codecov.io/gh/mhauru/MERA.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/mhauru/MERA.jl
