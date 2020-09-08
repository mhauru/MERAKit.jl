# MERA.jl reference

MERA.jl provides Julia implementations of some basic [Multiscale Entaglement Renormalization Ansatz](https://arxiv.org/abs/quant-ph/0610099) algorithms. For usage instructions, see the [GitHub page](https://github.com/mhauru/MERA.jl). Below you can find the reference documentation, listing all the types and functions.

## MERA types
```@docs
GenericMERA
TernaryMERA
BinaryMERA
ModifiedBinaryMERA
```

## Layer types
```@docs
Layer
SimpleLayer
TernaryLayer
BinaryLayer
ModifiedBinaryLayer
```

## Operator types
```@docs
ModifiedBinaryOp
```

## Utility functions
```@docs
num_translayers
layertype
operatortype
scalefactor
causal_cone_width
getlayer
outputspace
inputspace
internalspace
pseudoserialize
depseudoserialize
remove_symmetry
projectisometric
projectisometric!
reset_storage
```

## Generating and modifying MERAs
```@docs
replace_layer
release_transitionlayer
random_MERA
randomlayer
expand_bonddim
expand_internal_bonddim
```

## Measuring physical quantities
```@docs
expect
scalingdimensions
densitymatrix_entropies
```

## Ascending operators
```@docs
ascend
ascended_operator
scale_invariant_operator_sum
```

## Descending density matrices
```@docs
descend
densitymatrix
densitymatrices
fixedpoint_densitymatrix
```

## Optimizing a MERA
```@docs
minimize_expectation
environment
gradient
retract
transport!
inner
tensorwise_sum
tensorwise_scale
```

## Index
```@index
```
