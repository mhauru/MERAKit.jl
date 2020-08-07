# The Layer abstract type and empty function definitions, to be expanded by subtypes.
# To be included in MERA.jl

"""
Abstract supertype of all layer types, e.g. `BinaryLayer` and `TernaryLayer`.
"""
abstract type Layer end

"""
    randomlayer(::Type{T <: Layer}, T, Vin, Vout, Vint=Vout; random_disentangler=false)

Return a MERA layer with random tensors.

`T` is the `Layer` type, and `Vin` and `Vout` are the input and output spaces. `Vint` is an
internal vector space for the layer, connecting the disentanglers to the isometries. If
`random_disentangler=true`, the disentangler is also a random unitary, if `false` (default),
it is the identity or the product of two single-site isometries, depending on if the
disentanler is supposed to be unitary or isometric.

Each subtype of `Layer` should have its own method for this function.
"""
function randomlayer end

"""
    expand_inputspace(layer::Layer, V_new)

Return a new layer where the tensors have been padded with zeros as necessary to change the
input space to be `V_new`.

Each subtype of `Layer` should have its own method for this function.
"""
function expand_inputspace end

"""
    expand_outputspace(layer::Layer, V_new)

Return a new layer where the tensors have been padded with zeros as necessary to change the
output space to be `V_new`.

Each subtype of `Layer` should have its own method for this function.
"""
function expand_outputspace end

"""
    expand_internalspace(layer::Layer, V_new)

Return a new layer where the tensors have been padded with zeros as necessary to change the
interal vector space to be `V_new`.

Each subtype of `Layer` that has an internal vector space should have its own method for
this function.
"""
function expand_internalspace end


"""
    space_invar_intralayer(layer::Layer)

Return `true` if the indices within `layer` are compatible with each other, false otherwise.
"""
function space_invar_intralayer end

"""
    space_invar_interlayer(layer::T, next_layer::T) where {T <: Layer}

Return true if the indices between the two layers are compatible with each other,
false otherwise. `layer` is below `next_layer`.
"""
function space_invar_interlayer end

"""
    ascending_fixedpoint(layer::Layer)

Return the operator that is the fixed point of the average ascending superoperator of this
layer, normalised to have norm 1.
"""
function ascending_fixedpoint end

"""
    ascend(op, layer::Layer)
Ascend a local operator `op` from the bottom of `layer` to the top.
"""
function ascend end

"""
    descend(rho, layer::Layer)

Descend a local density matrix `rho` from the top of `layer` to the bottom.
"""
function descend end

"""
    environment(layer::Layer, op, rho; vary_disentanglers=true)

Compute the environments with respect to `op` of all the tensors in the layer, and return
them as a `Layer`. `rho` is the local density matrix at the top indices of this layer.

If `vary_disentanglers=false`, only compute the environments for the isometries, and set the
environments for the disentanglers to zero.
"""
function environment end

"""
    minimize_expectation_ev(layer::Layer, env::Layer; vary_disentanglers=true)

Return a new layer that minimizes the expectation value with respect to the environment
`env`, using an Evenbly-Vidal update.

If `vary_disentanglers = false`, only update the isometries.
"""
function minimize_expectation_ev end