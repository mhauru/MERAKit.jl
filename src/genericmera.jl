# The most important types of the package, GenericMERA and Layer, on which all specific MERA
# implementations (binary, ternary, ...) are built. Methods and functions that can be
# implemented on this level of abstraction, without having to know the details of the
# specific MERA type.
# To be included in MERA.jl.

"""
Layer is an abstract supertype of concrete types such as BinaryLayer. A typical Layer is a
collection of tensors, the orders and shapes of which depend on the type.
"""
abstract type Layer end

"""
A GenericMERA is a collection of Layers. The type of these layers then determines whether
the MERA is binary, ternary, etc.

A few notes on conventions and terminology:
- The physical indices of the MERA are at the "bottom", the scale invariant part at the
"top".
- The counting of layers starts from the bottom, so the layer with physical indices is layer
#1. The last layer is the scale invariant one, that then repeats upwards to infinity.
- Each layer is thought of as a linear map from its top, or input space to its bottom, or
output space.
"""
struct GenericMERA{T}
    layers::Vector{T}
    stored_densitymatrices::Vector
    stored_operators::Dict{Any, Vector}
    stored_environments::Dict{Any, Vector{Union{Nothing, T}}}
    # previous_fixedpoint_densitymatrix and previous_operatorsum will only ever store one
    # value, but we store them in a Vector, since that matches with the rest of the storage
    # format and keeps GenericMERA immutable.
    previous_fixedpoint_densitymatrix::Vector
    previous_operatorsum::Vector
    # Prevent the creation of GenericMERA{T} objects when T is not a subtype of Layer.
    function GenericMERA{T}(layers) where T <: Layer
        num_layers = length(layers)
        stored_densitymatrices = Vector{Any}(repeat([nothing], num_layers))
        stored_operators = Dict{Any, Vector}()
        stored_environments = Dict{Any, Vector{Union{Nothing, T}}}()
        previous_fixedpoint_densitymatrix = Any[nothing]
        previous_operatorsum = Any[nothing]
        new(collect(layers), stored_densitymatrices, stored_operators, stored_environments,
            previous_fixedpoint_densitymatrix, previous_operatorsum)
    end
end

# # # Basic utility functions

"""
Return the type of the layers of `m`.
"""
layertype(m::T) where {T <: GenericMERA} = layertype(T)
layertype(::Type{GenericMERA{T}}) where {T <: Layer} = T

Base.eltype(m::GenericMERA) = reduce(promote_type, map(eltype, m.layers))

"""
The ratio by which the number of sites changes when go down by a layer.
"""
scalefactor(::Type{GenericMERA{T}}) where T = scalefactor(T)

"""
Each MERA has a stable width causal cone, that depends on the type of layers the MERA has.
Return that width.
"""
causal_cone_width(::Type{GenericMERA{T}}) where T <: Layer = causal_cone_width(T)

"""
Return the number of transition layers, i.e. layers below the scale invariant one, in the
MERA.
"""
num_translayers(m::GenericMERA) = length(m.layers)-1

"""
Return the layer at the given depth. 1 is the lowest layer, i.e. the one with physical
indices.
"""
get_layer(m::GenericMERA, depth) = (depth > num_translayers(m) ?
                                    m.layers[end] : m.layers[depth])

"""
Set, in-place, one of the layers of the MERA to a new, given value. If check_invar=true,
check that the indices match afterwards.
"""
function set_layer!(m::GenericMERA, layer, depth; check_invar=true)
    depth > num_translayers(m) ? (m.layers[end] = layer) : (m.layers[depth] = layer)
    m = reset_storage!(m, depth)
    check_invar && space_invar(m)
    return m
end

"""
Add one more transition layer at the top of the MERA, by taking the lowest of the scale
invariant one and releasing it to vary independently.
"""
function release_transitionlayer!(m::GenericMERA)
    layer = get_layer(m, Inf)
    layer = copy(layer)
    push!(m.layers, layer)
    density_matrix = m.stored_densitymatrices[end]
    push!(m.stored_densitymatrices, density_matrix)
    for (k, v) in m.stored_environments
        push!(v, nothing)
    end
    return m
end

"""
Project all the tensors of the MERA to respect the isometricity condition.
"""
function TensorKitManifolds.projectisometric(m::T) where T <: GenericMERA
    return T(map(projectisometric, m.layers))
end

function TensorKitManifolds.projectisometric!(m::T) where T <: GenericMERA
    return T(map(projectisometric!, m.layers))
end

"""
Given a MERA and a depth, return the vector space of the downwards-pointing (towards the
physical level) indices of the layer at that depth.
"""
outputspace(m::GenericMERA, depth) = outputspace(get_layer(m, depth))

"""
Given a MERA and a depth, return the vector space of the upwards-pointing (towards scale
invariance) indices of the layer at that depth.
"""
inputspace(m::GenericMERA, depth) = inputspace(get_layer(m, depth))

"""
Compute the von Neumann entropy of a density matrix `rho`.
"""
function densitymatrix_entropy(rho)
    eigs = eigh(rho)[1]
    eigs = real.(diag(convert(Array, eigs)))
    if sum(abs.(eigs[eigs .<= 0.])) > 1e-13
        @warn("Significant negative eigenvalues for a density matrix: $eigs")
    end
    eigs = eigs[eigs .> 0.]
    S = -dot(eigs, log.(eigs))
    return S
end

densitymatrix_entropies(m::GenericMERA) = map(densitymatrix_entropy, densitymatrices(m))

# # # Storage of density matrices, ascended operators, and environments

# The storage formats for density matrices, operators, and environments are a little
# different: For density matrices we always store a Vector of the same length as there are
# layers. If a density matrix is not in store, `nothing` is kept as the corresponding
# element. For operators, m.stored_operators is a dictionary with operators as keys, and
# each element is a Vector that lists the ascended versions of that operator, starting from
# the physical one, which is nothing but the original operator. No place-holder `nothing`s
# are stored, the Vector ends when storage ends. For environments, m.stored_environments is
# a Dict just like m.stored_operators, but its values are Vectors that hold either Layers
# are Nothings, where each Layer has instead of tensors of the layer, environments for the
# tensors of a layer.
# To understand the reason for these differences, note that density matrices are naturally
# generated from the top, and there will never be more than the number of layers of them.
# Similarly it doesn't make sense to have more than one environment for each layer.
# Operators are generated from the bottom, and they may go arbitrarily high in the MERA
# (into the scale invariant part).

"""
Reset stored density matrices and ascended operators, so that they will be recomputed when
they are needed. By default all are reset. If the optional argument `depth` is given, only
ones invalidated by changing the layer at `depth` are removed.
"""
function reset_storage!(m::GenericMERA, depth)
    depth = Int(min(num_translayers(m)+1, depth))
    m.stored_densitymatrices[1:depth] .= nothing
    for (k, v) in m.stored_operators
        last_index = min(depth, length(v))
        m.stored_operators[k] = v[1:last_index]
    end
    # Changing anything always invalidates all environments, since they depend on things
    # both above and below.
    reset_environment_storage!(m)
    return m
end

function reset_storage!(m::GenericMERA)
    m.stored_densitymatrices[:] .= nothing
    reset_operator_storage!(m)
    reset_environment_storage!(m)
    return m
end

"""
Return whether the density matrix at given depth is already in store.
"""
function has_densitymatrix_stored(m::GenericMERA, depth)
    depth = Int(min(num_translayers(m)+1, depth))
    rho = m.stored_densitymatrices[depth]
    return rho !== nothing
end

"""
Return the density matrix at given depth, assuming it's in store.
"""
function get_stored_densitymatrix(m::GenericMERA, depth)
    depth = Int(min(num_translayers(m)+1, depth))
    rho = m.stored_densitymatrices[depth]
    if rho === nothing
        msg = "Density matrix at depth $(depth) not in storage."
        throw(ArgumentError(msg))
    end
    return rho
end

"""
Store the density matrix at given depth.
"""
function set_stored_densitymatrix!(m::GenericMERA, density_matrix, depth)
    m.stored_densitymatrices[depth] = density_matrix
    return m
end

"""
Return the stored ascended versions of `op`. Initialize storage for `op` if necessary.
"""
function operator_storage(m::GenericMERA, op)
    if !(op in keys(m.stored_operators))
        m.stored_operators[op] = Vector{Any}([op])
    end
    return m.stored_operators[op]
end

"""
Return whether the operator `op` ascended to `depth` is already in store.
"""
function has_operator_stored(m::GenericMERA, op, depth)
    storage = operator_storage(m, op)
    return length(storage) >= depth
end

"""
Return the operator `op` ascended to a given depth, assuming it's in store.
"""
function get_stored_operator(m::GenericMERA, op, depth)
    storage = operator_storage(m, op)
    return storage[depth]
end

"""
Store `opasc`, the operator `op` ascended to a given depth.
"""
function set_stored_operator!(m::GenericMERA, opasc, op, depth)
    storage = operator_storage(m, op)
    if length(storage) < depth-1
        msg = "Can't store an ascended operator if the lower versions of it aren't in storage already."
        throw(ArgumentError(msg))
    elseif length(storage) == depth-1
        push!(storage, opasc)
    else
        storage[depth] = opasc
    end
    return m
end

"""
Reset storage for a given operator.
"""
function reset_operator_storage!(m::GenericMERA, op)
    delete!(m.stored_operators, op)
    return m
end

"""
Reset storage for all operators.
"""
function reset_operator_storage!(m::GenericMERA)
    ops = keys(m.stored_operators)
    for op in ops
        reset_operator_storage!(m, op)
    end
    return m
end

"""
Return the environments related to `op`. Initialize storage for `op` if necessary.
"""
function environment_storage(m::GenericMERA{T}, op) where T
    if !(op in keys(m.stored_environments))
        num_layers = length(m.layers)
        m.stored_environments[op] = repeat(Union{Nothing, T}[nothing], num_layers)
    end
    return m.stored_environments[op]
end

"""
Return whether the environment related to `op` at `depth` is already in store.
"""
function has_environment_stored(m::GenericMERA, op, depth)
    storage = environment_storage(m, op)
    return storage[depth] !== nothing
end

"""
Return the environment related to `op` at `depth`. Return `nothing` is this environment is
not in store.
"""
function get_stored_environment(m::GenericMERA, op, depth)
    storage = environment_storage(m, op)
    return storage[depth]
end

"""
Store `env`, the environments related to `op` at `depth`.
"""
function set_stored_environment!(m::GenericMERA, env, op, depth)
    storage = environment_storage(m, op)
    storage[depth] = env
    return m
end

"""
Reset storage for environments.
"""
function reset_environment_storage!(m::GenericMERA)
    ops = keys(m.stored_environments)
    for op in ops
        delete!(m.stored_environments, op)
    end
    return m
end

# # # Generating random MERAs

"""
Generate a random MERA of type `T`, with `Vs` as the vector spaces of the various layers.
Number of layers will be the length of `Vs`, the `Vs[1]` will be the physical index space,
and `Vs[end]` will be the one at the scale invariant layer. An additional positional
argument `Ws` can be a vector/tuple of the same length as `Vs`, and contain additional
parameters passed to the constructor for each layer, e.g. additional intralayer bond
dimension. Also passed to the constructor for individual layers will be any additional
keyword arguments, but these will all be the same for each layer.
"""
function random_MERA(::Type{T}, Vs, Ws=nothing; kwargs...) where T <: GenericMERA
    num_layers = length(Vs)
    layers = []
    for i in 1:num_layers
        V = Vs[i]
        Vnext = (i < num_layers ? Vs[i+1] : V)
        if Ws === nothing
            layer = randomlayer(layertype(T), Vnext, V; kwargs...)
        else
            layer = randomlayer(layertype(T), Vnext, V, Ws[i]; kwargs...)
        end
        push!(layers, layer)
    end
    m = T(layers)
    return m
end

"""
Return a MERA layer with random tensors. The signature looks like this:
randomlayer(::Type{T}, Vin, Vout, W=nothing; kwargs...) where T <: Layer The first argument
is the `Layer` type, and the second and third are the input and output spaces. `W` and
`kwargs...` may contain any extra data needed, that depends on which layer type this is for.
Each subtype of `Layer` should have its own method for this function.
"""
function randomlayer end

"""
Expand the bond dimension of the MERA at the given depth. `depth=1` is the first virtual
level, just above the first layer of the MERA, and the numbering grows from there. The new
bond dimension is given by `newdims`, which for a non-symmetric MERA is just a number, and
for a symmetric MERA is a dictionary of {irrep => block dimension}. Not all irreps for a
bond need to be listed, the ones left out are left untouched.

The expansion is done by padding tensors with zeros. Note that this breaks isometricity of
the individual tensors. This is however of no consequence, since the MERA as a state remains
exactly the same. A round of optimization on the MERA will restore isometricity of each
tensor.
"""
function expand_bonddim!(m::GenericMERA, depth, newdims; check_invar=true)
    V = inputspace(m, depth)
    V = expand_vectorspace(V, newdims)

    layer = get_layer(m, depth)
    layer = expand_inputspace(layer, V)
    set_layer!(m, layer, depth; check_invar=false)

    next_layer = get_layer(m, depth+1)
    next_layer = expand_outputspace(next_layer, V)
    if depth == num_translayers(m)
        # next_layer is the scale invariant part, so we need to change its top
        # index too since we changed the bottom.
        next_layer = expand_inputspace(next_layer, V)
        # Pad the stored scale invariant initial guesses as well, when padding the scale
        # invariant isometries.
        old_rho = m.previous_fixedpoint_densitymatrix[1]
        if old_rho !== nothing
            width = causal_cone_width(typeof(m))
            for i in 1:width
                old_rho = pad_with_zeros_to(old_rho, i => V, (i+width) => V')
            end
            m.previous_fixedpoint_densitymatrix[1] = old_rho
        end
        old_opsum = m.previous_operatorsum[1]
        if old_opsum !== nothing
            width = causal_cone_width(typeof(m))
            for i in 1:width
                old_opsum = pad_with_zeros_to(old_opsum, i => V, (i+width) => V')
            end
            m.previous_operatorsum[1] = old_opsum
        end
    elseif depth > num_translayers(m)
            msg = "expand_bonddim! called with too large depth. To change the scale invariant bond dimension, use depth=num_translayers(m)."
            throw(ArgumentError(msg))
    end
    set_layer!(m, next_layer, depth+1; check_invar=check_invar)
end

"""
Expand the bond dimension of the layer-internal indices of the MERA at the given depth. The
new bond dimension is given by `newdims`, which for a non-symmetric MERA is just a number,
and for a symmetric MERA is a dictionary of {irrep => block dimension}. Not all irreps for a
bond need to be listed, the ones left out are left untouched.

The expansion is done by padding tensors with zeros. Note that this breaks isometricity of
the individual tensors. This is however of no consequence, since the MERA as a state remains
exactly the same. A round of optimization on the MERA will restore isometricity of each
tensor.

Note that not all MERAs have an internal bond dimension, and some may have several, so this
function will not make sense for all MERA types.
"""
function expand_internal_bonddim!(m::GenericMERA, depth, newdims; check_invar=true)
    V = internalspace(m, depth)
    V = expand_vectorspace(V, newdims)
    layer = get_layer(m, depth)
    layer = expand_internalspace(layer, V)
    set_layer!(m, layer, depth; check_invar=check_invar)
end

"""
Return a new layer where the tensors have been padded with zeros as necessary to change the
input space. The first argument is the layer, the second one is the new input space. Each
subtype of Layer should have its own method for this function.
"""
function expand_inputspace end

"""
Return a new layer where the tensors have been padded with zeros as necessary to change the
output space. The first argument is the layer, the second one is the new output space. Each
subtype of Layer should have its own method for this function.
"""
function expand_outputspace end

"""
Given a MERA which may possibly be built of symmetry preserving TensorMaps, return another
MERA that has the symmetry structure stripped from it, and all tensors are dense.
"""
remove_symmetry(m::T) where T <: GenericMERA = T(map(remove_symmetry, m.layers))

# # # Pseudo(de)serialization
# "Pseudo(de)serialization" refers to breaking the MERA down into types in Julia Base, and
# constructing it back. This can be used for storing MERAs on disk.
# TODO Once JLD or JLD2 works properly we should be able to get rid of this.
#
# Note that pseudoserialization discards stored_densitymatrices, stored_operators, and
# stored_environments, which then need to be recomputed after deserialization.

"""
Return a tuple of objects that can be used to reconstruct a given MERA, and that are all of
Julia base types.
"""
pseudoserialize(m::T) where T <: GenericMERA = (repr(T), map(pseudoserialize, m.layers))

"""
Reconstruct a MERA given the output of `pseudoserialize`.
"""
depseudoserialize(::Type{T}, args) where T <: GenericMERA = T([depseudoserialize(d...)
                                                               for d in args])

# Implementations for pseudoserialize(l::T) and depseudoserialize(::Type{T}, args...) should
# be written for each T <: Layer. They can make use of the (de)pseudoserialize methods for
# TensorMaps from `tensortools.jl`.

# # # Invariants

"""
Check that the indices of the various tensors in a given MERA are compatible with each
other. If not, throw an ArgumentError. If yes, return true. This relies on two checks,
`space_invar_intralayer` for checking indices within a layer and `space_invar_interlayer`
for checking indices between layers. These should be implemented for each subtype of Layer.
"""
function space_invar(m::GenericMERA{T}) where T
    layer = get_layer(m, 1)
    # We go to num_translayers(m)+2, to check that the scale invariant layer is consistent
    # with itself.
    for i in 2:(num_translayers(m)+2)
        next_layer = get_layer(m, i)
        if applicable(space_invar_intralayer, layer)
            if !space_invar_intralayer(layer)
                errmsg = "Mismatching bonds in MERA within layer $(i-1)."
                throw(ArgumentError(errmsg))
            end
        else
            msg = "space_invar_intralayer has no method for type $(T). We recommend writing one, to enable checking for space mismatches when assigning tensors."
            @warn(msg)
        end

        if applicable(space_invar_interlayer, layer, next_layer)
            if !space_invar_interlayer(layer, next_layer)
                errmsg = "Mismatching bonds in MERA between layers $(i-1) and $i."
                throw(ArgumentError(errmsg))
            end
        else
            msg = "space_invar_interlayer has no method for type $(T). We recommend writing one, to enable checking for space mismatches when assigning tensors."
            @warn(msg)
        end
        layer = next_layer
    end
    return true
end

"""
Given a Layer, return true if the indices within the layer are compatible with each other,
false otherwise.

This method should be implemented separately for each subtype of Layer.
"""
function space_invar_intralayer end

"""
Given two Layers, return true if the indices between them are compatible with each other,
false otherwise. The first argument is the layer below the second one.

This method should be implemented separately for each subtype of Layer.
"""
function space_invar_interlayer end

# # # Scaling functions

"""
Ascend the operator `op`, that lives on the layer `startscale`, through the MERA to the
layer `endscale`. Living "on" a layer means living on the indices right below it, so
`startscale=1` (the default) refers to the physical indices, and
`endscale=num_translayers(m)+1` (the default) to the indices just below the first scale
invariant layer.
"""
function ascend(op, m::GenericMERA; endscale=num_translayers(m)+1, startscale=1)
    if endscale < startscale
        throw(ArgumentError("endscale < startscale"))
    elseif endscale > startscale
        op = ascend(op, m; endscale=endscale-1, startscale=startscale)
        layer = get_layer(m, endscale-1)
        op = ascend(op, layer)
    end
    return op
end

"""
Descend the operator `op`, that lives on the layer `startscale`, through the MERA to the
layer `endscale`. Living "on" a layer means living on the indices right below it, so
`endscale=1` (the default) refers to the physical indices, and
`startscale=num_translayers(m)+1` (the default) to the indices just below the first scale
invariant layer.
"""
function descend(op, m::GenericMERA; endscale=1, startscale=num_translayers(m)+1)
    if endscale > startscale
        throw(ArgumentError("endscale > startscale"))
    elseif endscale < startscale
        op = descend(op, m; endscale=endscale+1, startscale=startscale)
        layer = get_layer(m, endscale)
        op = descend(op, layer)
    end
    return op
end

"""
Find the fixed point density matrix of the scale invariant part of the MERA.
"""
function fixedpoint_densitymatrix(m::GenericMERA, pars=Dict())
    f(x) = descend(x, m; endscale=num_translayers(m)+1, startscale=num_translayers(m)+2)
    # If we have stored the previous fixed point density matrix, and it has the right
    # dimensions, use that as the initial guess. Else, use a thermal density matrix.
    x0 = thermal_densitymatrix(m, Inf)
    old_rho = m.previous_fixedpoint_densitymatrix[1]
    if old_rho !== nothing && space(x0) == space(old_rho)
        x0 = old_rho
    end
    eigsolve_pars = get(pars, :scaleinvariant_krylovoptions, Dict())
    vals, vecs, info = eigsolve(f, x0, 1; eigsolve_pars...)
    rho = vecs[1]
    # We know the result should always be Hermitian, and scaled to have trace 1.
    rho = (rho + rho') / 2.0
    rho /= tr(rho)
    if eltype(m) <: Real
        # rho isn't generally real for generic matrices, but we know that it should be for
        # the descending superoperator.
        imag_norm = norm(imag(rho))
        if imag_norm > 1e-12
            msg = "The fixed point density matrix has a significant imaginary part, that we discard: $(imag_norm)"
            @warn(msg)
        end
        rho = real(rho)
    end
    m.previous_fixedpoint_densitymatrix[1] = rho
    if :verbosity in keys(pars) && pars[:verbosity] > 3
        msg = "Used $(info.numops) superoperator invocations to find the fixed point density matrix."
        @info(msg)
    end
    return rho
end

"""
Return the thermal density matrix for the indices right below the layer at `depth`. Used as
an initial guess for the fixed-point density matrix.
"""
function thermal_densitymatrix(m::GenericMERA, depth)
    V = inputspace(m, Inf)
    width = causal_cone_width(typeof(m))
    eye = id(Matrix{eltype(m)}, V)
    rho = ⊗(repeat([eye], width)...)
    return rho
end

"""
Return the density matrix right below the layer at `depth`.

This method stores every density matrix in memory as it computes them, and fetches them from
there if the same one is requested again.
"""
function densitymatrix(m::GenericMERA, depth, pars=Dict())
    if has_densitymatrix_stored(m, depth)
        rho = get_stored_densitymatrix(m, depth)
    else
        # If we don't find rho in storage, generate it.
        if depth > num_translayers(m)
            rho = fixedpoint_densitymatrix(m, pars)
        else
            rho_above = densitymatrix(m, depth+1, pars)
            rho = descend(rho_above, m; endscale=depth, startscale=depth+1)
        end
        # Store this density matrix for future use.
        set_stored_densitymatrix!(m, rho, depth)
    end
    return rho
end

"""
Return the density matrices starting for the layers from `lowest_depth` upwards up to and
including the scale invariant one.
"""
function densitymatrices(m::GenericMERA, pars=Dict())
    rhos = [densitymatrix(m, depth, pars) for depth in 1:num_translayers(m)+1]
    return rhos
end

"""
Return the operator `op` ascended from the physical level to `depth`. The benefit of using
this over `ascend(m, op; startscale=1, endscale=depth)` is that this one stores the result
in memory and fetches it from there if them same operator is requested again.
"""
function ascended_operator(m::GenericMERA, op, depth)
    # Note that if depth=1, has_operator_stored always returns true, as it initializes
    # storage for this operator.
    if has_operator_stored(m, op, depth)
        opasc = get_stored_operator(m, op, depth)
    else
        op_below = ascended_operator(m, op, depth-1)
        opasc = ascend(op_below, m; endscale=depth, startscale=depth-1)
        # Store this density matrix for future use.
        set_stored_operator!(m, opasc, op, depth)
    end
    return opasc
end

"""
Return the sum of the ascended versions of `op` in the scale invariant part of the MERA.
To be more precise, this sum is of course infinite, and what we return is the component of
it orthogonal to the dominant eigenoperator of the ascending superoperator (typically the
identity). This component converges to a finite number like a geometric series, since all
non-dominant eigenvalues of the ascending superoperator are smaller than 1.

To approximate the converging series, we use an iterative Krylov solver. The options for the
solver should be in a dictionary pars[:scaleinvariant_krylovoptions].
"""
function scale_invariant_operator_sum(m::GenericMERA, op, pars)
    nt = num_translayers(m)
    # fp is the dominant eigenvector of the ascending superoperator. We are not interested
    # in contributions to the sum along fp, since they will just be fp * infty, and fp is
    # merely the representation of the identity operator.
    fp = ascending_fixedpoint(get_layer(m, nt+1))
    function f(x)
        x = ascend(x, m; startscale=nt+1, endscale=nt+2)
        x = x - fp * dot(fp, x)
        return x
    end
    op_top = ascended_operator(m, op, nt+1)
    x0 = op_top
    old_opsum = m.previous_operatorsum[1]
    if old_opsum !== nothing && space(x0) == space(old_opsum)
        x0 = old_opsum
    end
    linsolve_pars = get(pars, :scaleinvariant_krylovoptions, Dict())
    one_ = one(eltype(m))
    opsum, info = linsolve(f, op_top, x0, one_, -one_; linsolve_pars...)
    # We know the result should always be Hermitian.
    opsum = (opsum + opsum') / 2.0
    m.previous_operatorsum[1] = opsum
    # We are not interested in the component along fp.
    opsum = opsum - fp * dot(fp, opsum)
    if :verbosity in keys(pars) && pars[:verbosity] > 3
        msg = "Used $(info.numops) superoperator invocations to find the scale invariant operator sum."
        @info(msg)
    end
    return opsum
end

"""
Return the environment related to `op` at `depth`. This function uses the cache to store the
environment and retrieve it from storage if it is already there.
"""
function environment(m::GenericMERA, op, depth, pars; vary_disentanglers=true)
    if has_environment_stored(m, op, depth)
        env = get_stored_environment(m, op, depth)
    else
        if depth <= num_translayers(m)
            op_below = ascended_operator(m, op, depth)
        else
            op_below = scale_invariant_operator_sum(m, op, pars)
        end
        op_below = normalise_hamiltonian(op_below)
        rho_above = densitymatrix(m, depth+1, pars)
        layer = get_layer(m, depth)
        env = environment(layer, op_below, rho_above; vary_disentanglers=vary_disentanglers)
        set_stored_environment!(m, env, op, depth)
    end
    return env
end

# # # Extracting CFT data

"""
Diagonalize the scale invariant ascending superoperator to compute the scaling dimensions of
the underlying CFT. The return value is a dictionary, the keys of which are symmetry sectors
for a possible internal symmetry of the MERA (Trivial() if there is no internal symmetry),
and values are scaling dimensions in this symmetry sector.
"""
function scalingdimensions(m::GenericMERA; howmany=20)
    V = inputspace(m, Inf)
    chi = dim(V)
    width = causal_cone_width(typeof(m))
    # Don't even try to get more than half of the eigenvalues. Its too expensive, and they
    # are garbage anyway.
    maxmany = Int(ceil(chi^width/2))
    howmany = min(maxmany, howmany)
    # Define a function that takes an operator and ascends it once through the scale
    # invariant layer.
    nm = num_translayers(m)
    f(x) = ascend(x, m; endscale=nm+2, startscale=nm+1)
    # Find out which symmetry sectors we should do the diagonalization in.
    interlayer_space = reduce(⊗, repeat([V], width))
    sects = sectors(fuse(interlayer_space))
    scaldim_dict = Dict()
    for irrep in sects
        # Diagonalize in each irrep sector.
        x0 = scalingoperator_initialguess(m, interlayer_space, irrep)
        S, U, info = eigsolve(f, x0, howmany, :LM)
        # sfact is the ratio by which the number of sites changes at each coarse-graining.
        sfact = scalefactor(typeof(m))
        scaldims = sort(-log.(sfact, abs.(real(S))))
        scaldim_dict[irrep] = scaldims
    end
    return scaldim_dict
end

"""
Return an initial guess to be used in the iterative eigensolver that solves for scaling
operators.
"""
function scalingoperator_initialguess(m::GenericMERA, interlayer_space, irrep)
    typ = eltype(m)
    inspace = interlayer_space
    outspace = interlayer_space
    # If this is a non-trivial irrep sector, expand the input space with a dummy leg.
    irrep !== Trivial() && (inspace = inspace ⊗ spacetype(inspace)(irrep => 1))
    # The initial guess for the eigenvalue search. Also defines the type for
    # eigenvectors.
    x0 = TensorMap(randn, typ, outspace ← inspace)
    return x0
end

# # # Evaluation

"""
Return the expecation value of operator `op` for this MERA. The layer on which `op` lives is
set by `opscale`, which by default is the physical one (`opscale=1`). `evalscale` can be
used to set whether the operator is ascended through the network or the density matrix is
descended.
"""
function expect(op, m::GenericMERA, pars=Dict(); opscale=1, evalscale=1)
    rho = densitymatrix(m, evalscale, pars)
    op = ascended_operator(m, op, evalscale)
    # If the operator is defined on a smaller support (number of sites) than rho, expand it.
    op = expand_support(op, support(rho))
    value = dot(rho, op)
    if abs(imag(value)/norm(op)) > 1e-13
        @warn("Non-real expectation value: $value")
    end
    value = real(value)
    return value
end


# # # Optimization

default_pars = Dict(:method => :lbfgs,
                    :isometrymanifold => :grassmann,
                    :retraction => :exp,
                    :transport => :exp,
                    :metric => :euclidean,
                    :precondition => true,
                    :gradient_delta => 1e-14,
                    :isometries_only_iters => 0,
                    :maxiter => 2000,
                    :ev_layer_iters => 1,
                    :ls_epsilon => 1e-6,
                    :lbfgs_m => 8,
                    :cg_flavor => :HagerZhang,
                    :verbosity => 2,
                    :scaleinvariant_krylovoptions => Dict(
                                                          :tol => 1e-13,
                                                          :krylovdim => 4,
                                                          :verbosity => 0,
                                                          :maxiter => 20,
                                                         ),
                   )

function minimize_expectation!(m, h, pars; finalize! = OptimKit._finalize!,
                               vary_disentanglers=true, kwargs...)
    pars = merge(default_pars, pars)
    # If pars[:isometries_only_iters] is set, but the optimization on the whole is supposed
    # to vary_disentanglers too, then first run a pre-optimization without touching the
    # disentanglers, with pars[:isometries_only_iters] as the maximum iteration count,
    # before moving on to the main optimization with all tensors varying.
    if vary_disentanglers && pars[:isometries_only_iters] > 0
        temp_pars = deepcopy(pars)
        temp_pars[:maxiter] = pars[:isometries_only_iters]
        m = minimize_expectation!(m, h, temp_pars; finalize! = finalize!, vary_disentanglers=false, kwargs...)
    end

    method = pars[:method]
    if method in (:cg, :conjugategradient, :gd, :gradientdescent, :lbfgs)
        return minimize_expectation_grad!(m, h, pars; vary_disentanglers=vary_disentanglers,
                                          finalize! = finalize!, kwargs...)
    elseif method == :ev || method == :evenblyvidal
        return minimize_expectation_ev!(m, h, pars; vary_disentanglers=vary_disentanglers,
                                        finalize! = finalize!, kwargs...)
    else
        msg = "Unknown optimization method $(method)."
        throw(ArgumentError(msg))
    end
end

"""
Optimize the MERA `m` to minimize the expectation value of `h` using the Evenbly-Vidal
method.

The optimization proceeds by looping over layers and optimizing each in turn, starting from
the bottom, and repeating this until convergence is reached.

The keyword argument `lowest_depth` sets the lowest layer in the MERA that the optimization
is allowed to change, by default `lowest=1` so all layers are optimized.

The argument `pars` is a dictionary with symbols as keys, that lists values for different
parameters for how to do the optimization. They all need to be specified by the user, and
have no default values. The different parameters are:
    :miniter, a minimum number of iterations to do over the layers before returning. The
     first half of the minimum number of iterations will be spent optimizing only isometric
     tensors, leaving the disentanglers untouched.
    :maxiter, the maximum number of iterations to do over the layers before returning.
    :densitymatrix_delta, the convergence threshold. Convergence is measured as changes in
     the reduced density matrices of the MERA. Once the maximum by which any of them changed
     over consecutive iterations falls below pars[:densitymatrix_delta], the optimization
     terminates. Change is measured as Frobenius norm of the difference. Note that this is
     much more demanding than convergence in just the expectation value.
    Other parameters may be required depending on the type of MERA. See documentation for
    the different Layer types. Typical parameters are for instance how many times to iterate
    optimizing individual tensors.
"""
function minimize_expectation_ev!(m, h, pars; finalize! = OptimKit._finalize!,
                                  lowest_depth=1, vary_disentanglers=true)
    nt = num_translayers(m)
    rhos = densitymatrices(m, pars)
    expectation = expect(h, m, pars)
    rhos_maxchange = Inf
    gradnorm = Inf
    counter = 0
    if pars[:verbosity] >= 2
        @info(@sprintf("E-V: initializing with f = %.12f,", expectation))
    end
    while gradnorm > pars[:gradient_delta] && counter < pars[:maxiter]
        counter += 1
        old_rhos = rhos
        old_expectation = expectation
        gradnorm_sq = 0.0

        for l in lowest_depth:nt+1
            env, layer = nothing, nothing
            for i in 1:pars[:ev_layer_iters]
                env = environment(m, h, l, pars; vary_disentanglers=vary_disentanglers)
                layer = get_layer(m, l)
                new_layer = minimize_expectation_ev(layer, env, pars;
                                                    vary_disentanglers=vary_disentanglers)
                set_layer!(m, new_layer, l)
            end
            # We use the latest env and the corresponding layer to compute the norm of the
            # gradient. This isn't quite the gradient at the end point, which is what we
            # would want, but close enough.
            gradnorm_sq += gradient_normsq(layer, env;
                                           isometrymanifold=pars[:isometrymanifold],
                                           metric=pars[:metric])
        end

        gradnorm = sqrt(gradnorm_sq)
        rhos = densitymatrices(m, pars)
        expectation = expect(h, m, pars)
        if old_rhos !== nothing
            rho_diffs = [norm(r - or) for (r, or) in zip(rhos, old_rhos)]
            rhos_maxchange = maximum(rho_diffs)
        end
        # The nothing is for the gradient, which we don't use, but OptimKit's format of
        # finalize! expects.
        m = finalize!(m, expectation, nothing, counter)[1]
        if pars[:verbosity] >= 2
            @info(@sprintf("E-V: iter %4d: f = %.12f, ‖∇f‖ = %.4e, max‖Δρ‖ = %.4e",
                           counter, expectation, gradnorm, rhos_maxchange))
        end
    end
    if pars[:verbosity] > 0
        if gradnorm <= pars[:gradient_delta]
            @info(@sprintf("E-V: converged after %d iterations: f = %.12f, ‖∇f‖ = %.4e, max‖Δρ‖ = %.4e",
                           counter, expectation, gradnorm, rhos_maxchange))
        else
            @warn(@sprintf("E-V: not converged to requested tol: f = %.12f, ‖∇f‖ = %.4e, max‖Δρ‖ = %.4e",
                           expectation, gradnorm, rhos_maxchange))
        end
    end
    return m
end

"""
Change the additive normalisation of a local Hamiltonian term to make it suitable for
computing an E-V update environment.
"""
function normalise_hamiltonian(h)
    lb, ub = gershgorin_bounds(h)
    eye = id(domain(h))
    return h - ub*eye
end

# # # Gradient optimization

function istangent(m::T, mtan::T) where T <: GenericMERA
    n = max(num_translayers(m), num_translayers(mtan)) + 1
    return all([istangent(get_layer(m, i), get_layer(mtan, i)) for i in 1:n])
end

function tensorwise_scale(m::T, alpha) where T <: GenericMERA
    return T([tensorwise_scale(l, alpha) for l in m.layers])
end

function tensorwise_sum(m1::T, m2::T) where T <: GenericMERA
    n = max(num_translayers(m1), num_translayers(m2)) + 1
    layers = [tensorwise_sum(get_layer(m1, i), get_layer(m2, i)) for i in 1:n]
    return T(layers)
end

function TensorKitManifolds.inner(m::T, m1::T, m2::T; metric=:euclidean
                                 ) where T <: GenericMERA
    n = max(num_translayers(m1), num_translayers(m2)) + 1
    res = sum([inner(get_layer(m, i), get_layer(m1, i), get_layer(m2, i); metric=metric)
               for i in 1:n])
    return res
end

function gradient(h, m::T, pars; vary_disentanglers=true, kwargs...) where T <: GenericMERA
    kwargs = Dict{Symbol, Any}(kwargs)
    :metric ∉ keys(kwargs) && (kwargs[:metric] = pars[:metric])
    :isometrymanifold ∉ keys(kwargs) && (kwargs[:isometrymanifold] = pars[:isometrymanifold])
    nt = num_translayers(m)
    layers = []
    for l in 1:nt+1
        layer = get_layer(m, l)
        env = environment(m, h, l, pars; vary_disentanglers=vary_disentanglers)
        grad = gradient(layer, env; kwargs...)
        push!(layers, grad)
    end
    g = T(layers)
    return g
end

function precondition_tangent(m::T, tan::T, pars) where T <: GenericMERA
    nt = num_translayers(m)
    tanlayers_prec = []
    for l in 1:nt+1
        layer = get_layer(m, l)
        tanlayer = get_layer(tan, l)
        rho = densitymatrix(m, l+1, pars)
        tanlayer_prec = precondition_tangent(layer, tanlayer, rho)
        push!(tanlayers_prec, tanlayer_prec)
    end
    tan_prec = T(tanlayers_prec)
    return tan_prec
end

function TensorKitManifolds.retract(m::T, mtan::T, alpha::Real; kwargs...
                                   ) where T <: GenericMERA
    layers, layers_tan = zip([retract(l, ltan, alpha; kwargs...)
                              for (l, ltan) in zip(m.layers, mtan.layers)]...)
    return T(layers), T(layers_tan)
end

function TensorKitManifolds.transport!(mvec::T, m::T, mtan::T, alpha::Real, mend::T;
                                       kwargs...) where T <: GenericMERA
    layers = [transport!(lvec, l, ltan, alpha, lend; kwargs...)
              for (lvec, l, ltan, lend)
              in zip(mvec.layers, m.layers, mtan.layers, mend.layers)]
    return T(layers)
end

function minimize_expectation_grad!(m, h, pars; finalize! = OptimKit._finalize!,
                                    lowest_depth=1, vary_disentanglers=true)
    if lowest_depth != 1
        # TODO Could implement this. It's not hard, just haven't seen the need.
        msg = "lowest_depth != 1 has not been implemented for gradient optimization."
        throw(ArgumentError(msg))
    end

    function fg(x)
        f = expect(h, x, pars)
        g = gradient(h, x, pars; vary_disentanglers=vary_disentanglers)
        return f, g
    end

    rtrct(args...; kwargs...) = retract(args...; alg=pars[:retraction], kwargs...)
    trnsprt!(args...; kwargs...) = transport!(args...; alg=pars[:transport], kwargs...)
    innr(args...; kwargs...) = inner(args...; metric=pars[:metric], kwargs...)
    scale(vec, beta) = tensorwise_scale(vec, beta)
    add(vec1, vec2, beta) = tensorwise_sum(vec1, scale(vec2, beta))
    linesearch = HagerZhangLineSearch(; ϵ=pars[:ls_epsilon])
    if pars[:precondition]
        precondition(x, g) = precondition_tangent(x, g, pars)
    else
        # The default that does nothing.
        precondition = OptimKit._precondition
    end

    algkwargs = Dict(:maxiter => pars[:maxiter], :linesearch => linesearch,
                     :verbosity => pars[:verbosity], :gradtol => pars[:gradient_delta])
    if pars[:method] == :cg || pars[:method] == :conjugategradient
        if pars[:cg_flavor] == :HagerZhang
            flavor = HagerZhang()
        elseif pars[:cg_flavor] == :HestenesStiefel
            flavor = HestenesStiefel()
        elseif pars[:cg_flavor] == :PolakRibierePolyak
            flavor = PolakRibierePolyak()
        elseif pars[:cg_flavor] == :DaiYuan
            flavor = DaiYuan()
        elseif pars[:cg_flavor] == :FletcherReeves
            flavor = FletcherReeves()
        else
            msg = "Unknown conjugate gradient flavor $(pars[:cg_flavor])"
            throw(ArgumentError(msg))
        end
        alg = ConjugateGradient(; flavor=flavor, algkwargs...)
    elseif pars[:method] == :gd || pars[:method] == :gradientdescent
        alg = GradientDescent(; algkwargs...)
    elseif pars[:method] == :lbfgs
        alg = LBFGS(pars[:lbfgs_m]; algkwargs...)
    else
        msg = "Unknown optimization method $(pars[:method])."
        throw(ArgumentError(msg))
    end
    res = optimize(fg, m, alg; scale! = scale, add! = add, retract=rtrct,
                   inner=innr, transport! = trnsprt!, isometrictransport=true,
                   precondition=precondition, finalize! = finalize!)
    m, expectation, normgrad, normgradhistory = res
    if pars[:verbosity] > 0
        @info("Gradient optimization done. Expectation = $(expectation).")
    end
    return m
end
