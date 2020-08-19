# The most important type of the package, GenericMERA, on which all specific MERA
# implementations (binary, ternary, ...) are built. Methods and functions that can be
# implemented on this level of abstraction, without having to know the details of the
# specific MERA type.
# To be included in MERA.jl.

"""
    GenericMERA{N, LT <: Layer, OT}

A `GenericMERA` is a collection of `Layer`s. The type of these layers then determines
whether the MERA is binary, ternary, etc.

On conventions and terminology:
* The physical indices of the MERA are at the "bottom", the scale invariant part at the
  "top".
* The counting of layers starts from the bottom, so the layer with physical indices is
  layer 1. The last layer is the scale invariant one, that then repeats upwards to infinity.
* Each layer is thought of as a linear map from its top, or input space to its bottom, or
  output space.

The type parameters are
`N`:  The number of distinct layers (N-1 transition layers and one scale invariant one).
`LT`: Layer type.
`OT`: Operator type. The type of ascended and descended operators for this MERA. Determined
      from `LT`. Typically a `TensorMap` with input and output indices matching the causal
      cone width.

`GenericMERA` is immutable, and the layers can not be changed after construction. All
functions that modify a `GenericMERA` return new objects.
"""
struct GenericMERA{N, LT <: Layer, OT}
    # LT stands for Layer Type, OT for Operator Type.
    layers::NTuple{N, LT}
    cache::MERACache{N, LT, OT}

    function GenericMERA{N, LT, OT}(layers::NTuple{N}, cache::MERACache{N, LT, OT}
                                   ) where {N, LT, OT}
        # Note that this prevents the creation of types like GenericMERA{3, SimpleLayer}:
        # The second type parameter must be exactly the element type of layers, specified at
        # as a concrete type. This is intentional, to avoid accidentally creating
        # unnecessarily abstract types that would hamper inference.
        @assert LT === eltype(typeof(layers))
        @assert isconcretetype(LT)
        @assert OT === operatortype(LT)
        return new{N, LT, OT}(layers, cache)
    end
end

# # # Constructors

function GenericMERA(layers::NTuple{N}, cache::MERACache{N}) where {N}
    LT = eltype(typeof(layers))
    OT = operatortype(LT)
    cache::MERACache{N, LT, OT}
    return GenericMERA{N, LT, OT}(layers, cache)
end

function GenericMERA(layers::NTuple{N}) where {N}
    LT = eltype(typeof(layers))
    OT = operatortype(LT)
    cache = MERACache{N, LT, OT}()
    return GenericMERA(layers, cache)
end

GenericMERA(layers) = GenericMERA(tuple(layers...))

function GenericMERA{N, LT, OT}(layers::NTuple{N}) where {N, LT, OT}
    cache = MERACache{N, LT, OT}()
    return GenericMERA{N, LT, OT}(layers, cache)
end

function GenericMERA{N, LT, OT}(layers) where {N, LT, OT}
    return GenericMERA{N, LT, OT}(tuple(layers...))
end

function (::Type{GenericMERA{M, LT, OT} where M})(layers::NTuple{N}) where {N, LT, OT}
    return GenericMERA(layers)
end

function (::Type{GenericMERA{M, LT, OT} where M})(layers) where {LT, OT}
    return (GenericMERA where {N})(tuple(layers...))
end

# # # Basic utility functions

"""
    layertype(m::GenericMERA)
    layertype(::Type{<:GenericMERA})

Return the type of the layers of the MERA.
"""
layertype(::Type{GenericMERA{N, LT, OT} where N}) where {LT, OT} = LT
layertype(::Type{GenericMERA{N, LT, OT}}) where {N, LT, OT} = LT

"""
    baselayertype(m::GenericMERA)
    baselayertype(::Type{<:GenericMERA})

Return the generic type of the layers of the MERA, without specific type parameters
"""
baselayertype(::Type{GenericMERA{N, LT, OT} where N}) where {LT, OT} = baselayertype(LT)
baselayertype(::Type{GenericMERA{N, LT, OT}}) where {N, LT, OT} = baselayertype(LT)

"""
    operatortype(m::GenericMERA)
    operatortype(::Type{<: GenericMERA})

Return the type of operator associate with this MERA or MERA type. That means the type of
operator that fits in the causal cone, and is naturally emerges as one ascends local
operators.
"""
operatortype(::Type{GenericMERA{N, LT, OT} where N}) where {LT, OT} = OT
operatortype(::Type{GenericMERA{N, LT, OT}}) where {N, LT, OT} = OT
# we could also just do:
# operatortype(M::Type{<:GenericMERA}) = operatortype(layertype(M))


"""
    scalefactor(::Type{<: GenericMERA})

The ratio by which the number of sites changes when one descends by one layer, e.g. 2 for
binary MERA, 3 for ternary.
"""
scalefactor(M::Type{<:GenericMERA}) = scalefactor(layertype(M))

"""
    causal_cone_width(::Type{<: GenericMERA})

Return the width of the stable causal cone for this MERA type.
"""
causal_cone_width(M::Type{<:GenericMERA}) = causal_cone_width(layertype(M))

Base.eltype(M::Type{<:GenericMERA}) = eltype(layertype(M))

"""
    num_translayers(m::GenericMERA)

Return the number of transition layers, i.e. layers below the scale invariant one, in the
MERA.
"""
num_translayers(::Type{<:GenericMERA{N}}) where {N} = N-1
num_translayers(m::GenericMERA) = num_translayers(typeof(m))

"""
    get_layer(m::GenericMERA, depth)

Return the layer at the given depth. 1 is the lowest layer, i.e. the one with physical
indices.
"""
function get_layer(m::GenericMERA, depth)
    return (depth > num_translayers(m) ?  m.layers[end] : m.layers[depth])
end

"""
    replace_layer(m::GenericMERA, layer, depth; check_invar=true)

Replace `depth` layer of `m` with `layer`. If check_invar=true, check that the indices match
afterwards.
"""
function replace_layer(m::GenericMERA{N, LT}, layer::LT, depth; check_invar=true
                      ) where {N, LT}
    index = min(num_translayers(m)+1, depth)
    new_layers = Base.setindex(m.layers, layer, index)
    new_cache = replace_layer(m.cache, depth)
    new_m = GenericMERA(new_layers, new_cache)
    check_invar && space_invar(new_m)
    return new_m
end

"""
    release_transitionlayer(m::GenericMERA)

Add one more transition layer at the top of the MERA, by taking the lowest of the scale
invariant ones and releasing it to vary independently.
"""
function release_transitionlayer(m::GenericMERA)
    new_layers = (m.layers..., m.layers[end])
    new_cache = release_transitionlayer(m.cache)
    new_m = GenericMERA(new_layers, new_cache)
    return new_m
end

"""
    projectisometric(m::GenericMERA)

Project all the tensors of the MERA to respect the isometricity condition.
"""
function TensorKitManifolds.projectisometric(m::GenericMERA)
    return typeof(m)(map(projectisometric, m.layers))
end

"""
    projectisometric!(m::GenericMERA)

Project all the tensors of the MERA to respect the isometricity condition, in place.
"""
function TensorKitManifolds.projectisometric!(m::GenericMERA)
    return typeof(m)(map(projectisometric!, m.layers))
end

"""
    outputspace(m::GenericMERA, depth)

Given a MERA and a depth, return the vector space of the downwards-pointing (towards the
physical level) indices of the layer at that depth.

See also: [`inputspace`](@ref)
"""
outputspace(m::GenericMERA, depth) = outputspace(get_layer(m, depth))

"""
    inputspace(m::GenericMERA, depth)

Given a MERA and a depth, return the vector space of the upwards-pointing (towards scale
invariance) indices of the layer at that depth.

See also: [`outputspace`](@ref)
"""
inputspace(m::GenericMERA, depth) = inputspace(get_layer(m, depth))

"""
    densitymatrix_entropies(m::GenericMERA)

Return a vector of entropies for the density matrices in the MERA. The first one is for the
density matrix at the physical level, the last one is the scale invariant density matrix.
"""
function densitymatrix_entropies(m::GenericMERA)
    return [densitymatrix_entropy(x) for x in densitymatrices(m)]
end

"""
    reset_storage(m::GenericMERA)

Reset cached operators, so that they will be recomputed when they are needed.
"""
reset_storage(m::GenericMERA) = typeof(m)(m.layers, typeof(m.cache)())

# # # Generating random MERAs

"""
    random_MERA(::Type{T <: GenericMERA}, ET, Vouts, Vints=Vouts; kwargs...)

Generate a random MERA of type `T`.

`ET` is the element type of the MERA, e.g. `Float64`. `Vouts` are the vector spaces between
the various layers. Number of layers will be the length of `Vouts`. `Vouts[1]` will be the
physical index space, and `Vs[end]` will be the one at the scale invariant layer. `Vints`
can be a vector/tuple of the same length as `Vouts`, and include vector spaces for the
internal indices of each layer. Alternatively, it can hold any other extra parameters
specific to each layer, as it is simply passed on to the function `randomlayer`. Also passed
to `randomlayer` will be any additional keyword arguments, but these will all be the same
for each layer.

See also: [`randomlayer`](@ref)
"""
function random_MERA(::Type{T}, ET, Vouts, Vints=Vouts; kwargs...) where T <: GenericMERA
    L = layertype(T)
    Vins = tuple(Vouts[2:end]..., Vouts[end])
    layers = ntuple(length(Vouts)) do i
        randomlayer(L, ET, Vins[i], Vouts[i], Vints[i]; kwargs...)
    end
    m = GenericMERA(layers)
    return m
end

# TODO Should the numbering be changed, so that the bond at `depth` would be the
# output bond of layer `depth`, instead of input? Would maybe be more consistent.
# TODO Replace the newdims Dict thing with just a vector space. Or allow for both?
"""
    expand_bonddim(m::GenericMERA, depth, newdims; check_invar=true)

Expand the bond dimension of the MERA at the given depth.

The indices to expand are the input indices of the layer at `depth`, i.e. `depth=1` means
the lowest virtual indices. The new bond dimension is given by `newdims`, which for a
non-symmetric MERA is just a number, and for a symmetric MERA is a dictionary of `irrep =>
block dimension`. Not all irreps for a bond need to be listed, the ones left out are left
untouched.

The expansion is done by padding tensors with zeros. Note that this breaks isometricity of
the individual tensors. This is however of no consequence, since the MERA as a state remains
exactly the same. A round of optimization on the MERA will restore isometricity of each
tensor, or `projectisometric` can be called to do so explicitly.

If `check_invar = true` the function checks that the bond dimensions of various layers match
after the expansion.
"""
function expand_bonddim(m::GenericMERA, depth, newdims; check_invar=true)
    if depth > num_translayers(m)
        msg = "expand_bonddim called with too large depth. To change the scale invariant bond dimension, use depth=num_translayers(m)."
        throw(ArgumentError(msg))
    end
    V = inputspace(m, depth)
    V = expand_vectorspace(V, newdims)
    layer = get_layer(m, depth)
    layer = expand_inputspace(layer, V)
    next_layer = get_layer(m, depth+1)
    next_layer = expand_outputspace(next_layer, V)
    if depth == num_translayers(m)
        # next_layer is the scale invariant part, so we need to change its top index too
        # since we changed the bottom.
        next_layer = expand_inputspace(next_layer, V)
    end
    m = replace_layer(m, layer, depth; check_invar=false)
    m = replace_layer(m, next_layer, depth+1; check_invar=check_invar)
    expand_bonddim!(m.cache, depth, V)
    return m
end

"""
    expand_internal_bonddim(m::GenericMERA, depth, newdims; check_invar=true)

Expand the bond dimension of the layer-internal indices of the MERA at the given depth.

The new bond dimension is given by `newdims`, which for a non-symmetric MERA is just a
number, and for a symmetric MERA is a dictionary of {irrep => block dimension}. Not all
irreps for a bond need to be listed, the ones left out are left untouched.

The expansion is done by padding tensors with zeros. Note that this breaks isometricity of
the individual tensors. This is however of no consequence, since the MERA as a state remains
exactly the same. A round of optimization on the MERA will restore isometricity of each
tensor, or `projectisometric` can be called to do so explicitly.

Note that not all MERAs have an internal bond dimension, and some may have several, so this
function will not make sense for all MERA types. Implementation relies on
the function `expand_internalspace`, defined for each `Layer` type.
"""
function expand_internal_bonddim(m::GenericMERA, depth, newdims; check_invar=true)
    V = internalspace(m, depth)
    V = expand_vectorspace(V, newdims)
    layer = get_layer(m, depth)
    layer = expand_internalspace(layer, V)
    m = replace_layer(m, layer, depth; check_invar=check_invar)
    return m
end

"""
    remove_symmetry(m::GenericMERA)

Given a MERA which may possibly be built of symmetry preserving `TensorMap`s, return
another, equivalent MERA that has the symmetry structure stripped from it, and all tensors
are dense.
"""
remove_symmetry(m::GenericMERA) = GenericMERA(map(remove_symmetry, m.layers))

# # # Pseudo(de)serialization
# "Pseudo(de)serialization" refers to breaking the MERA down into types in Julia Base, and
# constructing it back. This can be used for storing MERAs on disk.
# TODO Once JLD or JLD2 works properly we should be able to get rid of this.
#
# Note that pseudoserialization discards the cache, forcing recomputation.

"""
pseudoserialize(x)

Return a tuple of objects that can be used to reconstruct `x`, and that are all of Julia
base types.

The name refers to how this isn't quite serialization, since it doesn't break objects down
to bit strings, but kinda serves the same purpose: pseudoserialised objects can easily be
written and read from e.g. disk, without having to worry about quirks of the type system
(like with JLD).

See also: [`depseudoserialize`](@ref)
"""
function pseudoserialize(m::T) where T <: GenericMERA
    return (repr(T), map(pseudoserialize, m.layers))
end

"""
depseudoserialize(::Type{T}, args) where T <: GenericMERA

Reconstruct an object given the output of `pseudoserialize`:
`x -> depseudoserialize(pseudoserialize(x)...)` should be an effective noop.

See also: [`pseudoserialize`](@ref)
"""
function depseudoserialize(::Type{T}, args) where T <: GenericMERA
    return GenericMERA([depseudoserialize(d...) for d in args])
end

# # # Invariants

"""
    space_invar(m::GenericMERA)

Check that the indices of the various tensors in `m` are compatible with each other. If not,
throw an `ArgumentError`. If yes, return `true`.

This relies on two checks, `space_invar_intralayer` for checking indices within a layer and
`space_invar_interlayer` for checking indices between layers. These should be implemented
for each subtype of `Layer`.
"""
function space_invar(m::GenericMERA)
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
            msg = "space_invar_intralayer has no method for type $(baselayertype(m)). Please consider writing one, to enable checking for space mismatches when assigning tensors."
            @warn(msg)
        end

        if applicable(space_invar_interlayer, layer, next_layer)
            if !space_invar_interlayer(layer, next_layer)
                errmsg = "Mismatching bonds in MERA between layers $(i-1) and $i."
                throw(ArgumentError(errmsg))
            end
        else
            msg = "space_invar_interlayer has no method for type $(baselayertype(m)). Please consider writing one, to enable checking for space mismatches when assigning tensors."
            @warn(msg)
        end
        layer = next_layer
    end
    return true
end

# # # Scaling superoperators

"""
    ascend(op, m::GenericMERA, endscale=num_translayers(m)+1, startscale=1)

Ascend the operator `op`, that lives on the layer `startscale`, through the MERA `m` to the
layer `endscale`. Living "on" a layer means living on the indices right below it, so
`startscale=1` refers to the physical indices, and `endscale=num_translayers(m)+1` to the
indices just below the first scale invariant layer.
"""
function ascend(op, m::GenericMERA, endscale=num_translayers(m)+1, startscale=1)
    if endscale < startscale
        throw(ArgumentError("endscale < startscale"))
    elseif endscale > startscale
        op_pre = ascend(op, m, endscale-1, startscale)
        layer = get_layer(m, endscale-1)
        op_asc = ascend(op_pre, layer)
    else
        op_asc = convert(operatortype(m), op)
    end
    return op_asc
end

"""
    descend(op, m::GenericMERA, endscale=1, startscale=num_translayers(m)+1)

Descend the operator `op`, that lives on the layer `startscale`, through the MERA `m` to the
layer `endscale`. Living "on" a layer means living on the indices right below it, so
`endscale=1` refers to the physical indices, and `startscale=num_translayers(m)+1` to the
indices just below the first scale invariant layer.
"""
function descend(op, m::GenericMERA, endscale=1, startscale=num_translayers(m)+1)
    if endscale > startscale
        throw(ArgumentError("endscale > startscale"))
    elseif endscale < startscale
        op_pre = descend(op, m, endscale+1, startscale)
        layer = get_layer(m, endscale)
        op_desc = descend(op_pre, layer)
    else
        op_desc = convert(operatortype(m), op)
    end
    return op_desc
end

"""
    fixedpoint_densitymatrix(m::GenericMERA, pars::NamedTuple=(;))

Find the fixed point density matrix of the scale invariant part of the MERA.

To find the fixed point, we use an iterative Krylov solver. The options for the solver
should be in `pars.scaleinvariant_krylovoptions`, they will be passed to
`KrylovKit.eigsolve`.
"""
function fixedpoint_densitymatrix(m::GenericMERA, pars=(;))
    f(x) = descend(x, get_layer(m, num_translayers(m)+1))
    # If we have stored the previous fixed point density matrix, and it has the right
    # dimensions, use that as the initial guess. Else, use a thermal density matrix.
    x0 = thermal_densitymatrix(m, Inf)
    old_rho = m.cache.previous_fixedpoint_densitymatrix
    if old_rho !== nothing && space(x0) == space(old_rho)
        x0 = old_rho
    end
    eigsolve_pars = get(pars, :scaleinvariant_krylovoptions, (;))
    _, vecs, vals, info = schursolve(f, x0, 1, :LM, Arnoldi(; eigsolve_pars...))
    rho = vecs[1]
    # We know the result should always be Hermitian, and scaled to have trace 1.
    rho = (rho + rho')  # probably not even necessary
    rho /= tr(rho)
    m.cache.previous_fixedpoint_densitymatrix = rho
    if :verbosity in keys(pars) && pars[:verbosity] > 3
        msg = "Used $(info.numops) superoperator invocations to find the fixed point density matrix."
        @info(msg)
    end
    return rho
end

"""
    thermal_densitymatrix(m::GenericMERA, depth)

Return the thermal density matrix for the indices right below the layer at `depth`. Used as
an initial guess for the fixed-point density matrix.

See also: [`fixedpoint_densitymatrix`](@ref)
"""
function thermal_densitymatrix(m::GenericMERA, depth)
    width = causal_cone_width(typeof(m))
    V = ⊗(ntuple(n->inputspace(m, depth), Val(width))...)
    return convert(operatortype(m), id(storagetype(operatortype(m)), V))
end

"""
    densitymatrix(m::GenericMERA, depth, pars=(;))

Return the density matrix right below the layer at `depth`.

`pars` maybe a `NamedTuple` of options, passed on to the function
`fixedpoint_densitymatrix`.

This function utilises the cache, to avoid recomputation.

See also: [`fixedpoint_densitymatrix`](@ref), [`densitymatrices`](@ref)
"""
function densitymatrix(m::GenericMERA, depth, pars=(;))
    if !has_densitymatrix_stored(m.cache, depth)
        # If we don't find rho in storage, generate it.
        if depth > num_translayers(m)
            rho = fixedpoint_densitymatrix(m, pars)
        else
            rho_above = densitymatrix(m, depth+1, pars)
            rho = descend(rho_above, m, depth, depth+1)
        end
        # Store this density matrix for future use.
        set_stored_densitymatrix!(m.cache, rho, depth)
    end
    return get_stored_densitymatrix(m.cache, depth)
end

"""
    densitymatrices(m::GenericMERA, pars=(;))

Return all the distinct density matrices of the MERA, starting with the one at the physical level, and ending with the scale invariant one.

`pars` maybe a `NamedTuple` of options, passed on to the function
`fixedpoint_densitymatrix`.

See also: [`fixedpoint_densitymatrix`](@ref), [`densitymatrix`](@ref)
"""
function densitymatrices(m::GenericMERA, pars=(;))
    rhos = [densitymatrix(m, depth, pars) for depth in 1:num_translayers(m)+1]
    return rhos
end

"""
    ascended_operator(m::GenericMERA, op, depth)

Return the operator `op` ascended from the physical level to `depth`.

This function utilises the cache, to avoid recomputation.

See also: [`scale_invariant_operator_sum`](@ref)
"""
function ascended_operator(m::GenericMERA, op, depth)
    # Note that if depth=1, has_operator_stored always returns true, as it initializes
    # storage for this operator.
    if !has_operator_stored(m.cache, op, depth)
        op_below = ascended_operator(m, op, depth-1)
        opasc = ascend(op_below, m, depth, depth-1)
        # Store this density matrix for future use.
        set_stored_operator!(m.cache, opasc, op, depth)
    end
    return get_stored_operator(m.cache, op, depth)
end

"""
    scale_invariant_operator_sum(m::GenericMERA, op, pars)

Return the sum of the ascended versions of `op` in the scale invariant part of the MERA.

To be more precise, this sum is of course infinite, and what we return is the component of
it orthogonal to the dominant eigenoperator of the ascending superoperator (typically the
identity). This component converges to a finite result like a geometric series, since all
non-dominant eigenvalues of the ascending superoperator are smaller than 1.

To approximate the converging series, we use an iterative Krylov solver. The options for the
solver should be in `pars.scaleinvariant_krylovoptions`, they will be passed
to `KrylovKit.linsolve`.
"""
function scale_invariant_operator_sum(m::GenericMERA{N, LT, OT}, op, pars::NamedTuple=(;)
                                     ) where {N, LT, OT}
    nt = num_translayers(m)
    # fp is the dominant eigenvector of the ascending superoperator. We are not interested
    # in contributions to the sum along fp, since they will just be fp * infty, and fp is
    # merely the representation of the identity operator.
    fp = ascending_fixedpoint(get_layer(m, nt+1))
    function f(x::OT)
        xasc::OT = ascend(x, m, nt+2, nt+1)
        xnorm::OT = xasc - fp * dot(fp, xasc)
        return xnorm
    end
    op_top = ascended_operator(m, op, nt+1)
    x0::OT = op_top
    old_opsum = m.cache.previous_operatorsum
    if old_opsum !== nothing && space(x0) == space(old_opsum)
        x0 = old_opsum
    end
    linsolve_pars = get(pars, :scaleinvariant_krylovoptions, (;))
    one_ = one(eltype(m))
    opsum::OT, info = linsolve(f, op_top, x0, one_, -one_; linsolve_pars...)
    # We know the result should always be Hermitian.
    opsum = (opsum + opsum') / 2.0
    m.cache.previous_operatorsum = opsum
    # We are not interested in the component along fp.
    opsum = opsum - fp * dot(fp, opsum)
    if :verbosity in keys(pars) && pars[:verbosity] > 3
        msg = "Used $(info.numops) superoperator invocations to find the scale invariant operator sum."
        @info(msg)
    end
    return opsum
end

"""
    environment(m::GenericMERA, op, depth, pars; vary_disentanglers=true)

Return a `Layer` consisting of the environments of the various tensors of `m` at `depth`, with respect to
the expectation value of `op`. Note the return value isn't really a proper MERA layer, i.e.
the tensors are not isometric, it just has the same structure, and hence the same data
structure is used.

`pars` are parameters that are passed to `scale_invariant_operator_sum` and
`fixedpoint_densitymatrix`. `vary_disentanglers` gives the option of computing the
environments only for the isometries, setting the environments of the disentanglers to zero.

This function utilises the cache to avoid recomputation.

See also: [`fixedpoint_densitymatrix`](@ref), [`scale_invariant_operator_sum`](@ref)
"""
function environment(m::GenericMERA{N, LT, OT}, op, depth, pars; vary_disentanglers=true
                    ) where {N, LT, OT}
    if !has_environment_stored(m.cache, op, depth)
        local op_below::OT
        if depth <= num_translayers(m)
            op_below = ascended_operator(m, op, depth)
        else
            op_below = scale_invariant_operator_sum(m, op, pars)
        end
        op_below = normalise_hamiltonian(op_below)
        rho_above = densitymatrix(m, depth+1, pars)
        layer = get_layer(m, depth)
        env = environment(layer, op_below, rho_above; vary_disentanglers=vary_disentanglers)
        set_stored_environment!(m.cache, env, op, depth)
    end
    return get_stored_environment(m.cache, op, depth)
end

# # # Extracting CFT data

"""
    scalingdimensions(m::GenericMERA, howmany=20)

Diagonalize the scale invariant ascending superoperator to compute the scaling dimensions of
the underlying CFT.

The return value is a dictionary, the keys of which are symmetry sectors for a possible
internal symmetry of the MERA (Trivial() if there is no internal symmetry), and values are
scaling dimensions in this symmetry sector.

`howmany` controls how many of lowest scaling dimensions are computed.
"""
function scalingdimensions(m::GenericMERA, howmany=20)
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
    toplayer = get_layer(m, Inf)
    f(x) = ascend(x, toplayer)
    # Find out which symmetry sectors we should do the diagonalization in.
    interlayer_space = ⊗(Iterators.repeated(V, width)...)
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
    scalingoperator_initialguess(m::GenericMERA, interlayer_space, irrep)

Return an initial guess to be used in the iterative eigensolver that solves for scaling
operators.
"""
function scalingoperator_initialguess(m::GenericMERA{N, LT, OT}, interlayer_space, irrep
                                     ) where {N, LT, OT}
    typ = eltype(m)
    inspace = interlayer_space
    outspace = interlayer_space
    # If this is a non-trivial irrep sector, expand the input space with a dummy leg.
    irrep !== Trivial() && (inspace = inspace ⊗ spacetype(inspace)(irrep => 1))
    # The initial guess for the eigenvalue search. Also defines the type for
    # eigenvectors.
    x0::OT = convert(OT, TensorMap(randn, typ, outspace ← inspace))
    return x0
end

# # # Evaluation

"""
    expect(op, m::GenericMERA, pars=(;), opscale=1, evalscale=1)

Return the expecation value of operator `op` for the MERA `m`.

The layer on which `op` lives is set by `opscale`, which by default is the physical one.
`evalscale` can be used to set whether the operator is ascended through the network or the
density matrix is descended. `pars` can hold additional options that are further down the
line passed on to `fixedpoint_densitymatrix`.

See also: [`fixedpoint_densitymatrix`](@ref)
"""
function expect(op, m::GenericMERA, pars=(;), opscale=1, evalscale=1)
    rho = densitymatrix(m, evalscale, pars)
    op = ascended_operator(m, op, evalscale)
    value = dot(rho, op)
    if abs(imag(value)/norm(op)) > 1e-13
        @warn("Non-real expectation value: $value")
    end
    value = real(value)
    return value
end


# # # Optimization

const default_pars = (method = :lbfgs,
                      retraction = :exp,
                      transport = :exp,
                      metric = :euclidean,
                      precondition = true,
                      gradient_delta = 1e-14,
                      isometries_only_iters = 0,
                      maxiter = 2000,
                      ev_layer_iters = 1,
                      ls_epsilon = 1e-6,
                      lbfgs_m = 8,
                      cg_flavor = :HagerZhang,
                      verbosity = 2,
                      scaleinvariant_krylovoptions = (
                                                      tol = 1e-13,
                                                      krylovdim = 4,
                                                      verbosity = 0,
                                                      maxiter = 20,
                                                     ),
                     )

"""
    minimize_expectation(m::GenericMERA, h, pars=(;);
                         finalize! = OptimKit._finalize!, vary_disentanglers=true,
                         kwargs...)

Return a MERA optimized to minimize the expectation value of operator `h`, starting with `m`
as the initial guess.

`pars` is a `NamedTuple` of parameters for the optimisation. They are,
* `method`: A `Symbol` that chooses which optimisation method to use. Options are `:lbfgs`
  for L-BFGS (default), `:ev` for Evenbly-Vidal, `:cg` for conjugate gradient, and `:gd` for
  gradient descent. `:lbfgs`, `:cg`, and `:gd` are together known as the gradient methods.
* `maxiter`: Maximum number of iterations to use. 2000 by default.
* `gradient_delta`: Convergence threshold, as measured by the norm of the gradient. `1e-14`
  by default.
* `precondition`: Whether to apply preconditioning with the physical Hilbert space inner
  product. `true` by default. See https://arxiv.org/abs/2007.03638 for more details.
* `verbosity`: How much output to log. 2 by default.
* `isometries_only_iters`: An integer for how many iterations should at first be done
  optimising only the isometries, and leaving the disentangler be. 0 by default.
* `scaleinvariant_krylovoptions`: A `NamedTuple` of keyword arguments passed to
  `KrylovKit.linsolve` and `KrylovKit.eigsolve`, when solving for the fixed-point density
  matrix and the scale invariant operator sum. The default is
  `(tol = 1e-13, krylovdim = 4, verbosity = 0, maxiter = 20)`.
  * `retraction`: Which retraction method to use. Options are `:exp` for geodesics (default), and `:cayley` for Cayley transforms. Only affects gradient methods.
* transport: Which vector transport method to use. Currently each retraction` method only
  comes with a single compatible transport, so one should always use `transport` to be the
  same as `retraction`. This may change. Only affects gradient methods.
* `metric`: Which metric to use for Stiefel manifold. Options are `:euclidean`
  (default) and `:canonical`. Only affects gradient methods.
* `ls_epsilon`: The ϵ parameter for the Hager-Zhang line search. `1e-6` be default. Only affects gradient methods.
* `lbfgs_m`: The rank of the approximation of the inverse Hessian in L-BFGS. 8 by default. Only affects the `:lbfgs` method.
* `cg_flavor`: The "flavor" of conjguate gradient to use. `:HagerZhang` by default. Only affects the `:cg:` method.
* `ev_layer_iters`: How many times a single layer is optimised before moving to the next layer in the Evenbly-Vidal algorithm. `1` by default. Only affects the `:ev` method.
If any of these are specified in `pars`, the specified values override the defaults.

`finalize!` is a function that will be called at every iteration. It can be used to for
instance log the development of some quantity during the optimisation, or modify the MERA in
some custom (although undefined behavior may follow depending on how the state is changed).
Its signature is `finalize!(m, f, g, counter)` where `m` is the current MERA, `f` is its
expectation value for `h`, `g` is the gradient MERA at `m`, and `counter` is the current
iteration number. It should return the possibly modified `m`, `f`, and `g`. If `method =
:ev`, then it should also be able to handle the case `g = nothing`, since the Evenbly-Vidal
algorithm does not use gradients.

`vary_disentanglers` gives the option of running the optimisation but only for the
isometries, leaving the disentanglers as they are.
"""
function minimize_expectation(m::GenericMERA, h, pars=(;); finalize! = OptimKit._finalize!,
                              vary_disentanglers=true)
    pars = merge(default_pars, pars)
    # If pars[:isometries_only_iters] is set, but the optimization on the whole is supposed
    # to vary_disentanglers too, then first run a pre-optimization without touching the
    # disentanglers, with pars[:isometries_only_iters] as the maximum iteration count,
    # before moving on to the main optimization with all tensors varying.
    if vary_disentanglers && pars[:isometries_only_iters] > 0
        temp_pars = merge(pars, (maxiter = pars[:isometries_only_iters],))
        m = minimize_expectation(m, h, temp_pars; finalize! = finalize!,
                                 vary_disentanglers=false)
    end

    method = pars[:method]
    if method in (:cg, :conjugategradient, :gd, :gradientdescent, :lbfgs)
        return minimize_expectation_grad(m, h, pars; vary_disentanglers=vary_disentanglers,
                                         finalize! = finalize!)
    elseif method == :ev || method == :evenblyvidal
        return minimize_expectation_ev(m, h, pars; vary_disentanglers=vary_disentanglers,
                                       finalize! = finalize!)
    else
        msg = "Unknown optimization method $(method)."
        throw(ArgumentError(msg))
    end
end

"""
    minimize_expectation_ev(m::GenericMERA, h, pars;
                            finalize! = OptimKit._finalize!, vary_disentanglers=true)

Return a MERA optimized with the Evenbly-Vidal method to minimize the expectation value of
operator `h`, starting with `m` as the initial guess. See [`minimize_expectation`](@ref) for
details.
"""
function minimize_expectation_ev(m::GenericMERA, h, pars; finalize! = OptimKit._finalize!,
                                 vary_disentanglers=true)
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

        for l in 1:nt+1
            local env, layer
            for i in 1:pars[:ev_layer_iters]
                env = environment(m, h, l, pars; vary_disentanglers=vary_disentanglers)
                layer = get_layer(m, l)
                new_layer = minimize_expectation_ev(layer, env;
                                                    vary_disentanglers=vary_disentanglers)
                m = replace_layer(m, new_layer, l)
            end
            # We use the latest env and the corresponding layer to compute the norm of the
            # gradient. This isn't quite the gradient at the end point, which is what we
            # would want, but close enough.
            gradnorm_sq += gradient_normsq(layer, env;
                                           metric=pars[:metric])
        end

        gradnorm = sqrt(gradnorm_sq)
        rhos = densitymatrices(m, pars)
        expectation = expect(h, m, pars)
        rho_diffs = [norm(r - or) for (r, or) in zip(rhos, old_rhos)]
        rhos_maxchange = maximum(rho_diffs)
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
    normalise_hamiltonian(h)

Change the additive normalisation of a local Hamiltonian term `h` to make it suitable for
computing an Evenbly-Vidal update environment.

More specifically, return `h - ub*eye`, where `eye` is the identity operator, and `ub` is an
upper bound for the largest eigenvalue of `h`
"""
function normalise_hamiltonian(h)
    lb, ub = gershgorin_bounds(h)
    eye = id(domain(h))
    return h - ub*eye
end

# # # Gradient optimization

"""
    tensorwise_scale(m::GenericMERA, alpha::Number)

Scale all the tensors of `m` by `alpha`.
"""
function tensorwise_scale(m::GenericMERA, alpha::Number)
    return GenericMERA(tensorwise_scale.(m.layers, alpha))
end

"""
    tensorwise_sum(m1::T, m2::T) where T <: GenericMERA

Return a MERA for which each tensor is the sum of the corresponding tensors of `m1` and
`m2`.
"""
function tensorwise_sum(m1::T, m2::T) where T <: GenericMERA
    n = max(num_translayers(m1), num_translayers(m2)) + 1
    layers = ntuple(Val(n)) do i
        tensorwise_sum(get_layer(m1, i), get_layer(m2, i))
    end
    return GenericMERA(layers)
end

"""
    inner(m::GenericMERA, m1::GenericMERA, m2::GenericMERA; metric=:euclidean)

Given two tangent MERAs `m1` and `m2`, both at base point `m`, compute their inner product.
This means the sum of the inner products of the individual tensors.

See `TensorKitManifolds.inner` for more details.
"""
function TensorKitManifolds.inner(m::GenericMERA, m1::GenericMERA, m2::GenericMERA;
                                  metric=:euclidean)
    n = max(num_translayers(m1), num_translayers(m2)) + 1
    res = sum([inner(get_layer(m, i), get_layer(m1, i), get_layer(m2, i); metric=metric)
               for i in 1:n])
    return res
end

"""
    gradient(h, m::GenericMERA, pars::NamedTuple; vary_disentanglers=true)

Compute the gradient of the expectation value of `h` at the point `m`.

`pars` should have `pars.metric` that specifies whether to use the `:euclidean` or
`:canonical` metric for the Stiefel manifold, and `pars.scaleinvariant_krylovoptions` that
is passed on to [`environment`](@ref).

`vary_disentanglers` allows computing the gradients only for the isometries, and setting the
gradients for the disentanglers to zero.

The return value is a "tangent MERA": An object of a similar type as `m`, but instead of
regular layers with tensors that have isometricity constraints, instead each layer holds the
corresponding gradients for each tensor.
"""
function gradient(h, m::GenericMERA{N}, pars::NamedTuple; vary_disentanglers=true) where {N}
    nt = num_translayers(m)
    layers = ntuple(Val(nt+1)) do l
        layer = get_layer(m, l)
        env = environment(m, h, l, pars; vary_disentanglers=vary_disentanglers)
        gradient(layer, env; metric=pars[:metric])
    end
    g = GenericMERA(layers)
    return g
end

"""
    precondition_tangent(m::GenericMERA, tan::GenericMERA, pars::NamedTuple)

Precondition the gradient `tan`, living at the base point `m`, using the physical Hilbert
space inner product.

For details on how the preconditioning is done, see https://arxiv.org/abs/2007.03638.

`pars` is passed on to [`densitymatrix`](@ref).
"""
function precondition_tangent(m::GenericMERA, tan::GenericMERA, pars::NamedTuple)
    nt = num_translayers(m)
    tanlayers_prec = (begin
                          layer = get_layer(m, l)
                          tanlayer = get_layer(tan, l)
                          rho = densitymatrix(m, l+1, pars)
                          precondition_tangent(layer, tanlayer, rho)
                      end
                      for l in 1:nt+1)
    tan_prec = typeof(tan)(tanlayers_prec)
    return tan_prec
end

"""
    retract(m::GenericMERA, mtan::GenericMERA, alpha::Real; kwargs...)

Given a "tangent MERA" `mtan`, at base point `m`, retract in the direction of `mtan` by
distance `alpha`. This is done tensor-by-tensor, i.e. each tensor is retracted along its
respective Stiefel/Grassmann tangent.

The additional keyword argument are passed on to the respective `TensorKitManifolds`
function.

See `TensorKitManifolds.retract` for more details.

See also: [`transport!`](@ref)
"""
function TensorKitManifolds.retract(m::T1, mtan::T2, alpha::Real; kwargs...
                                   ) where {T1 <: GenericMERA, T2 <: GenericMERA}
    layers, layers_tan = zip((retract(l, ltan, alpha; kwargs...)
                              for (l, ltan) in zip(m.layers, mtan.layers))...)
    # TODO The following two lines just work around a compiler bug in Julia < 1.6.
    layers = tuple(layers...)
    layers_tan = tuple(layers_tan...)
    return T1(layers), T2(layers_tan)
end

"""
    transport!(mvec::GenericMERA, m::GenericMERA, mtan::GenericMERA, alpha::Real,
               mend::GenericMERA; kwargs...)

Given a "tangent MERAs" `mtan` and `mvec`, at base point `m`, transport `mvec` in the
direction of `mtan` by distance `alpha`. This is done tensor-by-tensor, i.e. each tensor is
transported along its respective Stiefel/Grassmann tangent.

`mend` is the endpoint on the manifold, i.e. the result of retracting by `alpha` in the
direction of `mtan`. The additional keyword argument are passed on to the respective
`TensorKitManifolds` function.

See `TensorKitManifolds.transport!` for more details.

See also: [`retract`](@ref)
"""
function TensorKitManifolds.transport!(mvec::T2, m::T1, mtan::T2, alpha::Real, mend::T1;
                                       kwargs...) where {T1 <: GenericMERA,
                                                         T2 <: GenericMERA}
    layers = (transport!(lvec, l, ltan, alpha, lend; kwargs...)
              for (lvec, l, ltan, lend)
              in zip(mvec.layers, m.layers, mtan.layers, mend.layers))
    return T2(layers)
end

"""
    minimize_expectation_grad(m, h, pars;
                              finalize! = OptimKit._finalize!, vary_disentanglers=true)

Return a MERA optimized with one of the gradient methods to minimize the expectation value
of operator `h`, starting with `m` as the initial guess. See [`minimize_expectation`](@ref)
for details.
"""
function minimize_expectation_grad(m, h, pars; finalize! = OptimKit._finalize!,
                                   vary_disentanglers=true)
    function fg(x)
        f = expect(h, x, pars)
        g = gradient(h, x, pars; vary_disentanglers=vary_disentanglers)
        return f, g
    end

    rtrct(args...) = retract(args...; alg=pars[:retraction])
    trnsprt!(args...) = transport!(args...; alg=pars[:transport])
    innr(args...) = inner(args...; metric=pars[:metric])
    scale(vec, beta) = tensorwise_scale(vec, beta)
    add(vec1, vec2, beta) = tensorwise_sum(vec1, scale(vec2, beta))
    linesearch = HagerZhangLineSearch(; ϵ=pars[:ls_epsilon])
    if pars[:precondition]
        precondition(x, g) = precondition_tangent(x, g, pars)
    else
        # The default that does nothing.
        precondition = OptimKit._precondition
    end

    algkwargs = (maxiter = pars[:maxiter],
                 linesearch = linesearch,
                 verbosity = pars[:verbosity],
                 gradtol = pars[:gradient_delta])
    local alg
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
