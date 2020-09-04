# A cache type for GenericMERA.
# To be included in MERA.jl.

# The storage formats for density matrices, operators, and environments are a little
# different:
#
# For density matrices we always store a `Vector` of the same length as there are
# layers. If a density matrix is not in store, `nothing` is kept as the corresponding
# element.
#
# `operators` is a dictionary with operators as keys, and each element is a
# `Vector` that lists the ascended versions of that operator, starting from the physical
# one, which is nothing but the original operator, possibly wrapped in some container type.
# No place-holder `nothing`s are stored, the `Vector` ends when storage ends.
#
# `environments` is a `Dict` just like `operators`, but its values are `Vectors` that hold
# either `Layer`s or `Nothing`s, where each `Layer` has instead of tensors of the layer,
# environments for the tensors of a layer.
#
# To understand the reason for these differences, note that density matrices are naturally
# generated from the top, and there will never be more than the number of layers of them.
# Similarly it doesn't make sense to have more than one environment for each layer.
# Operators are generated from the bottom, and they may go arbitrarily high in the MERA
# (into the scale invariant part).
#
# In addition we keep `previous_fixedpoint_densitymatrix` and `previous_operatorsum`, to use
# as initial guesses when solving for the fixed point density matrix or scale invariant
# operator sum.

"""
    MERACache{N, LT <: Layer, OT}

A cache for `GenericMERA`. The type parameters are the same as for `GenericMERA`.

See also: [`GenericMERA`](@ref)
"""
mutable struct MERACache{N, LT <: Layer, OT}
    # LT stands for Layer Type, OT for Operator Type.
    # TODO Should we use NTuple{N}s instead for densitymatrices and environments?
    densitymatrices::Vector{Union{Nothing, OT}}
    # probably a better choice to use IdDicts here
    operators::Dict{Any, Vector{OT}}
    environments::Dict{Any, Vector{Union{Nothing, LT}}}
    previous_fixedpoint_densitymatrix::Union{Nothing, OT}
    previous_operatorsum::Union{Nothing, OT}

    function MERACache{N, LT, OT}() where {N, LT, OT}
        @assert OT === operatortype(LT)
        densitymatrices = Vector{Union{Nothing, OT}}(fill(nothing, N))
        operators = Dict{Any, Vector{OT}}()
        environments = Dict{Any, Vector{Union{LT}}}()
        previous_fixedpoint_densitymatrix = nothing
        previous_operatorsum = nothing
        new{N, LT, OT}(densitymatrices, operators, environments,
                       previous_fixedpoint_densitymatrix,
                       previous_operatorsum)
    end
end

function MERACache{N, LT}() where {N, LT}
    OT = operatortype(LT)
    return MERACache{N, LT, OT}()
end

operatortype(::Type{MERACache{N,LT,OT}}) where {N, LT, OT} = OT
layertype(::Type{MERACache{N,LT,OT}}) where {N, LT, OT} = LT
baselayertype(::Type{MERACache{N,LT,OT}}) where {N, LT, OT} = baselayertype(LT)
causal_cone_width(C::Type{<:MERACache}) = causal_cone_width(layertype(C))

function Base.copy!(dst::MERACache, src::MERACache)
    dst.densitymatrices = copy(src.densitymatrices)
    dst.operators = Dict()
    dst.environments = Dict()
    for (k, v) in src.operators
        dst.operators[k] = copy(v)
    end
    for (k, v) in src.environments
        dst.environments[k] = copy(v)
    end
    dst.previous_operatorsum = src.previous_operatorsum
    dst.previous_fixedpoint_densitymatrix = src.previous_fixedpoint_densitymatrix
    return dst
end

Base.copy(c::MERACache) = copy!(typeof(c)(), c)

"""
    replace_layer(c::MERACache, depth)

Create a new `MERACache` that has all the stored pieces removed that are invalidated by
changing the layer at `depth`.
"""
replace_layer(c::MERACache, depth) = replace_layer!(copy(c), depth)

function replace_layer!(c::MERACache{N}, depth) where N
    c = copy(c)
    depth = Int(min(N, depth))
    c.densitymatrices[1:depth] .= nothing
    for v in values(c.operators)
        last_index = min(depth, length(v))
        deleteat!(v, last_index+1:length(v))
    end
    # Changing anything always invalidates all environments, since they depend on things
    # both above and below.
    reset_environment_storage!(c)
    return c
end

function release_transitionlayer(c::MERACache{N, LT}) where {N, LT}
    new_c = MERACache{N+1, LT}()
    copy!(new_c, c)
    density_matrix = new_c.densitymatrices[end]
    push!(new_c.densitymatrices, density_matrix)
    for (k, v) in new_c.environments
        push!(v, nothing)
    end
    return new_c
end

"""
    has_densitymatrix_stored(c::MERACache, depth)

Return whether the density matrix at given depth is already in the cache.

See also: [`set_stored_densitymatrix!`](@ref), [`get_stored_densitymatrix`](@ref)
"""
function has_densitymatrix_stored(c::MERACache{N}, depth) where N
    depth = Int(min(N, depth))
    rho = c.densitymatrices[depth]
    return rho !== nothing
end

"""
    get_stored_densitymatrix(c::MERACache, depth)

Return the density matrix at given depth, assuming it's in the cache.

See also: [`set_stored_densitymatrix!`](@ref), [`has_densitymatrix_stored`](@ref)
"""
function get_stored_densitymatrix(c::MERACache{N}, depth) where N
    depth = Int(min(N, depth))
    rho = c.densitymatrices[depth]
    @assert rho !== nothing
    return rho
end

"""
    set_stored_densitymatrix!(c::MERACache, density_matrix, depth)

Cache the density matrix at given depth.

See also: [`get_stored_densitymatrix`](@ref), [`has_densitymatrix_stored`](@ref)
"""
function set_stored_densitymatrix!(c::MERACache, density_matrix, depth)
    c.densitymatrices[depth] = density_matrix
    return c
end

"""
    operator_storage!(c::MERACache, op)

Return the cached ascended versions of `op`. Initialize the cache for `op` if necessary.
"""
function operator_storage!(c::MERACache{N, LT, OT}, op) where {N, LT, OT}
    if !haskey(c.operators, op)
        op_conv = convert(OT, expand_support(op, causal_cone_width(c)))
        c.operators[op] = [op_conv]
    end
    return c.operators[op]
end

"""
    has_operator_stored(c::MERACache, op, depth)

Return whether the operator `op` ascended to `depth` is already in the cache.

See also: [`set_stored_operator!`](@ref), [`get_stored_operator`](@ref)
"""
function has_operator_stored(c::MERACache, op, depth)
    storage = operator_storage!(c, op)
    return length(storage) >= depth
end

"""
    get_stored_operator(c::MERACache, op, depth)

Return the operator `op` ascended to a given depth, assuming it's in the cache.

See also: [`set_stored_operator!`](@ref), [`has_operator_stored`](@ref)
"""
function get_stored_operator(c::MERACache, op, depth)
    storage = operator_storage!(c, op)
    return storage[depth]
end

"""
    set_stored_operator!(c::MERACache, opasc, op, depth)

Cache `opasc`, the operator `op` ascended to a given depth.

See also: [`get_stored_operator`](@ref), [`has_operator_stored`](@ref)
"""
function set_stored_operator!(c::MERACache, opasc, op, depth)
    storage = operator_storage!(c, op)
    if length(storage) < depth-1
        msg = "Can't store an ascended operator if the lower versions of it aren't in storage already."
        throw(ArgumentError(msg))
    elseif length(storage) == depth-1
        push!(storage, opasc)
    else
        storage[depth] = opasc
    end
    return c
end

"""
    reset_operator_storage!(c::MERACache, op)

Reset cache for a given operator, removing its ascended memory and forcing recomputation.
"""
function reset_operator_storage!(c::MERACache, op)
    delete!(c.operators, op)
    return c
end

"""
    reset_operator_storage!(c::MERACache)

Reset storage for all operators.
"""
function reset_operator_storage!(c::MERACache)
    empty!(c.operators)
    return c
end

"""
    environment_storage!(c::MERACache, op)

Return the environments related to `op`. Initialize the storage `Vector` if necessary.
"""
function environment_storage!(c::MERACache{N}, op) where N
    if !haskey(c.environments, op)
        c.environments[op] = repeat(Union{Nothing, operatortype(c)}[nothing], N)
    end
    return c.environments[op]
end

"""
    has_environment_stored(c::MERACache, op, depth)

Return whether the environment related to `op` at `depth` is already in store.

See also: [`set_stored_environment!`](@ref), [`get_stored_environment`](@ref)
"""
function has_environment_stored(c::MERACache, op, depth)
    storage = environment_storage!(c, op)
    return storage[depth] !== nothing
end

"""
    get_stored_environment(c::MERACache, op, depth)

Return the environment related to `op` at `depth`. A type assertion error is thrown if the
requested environment is not in storage.

See also: [`set_stored_environment!`](@ref), [`has_environment_stored`](@ref)
"""
function get_stored_environment(c::MERACache, op, depth)
    storage = environment_storage!(c, op)
    env = storage[depth]
    @assert env !== nothing
    return env
end

"""
    set_stored_environment!(c::MERACache, env, op, depth)

Store `env`, the environments related to `op` at `depth`.

See also: [`get_stored_environment`](@ref), [`has_environment_stored`](@ref)
"""
function set_stored_environment!(c::MERACache, env, op, depth)
    storage = environment_storage!(c, op)
    storage[depth] = env
    return c
end

"""
    reset_environment_storage!(c::MERACache)

Reset storage for environments.
"""
function reset_environment_storage!(c::MERACache)
    empty!(c.environments)
    return c
end

function expand_bonddim!(c::MERACache{N, LT}, depth, V) where {N, LT}
    depth < N && return c
    width = causal_cone_width(LT)
    # Pad the stored scale invariant initial guesses.
    old_rho = m.previous_fixedpoint_densitymatrix[1]
    if old_rho !== nothing
        for i in 1:width
            old_rho = pad_with_zeros_to(old_rho, i => V, (i+width) => V')
        end
        m.previous_fixedpoint_densitymatrix[1] = old_rho
    end
    old_opsum = m.previous_operatorsum[1]
    if old_opsum !== nothing
        for i in 1:width
            old_opsum = pad_with_zeros_to(old_opsum, i => V, (i+width) => V')
        end
        m.previous_operatorsum[1] = old_opsum
    end
    return c
end
