# The SimpleLayer abstract type and methods for it.
# To be `included` in MERA.jl.

"""
    SimpleLayer <: Layer

A `SimpleLayer` is a MERA layer that consists of a set of isometries and/or unitaries, and
nothing else. This allows writing convenient generic versions of many methods, reducing code
duplication for the concrete `Layer` types. Every subtype of `SimpleLayer` should implement
the iteration and indexing interfaces to return the various tensors of the layer in the same
order in which the constructor takes them in.
"""
abstract type SimpleLayer <: Layer end

Base.convert(::Type{T}, t::Tuple) where T <: SimpleLayer= T(t...)
Base.copy(layer::SimpleLayer) = typeof(layer)(map(deepcopy, _tuple(layer))...)

@inline function Base.iterate(layer::SimpleLayer, ::Val{i} = Val(1)) where {i}
    t = _tuple(layer)
    if i > length(t)
        return nothing
    else
        return t[i], Val(i+1)
    end
end
Base.indexed_iterate(layer::SimpleLayer, i::Int, args...) =
    Base.indexed_iterate(_tuple(layer), i, args...)
Base.length(layer::SimpleLayer) = _tuple(layer)

function remove_symmetry(layer::SimpleLayer)
    layertype(layer)(map(remove_symmetry, _tuple(layer))...)
end

function TensorKitManifolds.projectisometric(layer::SimpleLayer)
    typeof(layer)(map(projectisometric, _tuple(layer))...)
end

function TensorKitManifolds.projectisometric!(layer::SimpleLayer)
    foreach(projectisometric!, _tuple(layer))
    return layer
end

function pseudoserialize(layer::SimpleLayer)
    return repr(typeof(layer)), map(pseudoserialize, _tuple(layer))
end

function depseudoserialize(::Type{T}, data) where T <: SimpleLayer
    return T(map(d->depseudoserialize(d...), data)...)
end

# TODO Definitions like the following are a little dangerous, since they allow for instance
# summing a BinaryLayer with a TernaryLayer. Unfortunately I haven't been able to come up
# with a better implementation, because l1 and l2 can not be restricted to be of the exact
# same, fully parametrized type.

function tensorwise_scale(layer::SimpleLayer, alpha::Number)
    return layertype(layer)((_tuple(layer) .* alpha)...)
end

function tensorwise_sum(l1::SimpleLayer, l2::SimpleLayer)
    @assert layertype(l1) == layertype(l2)
    return layertype(l1)((_tuple(l1) .+ _tuple(l2))...)
end

# # # Manifold functions

function TensorKitManifolds.inner(l::SimpleLayer, l1::SimpleLayer, l2::SimpleLayer;
                                  metric=:euclidean)
    get_metric(t) = isa(t, Stiefel.StiefelTangent) ? metric : :euclidean
    return sum(inner(t, t1, t2; metric=get_metric(t1)) for (t, t1, t2) in zip(l, l1, l2))
end

function TensorKitManifolds.retract(l::SimpleLayer, ltan::SimpleLayer, alpha::Real;
                                    alg=:exp)
    ts_and_ttans = (retract(t..., alpha; alg=alg) for t in zip(l, ltan))
    ts, ttans = zip(ts_and_ttans...)
    # TODO The following two lines just work around a compiler bug in Julia < 1.6.
    ts = tuple(ts...)
    ttans = tuple(ttans...)
    return layertype(l)(ts...), layertype(l)(ttans...)
end

function TensorKitManifolds.transport!(lvec::SimpleLayer, l::SimpleLayer, ltan::SimpleLayer,
                                       alpha::Real, lend::SimpleLayer; alg=:exp)
    return layertype(l)((transport!(t[1], t[2], t[3], alpha, t[4]; alg=alg)
                         for t in zip(lvec, l, ltan, lend))...)
end

"""
    gradient_normsq(layer::Layer, env::Layer; metric=:euclidean)

Compute the norm of the gradient, given the enviroment layer `env` and the base point
`layer`.

See also: [`gradient`](@ref)
"""
function gradient_normsq(layer::SimpleLayer, env::SimpleLayer; metric=:euclidean)
    grad = gradient(layer, env; metric=metric)
    return sum(inner(x, z, z; metric=metric) for (x, z) in zip(layer, grad))
end
