# The SimpleLayer abstract type and methods for it.
# To be `included` in MERA.jl.

"""A SimpleLayer is a MERA layer that consists of a set of isometries and/or unitaries, and
nothing else. This allows writing convenient generic versions of many methods, reducing code
duplication for the concrete Layer types. Every subtype of SimpleLayer should implement the
iteration and indexing interfaces to return the various tensors of the layer in the same
order in which the constructor takes them in.
"""
abstract type SimpleLayer <: Layer end

Base.convert(::Type{T}, tuple::Tuple) where T <: SimpleLayer= T(tuple...)
Base.eltype(layer::SimpleLayer) = reduce(promote_type, map(eltype, layer))
Base.copy(layer::T) where T <: SimpleLayer = T(map(deepcopy, layer)...)

"""Strip a layer of its internal symmetries."""
function remove_symmetry(layer::SimpleLayer)
    return layertype(layer)(map(remove_symmetry, layer)...)
end

function TensorKitManifolds.projectisometric(layer::T) where T <: SimpleLayer
    return T(map(projectisometric, layer)...)
end

function TensorKitManifolds.projectisometric!(layer::T) where T <: SimpleLayer
    return T(map(projectisometric!, layer)...)
end

function pseudoserialize(layer::T) where T <: SimpleLayer
    return repr(T), map(pseudoserialize, layer)
end

function depseudoserialize(::Type{T}, data) where T <: SimpleLayer
    return T([depseudoserialize(d...) for d in data]...)
end

# TODO Definitions like the following are a little dangerous, since they allow for instance
# summing a BinaryLayer with a TernaryLayer. Unfortunately I haven't been able to come up
# with a better implementation, because l1 and l2 can not be restricted to be of the exact
# same, fully parametrized type.

function tensorwise_scale(layer::SimpleLayer, alpha::Number)
    return layertype(layer)((t*alpha for t in layer)...)
end

function tensorwise_sum(l1::SimpleLayer, l2::SimpleLayer)
    return layertype(l1)((t1+t2 for (t1, t2) in zip(l1, l2))...)
end

# # # Manifold functions

function TensorKitManifolds.inner(l::SimpleLayer, l1::SimpleLayer, l2::SimpleLayer;
                                  metric=:euclidean)
    get_metric(t) = isa(t, Stiefel.StiefelTangent) ? metric : :euclidean
    return sum(inner(t, t1, t2; metric=get_metric(t1)) for (t, t1, t2) in zip(l, l1, l2))
end

function TensorKitManifolds.retract(l::SimpleLayer, ltan::SimpleLayer, alpha::Real;
                                    alg=:exp)
    ts_and_ttans = [retract(t..., alpha; alg=alg) for t in zip(l, ltan)]
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

function gradient_normsq(layer::SimpleLayer, env::SimpleLayer; isometrymanifold=:grassmann,
                         metric=:euclidean)
    grad = gradient(layer, env; isometrymanifold=isometrymanifold, metric=metric)
    return sum(inner(x, z, z; metric=metric) for (x, z) in zip(layer, grad))
end
