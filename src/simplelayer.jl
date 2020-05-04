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
function remove_symmetry(layer::T) where T <: SimpleLayer
    return T(map(remove_symmetry, layer)...)
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

function tensorwise_sum(l1::T, l2::T) where T <: SimpleLayer
    return T(map(sum, zip(l1, l2))...)
end

function tensorwise_scale(layer::T, alpha::Number) where T <: SimpleLayer
    return T((t*alpha for t in layer)...)
end

# # # Manifold functions

function TensorKitManifolds.inner(l::T, l1::T, l2::T; metric=:euclidean
                                 ) where T <: SimpleLayer
    return sum((inner(t...; metric=metric) for t in zip(l, l1, l2)))
end

function TensorKitManifolds.retract(l::T, ltan::T, alpha::Real; alg=:exp
                                   ) where T <: SimpleLayer
    ts_and_ttans = [retract(t..., alpha; alg=alg) for t in zip(l, ltan)]
    ts, ttans = zip(ts_and_ttans...)
    return T(ts...), T(ttans...)
end

function TensorKitManifolds.transport!(lvec::T, l::T, ltan::T, alpha::Real, lend::T;
                                       alg=:exp) where {T <: SimpleLayer}
    return T((transport!(t[1], t[2], t[3], alpha, t[4]; alg=alg)
              for t in zip(lvec, l, ltan, lend))...)
end

function gradient_normsq(layer::T, env::T; isometrymanifold=:grassmann, metric=:euclidean
                        ) where {T <: SimpleLayer}
    grad = gradient(layer, env; isometrymanifold=isometrymanifold, metric=metric)
    return sum(inner(x, z, z; metric=metric) for (x, z) in zip(layer, grad))
end
