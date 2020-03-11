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

# # # Stiefel manifold functions

function istangent(l::T, ltan::T) where T <: SimpleLayer
    return all(istangent_isometry(t...) for t in zip(l, ltan))
end

function stiefel_inner(l::T, l1::T, l2::T) where T <: SimpleLayer
    return sum((stiefel_inner(t...) for t in zip(l, l1, l2)))
end

function cayley_retract(l::T, ltan::T, alpha::Number) where T <: SimpleLayer
    ts_and_ttans = [cayley_retract(t..., alpha) for t in zip(l, ltan)]
    ts, ttans = zip(ts_and_ttans...)
    return T(ts...), T(ttans...)
end

function cayley_transport(l::T, ltan::T, lvec::T, alpha::Number) where T <: SimpleLayer
    return T((cayley_transport(t..., alpha) for t in zip(l, ltan, lvec))...)
end
