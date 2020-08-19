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
Base.length(layer::SimpleLayer) = length(_tuple(layer))

function remove_symmetry(layer::SimpleLayer)
    return baselayertype(layer)(map(remove_symmetry, _tuple(layer))...)
end

function TensorKitManifolds.projectisometric(layer::SimpleLayer)
    return typeof(layer)(map(projectisometric, _tuple(layer))...)
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

function tensorwise_scale(layer::SimpleLayer, alpha::Number)
    return baselayertype(layer)((_tuple(layer) .* alpha)...)
end

function tensorwise_sum(l1::SimpleLayer, l2::SimpleLayer)
    @assert baselayertype(l1) == baselayertype(l2)
    return baselayertype(l1)((_tuple(l1) .+ _tuple(l2))...)
end

# # # Manifold functions

function TensorKitManifolds.inner(l::SimpleLayer, l1::SimpleLayer, l2::SimpleLayer;
                                  metric=:euclidean)
    custominner(t, t1, t2) =
        inner(t, t1, t2; metric = isa(t1, Stiefel.StiefelTangent) ? metric : :euclidean)
    return sum(custominner.(_tuple(l), _tuple(l1), _tuple(l2)))
end

function TensorKitManifolds.retract(l::SimpleLayer, ltan::SimpleLayer, alpha::Real;
                                    alg=:exp)

    ts_and_ttans = retract.(_tuple(l), _tuple(ltan), alpha; alg = alg)
    ts = first.(ts_and_ttans)
    ttans = last.(ts_and_ttans)
    return baselayertype(l)(ts...), baselayertype(l)(ttans...)
end

function TensorKitManifolds.transport!(lvec::SimpleLayer, l::SimpleLayer, ltan::SimpleLayer,
                                       alpha::Real, lend::SimpleLayer; alg=:exp)
    ttans = transport!.(_tuple(lvec), _tuple(l), _tuple(ltan), alpha, _tuple(lend);
                            alg = alg)
    return baselayertype(l)(ttans...)
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
