# Utilities for creating and modifying vector spaces and TensorMaps.
# To be `included` in MERA.jl.

"""
Given two vector spaces, create an isometric/unitary TensorMap from one to the other.
"""
function randomisometry(Vout, Vin)
    temp = TensorMap(randn, ComplexF64, Vout ← Vin)
    U, S, Vt = svd(temp)
    u = U * Vt
    return u
end

"""
Given two vector spaces, create an isometric/unitary TensorMap from one to the other that
is the identity, with truncations as needed.
"""
function identitytensor(Vout, Vin)
    u = TensorMap(I, ComplexF64, Vout ← Vin)
    return u
end

"""
Given a vector space and a dictionary of dimensions for the various irrep sectors, return
another vector space of the same kind but with these new dimension. If some irrep sectors
are not in the dictionary, the dimensions of the original space are used.
"""
function expand_vectorspace(V::CartesianSpace, newdim)
    d = length(newdim) > 0 ? first(values(newdim)) : dim(V)
    return typeof(V)(d)
end

function expand_vectorspace(V::ComplexSpace, newdim)
    d = length(newdim) > 0 ? first(values(newdim)) : dim(V)
    return typeof(V)(d, V.dual)
end

function expand_vectorspace(V::GeneralSpace, newdim)
    d = length(newdim) > 0 ? first(values(newdim)) : dim(V)
    return typeof(V)(d, V.dual, V.conj)
end

function expand_vectorspace(V::RepresentationSpace, newdims)
    sectordict = merge(Dict(s => dim(V, s) for s in sectors(V)), newdims)
    return typeof(V)(sectordict; dual=V.dual)
end

# TODO How does this behave if V is smaller than the current space?
"""
Pad the TensorMap `T` with zeros, so that its index number `ind` now has the space `V`.
"""
function pad_with_zeros_to(T::TensorMap, ind, V)
    expander = TensorMap(I, eltype(T), space(T, ind)' ← V');
    sizedomain = length(domain(T))
    sizecodomain = length(codomain(T))
    numinds = sizedomain + sizecodomain
    indsfinal = collect(-1:-1:-numinds);
    indsT = copy(indsfinal)
    indsT[ind] = ind;
    indsexpander = [ind, -ind]
    eval(:(@tensor T_new_tensor[$(indsfinal...)] := $T[$(indsT...)] * $expander[$(indsexpander...)]))
    T_new = permuteind(T_new_tensor, tuple(1:sizecodomain...),
                       tuple(sizecodomain+1:numinds...))
    return T_new
end
