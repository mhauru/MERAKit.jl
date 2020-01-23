# Utilities for creating and modifying vector spaces and TensorMaps.
# To be `included` in MERA.jl.

# TODO I feel like there's a nicer way of writing this, than the really long where {...}.
"""
A TensorMap from N indices to N indices.
"""
SquareTensorMap{N} = TensorMap{S1, N, N, S2, T1, T2, T3} where {S1, S2, T1, T2, T3}

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
identitytensor(Vout, Vin) = TensorMap(I, ComplexF64, Vout ← Vin)

"""
Return the number of sites/indices `m` that an operator is supported on, assuming it is an
operator from `m` sites to `m` sites.
"""
support(op::SquareTensorMap{N}) where {N} = N

"""
Given an operator as TensorMap from a number of indices to the same number of indices,
expand its support to a larger number of sites `n` by tensoring with the identity. The
different ways of doing the expansion are averaged over.
"""
function expand_support(op::SquareTensorMap{N}, n::Integer) where {N}
    V = space(op, 1)
    eye = TensorMap(I, eltype(op), V ← V)
    m = N
    while m < n
        opeye = op ⊗ eye
        eyeop = eye ⊗ op
        op = (opeye + eyeop)/2
        m += 1
    end
    return op
end

"""Strip a real ElementarySpace of its symmetry structure."""
remove_symmetry(V::ElementarySpace{ℝ}) = CartesianSpace(dim(V))
"""Strip a complex ElementarySpace of its symmetry structure."""
remove_symmetry(V::ElementarySpace{ℂ}) = ComplexSpace(dim(V), isdual(V))

""" Strip a TensorMap of its internal symmetries."""
function remove_symmetry(t::TensorMap)
    domain_nosym = reduce(⊗, map(remove_symmetry, domain(t)))
    codomain_nosym = reduce(⊗, map(remove_symmetry, codomain(t)))
    t_nosym = TensorMap(zeros, eltype(t), codomain_nosym ← domain_nosym)
    t_nosym.data[:] = convert(Array, t)
    return t_nosym
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

"""
Return a tuple of objects that can be used to reconstruct a given TensorMap, and that are
all of Julia base types.
"""
function pseudoserialize(t::T) where T <: TensorMap
    # We make use of the nice fact that many TensorKit objects return on repr
    # strings that are valid syntax to reconstruct these objects.
    domstr = repr(t.dom)
    codomstr = repr(t.codom)
    eltyp = eltype(t)
    if isa(t.data, AbstractArray)
        data = t.data
    else
        data = Dict(repr(s) => deepcopy(d) for (s, d) in t.data)
    end
    return repr(T), (domstr, codomstr, eltyp, data)
end

"""
Reconstruct a TensorMap given the output of `pseudoserialize`.
"""
function depseudoserialize(::Type{T}, args) where T <: TensorMap
    # We make use of the nice fact that many TensorKit objects return on repr
    # strings that are valid syntax to reconstruct these objects.
    domstr, codomstr, eltyp, data = args
    dom = eval(Meta.parse(domstr))
    codom = eval(Meta.parse(codomstr))
    t = TensorMap(zeros, eltyp, codom ← dom)
    if isa(t.data, AbstractArray)
        t.data[:] = data
    else
        for (irrepstr, irrepdata) in data
            irrep = eval(Meta.parse(irrepstr))
            t.data[irrep][:] = irrepdata
        end
    end
    return t
end

"""
Transform a TensorMap `t` to change the vector spaces of its indices. `spacedict` should be
a dictionary of index labels to VectorSpaces, that tells which indices should have their
space changed. Instead of a dictionary, a varargs of Pairs `index => vectorspace` also
works.

For each index `i`, its current space `Vorig = space(t, i)` and new space `Vnew =
spacedict[i]` should be of the same type. If `Vnew` is strictly larger than `Vold` then `t`
is padded with zeros to fill in the new elements. Otherwise some elements of `t` will be
truncated away.
"""
function pad_with_zeros_to(t::TensorMap, spacedict::Dict)
    # Expanders are the matrices by which each index will be multiplied to change the space.
    expanders = [TensorMap(I, eltype(t), V ← space(t, ind)) for (ind, V) in spacedict]
    sizedomain = length(domain(t))
    sizecodomain = length(codomain(t))
    numinds = sizedomain + sizecodomain
    inds_t = [ind in keys(spacedict) ? ind : -ind for ind in 1:numinds]
    inds_expanders = [[-ind, ind] for ind in keys(spacedict)]
    tensors = [t, expanders...]
    inds = [inds_t, inds_expanders...]
    t_new_tensor = @ncon(tensors, inds)
    t_new = permuteind(t_new_tensor, tuple(1:sizecodomain...),
                       tuple(sizecodomain+1:numinds...))
    return t_new
end

pad_with_zeros_to(t::TensorMap, spaces...) = pad_with_zeros_to(t, Dict(spaces))
