# Utilities for creating and modifying vector spaces and TensorMaps.
# To be `included` in MERA.jl.

"""
A TensorMap from N indices to N indices.
"""
SquareTensorMap{N} = TensorMap{S1, N, N} where {S1}

"""
Given two vector spaces, create an isometric/unitary TensorMap from one to the other. This
is done by creating a random Gaussian tensor and SVDing it. If `symmetry_permutation` is
given, symmetrise the random tensor over this permutation before doing the SVD.
"""
function randomisometry(Vout, Vin, T=ComplexF64; symmetry_permutation=nothing)
    temp = TensorMap(randn, T, Vout ← Vin)
    if symmetry_permutation !== nothing
        temp = temp + permute(temp, symmetry_permutation...)
    end
    U, S, Vt = tsvd(temp)
    u = U * Vt
    return u
end

"""
Return the number of sites/indices `m` that an operator is supported on, assuming it is an
operator from `m` sites to `m` sites.
"""
support(op::SquareTensorMap{N}) where {N} = N

"""
Given a TensorMap from a number of indices to the same number of indices, expand its support
to a larger number of indices `n` by tensoring with the identity. The different ways of
doing the expansion, e.g. I ⊗ op and op ⊗ I, are averaged over.
"""
function expand_support(op::SquareTensorMap{N}, n::Integer) where {N}
    V = space(op, 1)
    eye = id(V)
    op_support = N
    while op_support < n
        opeye = op ⊗ eye
        eyeop = eye ⊗ op
        op = (opeye + eyeop)/2
        op_support += 1
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
    olddims = Dict(s => dim(V, s) for s in sectors(V))
    sectordict = merge(olddims, newdims)
    return typeof(V)(sectordict; dual=V.dual)
end

"""
If the first argument given to depseudoserialize is a String, we assume its a representation
of a an object that can `eval`uated. So we evaluate it and call depseudoserialize again.
"""
depseudoserialize(str::String, args...) = depseudoserialize(eval(Meta.parse(str)), args...)

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
        data = deepcopy(t.data)
    else
        data = Dict(repr(s) => deepcopy(d) for (s, d) in t.data)
    end
    return repr(T), domstr, codomstr, eltyp, data
end

"""
Reconstruct a TensorMap given the output of `pseudoserialize`.
"""
function depseudoserialize(::Type{T}, domstr, codomstr, eltyp, data) where T <: TensorMap
    # We make use of the nice fact that many TensorKit objects return on repr
    # strings that are valid syntax to reconstruct these objects.
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
    idmat(T, shp) = Array{T}(I, shp)
    expanders = [TensorMap(idmat, eltype(t), V ← space(t, ind)) for (ind, V) in spacedict]
    sizedomain = length(domain(t))
    sizecodomain = length(codomain(t))
    # Prepare the @ncon call that contracts each index of `t` with the corresponding
    # expander, if one exists.
    numinds = sizedomain + sizecodomain
    inds_t = [ind in keys(spacedict) ? ind : -ind for ind in 1:numinds]
    inds_expanders = [[-ind, ind] for ind in keys(spacedict)]
    tensors = [t, expanders...]
    inds = [inds_t, inds_expanders...]
    t_new_tensor = @ncon(tensors, inds)
    # Permute inds to have the codomain and domain match with those of the input.
    t_new = permute(t_new_tensor, tuple(1:sizecodomain...),
                    tuple(sizecodomain+1:numinds...))
    return t_new
end

pad_with_zeros_to(t::TensorMap, spaces...) = pad_with_zeros_to(t, Dict(spaces))

"""
Inner product of two tensors as tangents on a Stiefel manifold. The first argument is the
point on the manifold that we are at, the next two are the tangent vectors.
"""
function stiefel_inner(t::TensorMap, t1::TensorMap, t2::TensorMap)
    # TODO Could write a faster version for unitaries, where the two terms are the same.
    # TODO a1 and a2 are supposed to be skew-symmetric. Should we enforce or assume that?
    a1 = t'*t1
    a2 = t'*t2
    inner = tr(t1'*t2) - 0.5*tr(a1'*a2)
    return inner
end

"""
Starting from the point `u` on a Stiefel manifold of unitary matrices, travel along the
geodesic generated by the tangent `utan` for distance `alpha`, and return the end-point and
the tangent of the geodesic at the end-point. Both `x` and `tan` maybe tensors instead of
matrices, and the return values will be reshaped to the same shape as the inputs.

See page 14 of https://www.cis.upenn.edu/~cis515/Stiefel-Grassmann-manifolds-Edelman.pdf for
how this is done.
"""
function stiefel_geodesic_unitary(u::TensorMap, utan::TensorMap, alpha::Number)
    a = u' * utan
    # In a perfect world, a is already skew-symmetric, but for numerical errors we enforce
    # that.
    # TODO Should we instead raise a warning if this does not already hold?
    a = (a - a')/2
    m = exp(alpha * a)
    u_end = u * m
    utan_end = utan * m
    # Creeping numerical errors may cause loss of isometricity, so explicitly isometrize.
    # TODO Maybe check that S is almost all ones, alert the user if it's not.
    U, S, Vt = tsvd(u_end)
    u_end = U*Vt
    return u_end, utan_end
end

"""
Starting from the point `w` on a Stiefel manifold of isometric matrices, travel along the
geodesic generated by the tangent `wtan` for distance `alpha`, and return the end-point and
the tangent of the geodesic at the end-point. Both `x` and `tan` maybe tensors instead of
matrices, and the return values will be reshaped to the same shape as the inputs.

See page 14 of https://www.cis.upenn.edu/~cis515/Stiefel-Grassmann-manifolds-Edelman.pdf for
how this is done.
"""
function stiefel_geodesic_isometry(w::TensorMap, wtan::TensorMap, alpha::Number)
    a = w' * wtan
    # In a perfect world, a is already skew-symmetric, but for numerical errors we enforce
    # that.
    # TODO Should we instead raise a warning if this does not already hold?
    a = (a - a')/2
    k = wtan - w * a
    q, r = leftorth(k)
    b = catcodomain(catdomain(a, -r'), catdomain(r, zero(a)))
    expb = exp(alpha * b)
    eye = id(domain(a))
    uppertrunc = catcodomain(eye, zero(eye))
    lowertrunc = catcodomain(zero(eye), eye)
    m = uppertrunc' * expb * uppertrunc
    n = lowertrunc' * expb * uppertrunc
    w_end = w*m + q*n
    wtan_end = wtan*m - w*r'*n
    # Creeping numerical errors may cause loss of isometricity, so explicitly isometrize.
    # TODO Maybe check that S is almost all ones, alert the user if it's not.
    U, S, Vt = tsvd(w_end)
    w_end = U*Vt
    return w_end, wtan_end
end

"""
Return true or false for whether `utan` is a Stiefel-manifold-tangent at the point `u`. In
other words, return whether u' * u is (approximately) anti-Hermitian.
"""
function istangent_isometry(u, utan)
    a = u' * utan
    return all(a ≈ -a')
end

# The Cayley transformations need the same kind of things computed for both retractions and
# transport, and we don't want to compute them again repeatedly. Nor do we want to recompute
# everything when we for instance do retractions along the same curve by different amounts.
# So here's a global Least-Recent-Use (LRU) cache that keeps track of the precursors that
# would otherwise be repeteadly recomputed.
g_cachesize = 40
g_cayleycache = LRU{Tuple{TensorMap, TensorMap}, Dict}(; maxsize=g_cachesize)

"""
Get the precursors for doing a Cayley retraction or parallel transport from a Stiefel
manifold point `x` along the tangent `tan`. The results are cached in a global cache.
"""
function get_cayley_precursors(x::TensorMap, tan::TensorMap)
    if (x, tan) in keys(g_cayleycache)
        precursors = g_cayleycache[(x, tan)]
    else
        precursors = cayley_precursors!(x, tan)
        g_cayleycache[(x, tan)] = precursors
    end
    return precursors
end

"""
Get the precursors for doing a Cayley retraction or parallel transport from a Stiefel
manifold point `x` along the tangent `tan` by distance `alpha`. The results are cached in a
global cache.
"""
function get_cayley_precursors(x::TensorMap, tan::TensorMap, alpha::Number)
    precursors = get_cayley_precursors(x, tan)
    precursors = cayley_precursors!(x, tan, alpha, precursors)
    return precursors
end

"""
For a given point `x` on a Stiefel manifold and a tangent `tan` at this point, compute
various precursor-matrices that will be needed for doing Cayley transformations for
retraction and transport. See `cayley_retract` and `cayley_transport` for more.

The name has a ! at the end because some methods of this function modify an existing set of
precursors.
"""
function cayley_precursors!(x::TensorMap, tan::TensorMap)
    dom = domain(x)
    domfuser = isomorphism(fuse(dom), dom)
    xf = x * domfuser'
    tanf = tan * domfuser'
    xtanf = xf' * tanf
    Ptanf = tanf - 0.5*(xf * xtanf)
    u = catdomain(Ptanf, xf)
    v = catdomain(xf, -Ptanf)
    m1 = v' * x
    m2 = v' * u
    precursors = Dict{Symbol, Any}()
    precursors[:u] = u
    precursors[:v] = v
    precursors[:m1] = m1
    precursors[:m2] = m2
    precursors[:alphadict] = Dict{Float64, Any}()
    return precursors
end

"""
For a given point `x` on a Stiefel manifold and a tangent `tan` at this point, compute
various precursor-matrices that will be needed for doing Cayley transformations for
retraction and transport, and specifically matrices required for doing them by a distance
alpha. See `cayley_retract` and `cayley_transport` for more.
"""
function cayley_precursors!(x::TensorMap, tan::TensorMap, alpha::Number)
    precursors = cayley_precursors!(x, tan)
    return cayley_precursors!(x, tan, alpha, precursors)
end

function cayley_precursors!(x::TensorMap, tan::TensorMap, alpha::Number, precursors::Dict)
    alphadict = precursors[:alphadict]
    if !(alpha in keys(alphadict))
        u, m2 = precursors[:u], precursors[:m2]
        eye = id(domain(u))
        Minv = inv(eye - (alpha/2) * m2)
        uMinv = u * Minv
        alphadict[alpha] = (Minv, uMinv)
    end
    return precursors
end

"""
For a point `x` on a Stiefel manifold and a tangent `tan` at this point, generate a curve in
the direction of `tan` using a Cayley transform, travel along that curve for distance
`alpha`, and return the end-point and the tangent at the end-point. Both `x` and `tan` maybe
tensors instead of matrices, and the return values will be reshaped to the same shape as the
inputs.

See Section 3.1 of http://www.optimization-online.org/DB_FILE/2016/09/5617.pdf for how this
works.
"""
function cayley_retract(x::TensorMap, tan::TensorMap, alpha::Number)
    precursors = get_cayley_precursors(x, tan, alpha)
    u, m1, m2 = precursors[:u], precursors[:m1], precursors[:m2]
    Minv, uMinv = precursors[:alphadict][alpha]
    m3 = Minv * m1
    m23 = m2 * m3
    x_end = x + alpha * u * m3
    tan_end = u * (m1 + (alpha/2) * m23 + (alpha/2) * Minv * m23)
    return x_end, tan_end
end

"""
For a point `x` on a Stiefel manifold and two tangents `tan` and `vec` at this point,
generate a curve in the direction of `tan` using a Cayley transform, and transport `vec`
along that curve for distance `alpha`, using a form of isometric Cayley transport. Return
the transported vector.  `x`, `tan`, and `vec` maybe tensors instead of matrices, and the
return values will be reshaped to the same shape as the inputs.

Note that the returned vector is a Stiefel-tangent-vector for the point return by
`cayley_retract(x, tan, alpha)`, but the transport isn't a generalization of the retraction
in the sense that `cayley_transport(x, tan, tan, alpha)` does not return the tangent vector
at the end point.

The transport implemented here is the one from Section 3.3 of
http://www.optimization-online.org/DB_FILE/2016/09/5617.pdf. See there for the mathematical
details.
"""
function cayley_transport(x::TensorMap, tan::TensorMap, vec::TensorMap, alpha::Number)
    precursors = get_cayley_precursors(x, tan, alpha)
    u, v = precursors[:u], precursors[:v]
    Minv, uMinv = precursors[:alphadict][alpha]
    vec_end = vec + alpha * uMinv * (v' * vec)
    return vec_end
end
