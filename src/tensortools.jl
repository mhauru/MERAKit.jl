# Utilities for creating and modifying vector spaces and TensorMaps.
# To be `included` in MERA.jl.

"""
A TensorMap from N indices to N indices.
"""
SquareTensorMap{N} = AbstractTensorMap{S1, N, N} where {S1}

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
Create a 2-to-2 disentangler, from `Vin ⊗ Vin` to `Vout ⊗ Vout`. The arguments `random` and
`T` set whether the disentangler should be a random isometry or the identity, and what its
element type should be.
"""
function initialize_disentangler(Vout, Vin, random, T)
    if random
        u = randomisometry(Vout ⊗ Vout, Vin ⊗ Vin, T)
    else
        if Vin == Vout
            u = isomorphism(Vout ⊗ Vout, Vin ⊗ Vin)
            T <: Complex && (u = complex(u))
        else
            uhalf = randomisometry(Vout, Vin, T)
            u = uhalf ⊗ uhalf
        end
    end
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
function depseudoserialize(::Type{T}, domstr, codomstr, eltyp, data
                          ) where T <: AbstractTensorMap
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
function pad_with_zeros_to(t::AbstractTensorMap, spacedict::Dict)
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

pad_with_zeros_to(t::AbstractTensorMap, spaces...) = pad_with_zeros_to(t, Dict(spaces))

"""
Fuse the domain and codomain of a TensorMap, and return the resulting matrix (as a
TensorMap).
"""
function convert_to_matrix(t::AbstractTensorMap)
    dom, codom = domain(t), codomain(t)
    domainfuser = isomorphism(dom, fuse(dom))
    codomainfuser = isomorphism(codom, fuse(codom))
    mt = codomainfuser' * t * domainfuser
    return mt
end

"""
For a Hermitian square tensor (fusing the domain and codomain into single indices), return a
lower and upper bound between which all its eigenvalues lie. This costs O(D^2) where D is
the matrix dimension.
"""
function gershgorin_bounds(t::AbstractTensorMap{S, N, N}) where {S, N}
    return gershgorin_bounds(convert(Array, convert_to_matrix(t)))
end

"""
For a square tensor (fusing the domain and codomain into single indices), return a list of
its Gershgorin discs, as pairs (c, r) where c is the centre and r is the radius. This costs
O(D^2) where D is the matrix dimension.
"""
function gershgorin_discs(t::AbstractTensorMap{S, N, N}) where {S, N}
    return gershgorin_discs(convert(Array, convert_to_matrix(t)))
end

function gershgorin_bounds(a::Array{S, 2}) where {S}
    nonhermiticity = norm(a - a')/norm(a)
    if nonhermiticity > 1e-12
        msg = "Can't compute gershgorin_bounds for a non-Hermitian operator (non-hermiticity = $(nonhermiticity))."
        throw(ArgumentError(msg))
    end
    discs = gershgorin_discs(a)
    lb = minimum(real(c) - r for (c, r) in discs)
    ub = maximum(real(c) + r for (c, r) in discs)
    return lb, ub
end

function gershgorin_discs(a::Array{S, 2}) where {S}
    d = size(a, 1)
    if size(a, 2) != d
        msg = "Can't apply Gershgorin disc theorem to a non-square matrix."
        throw(ArgumentError(msg))
    end
    centres = diag(a)
    abs_centres = abs.(centres)
    radii1 = dropdims(sum(abs.(a), dims=1), dims=1) .- abs_centres
    radii2 = dropdims(sum(abs.(a), dims=2), dims=2) .- abs_centres
    radii = min.(radii1, radii2)
    discs = tuple(zip(centres, radii)...)
    return discs
end

"""
Get the module for the type of tensor manifold specified by the Symbol `s`, which should one
of `:grassmann, :stiefel, :unitary`.
"""
function manifoldmodule(s::Symbol)
    if s === :grassmann
        manifold = Grassmann
    elseif s === :stiefel
        manifold = Stiefel
    elseif s === :unitary
        manifold = Unitary
    else
        msg = "Unknown tensor manifold type: $(isometrymanifold)"
        throw(ArgumentError(msg))
    end
    return manifold
end

"""
Precondition the tangent vector `X` at `W` using the preconditioned metric with the
positive definite tensor `rho`, i.e. Tr[X' Y rho].
"""
function precondition_tangent(X::Stiefel.StiefelTangent, rho::AbstractTensorMap)
    W, A, Z = X.W, X.A, X.Z
    E, U = eigh(rho)
    Einv = inv(real(sqrt(E^2 + precondition_regconst(rho)^2*id(domain(E)))))
    rhoinv = U * Einv * U'
    Z_prec = projectcomplement!(Z * rhoinv, W)
    A_prec = projectantihermitian!(symmetric_sylvester(E, U, 2*A))
    return Stiefel.StiefelTangent(W, A_prec, Z_prec)
end

function precondition_tangent(X::Grassmann.GrassmannTangent, rho::AbstractTensorMap)
    W, Z = X.W, X.Z
    E, U = eigh(rho)
    Einv = inv(real(sqrt(E^2 + precondition_regconst(rho)^2*id(domain(E)))))
    rhoinv = U * Einv * U'
    Z_prec = projectcomplement!(Z * rhoinv, W)
    return Grassmann.GrassmannTangent(W, Z_prec)
end

function precondition_tangent(X::Unitary.UnitaryTangent, rho::AbstractTensorMap)
    W, A = X.W, X.A
    E, U = eigh(rho)
    Einv = inv(real(sqrt(E^2 + precondition_regconst(rho)^2*id(domain(E)))))
    rhoinv = U * Einv * U'
    A_prec = projectantihermitian!(symmetric_sylvester(E, U, 2*A))
    return Unitary.UnitaryTangent(W, A_prec)
end

"""
The regularisation constant to use when inverting the density matrix in preconditioning.
"""
precondition_regconst(X) = sqrt(eps(real(float(one(eltype(X))))))

"""
Solve the Sylvester equation A X + X A = C, where we know A = A' and the arguments E and U
are the eigenvalue decomposition of A.
"""
function symmetric_sylvester(E::AbstractArray, U::AbstractArray, C::AbstractArray)
    temp1 = typeof(C)(undef, size(C))
    temp2 = typeof(C)(undef, size(C))
    mul!(temp1, U', C)
    mul!(temp2, temp1, U)
    for i in CartesianIndices(temp2)
        temp2[i] /= sqrt((E[i[1]] + E[i[2]])^2 + precondition_regconst(E)^2)
    end
    mul!(temp1, U, temp2)
    mul!(temp2, temp1, U')
    return temp2
end

function symmetric_sylvester(E::AbstractTensorMap, U::AbstractTensorMap,
                             C::AbstractTensorMap)
    cod = domain(C)
    dom = codomain(C)
    sylAB(c) = symmetric_sylvester(diag(block(E, c)), block(U, c), block(C, c))
    data = TensorKit.SectorDict(c => sylAB(c) for c in blocksectors(cod ← dom))
    return TensorMap(data, cod ← dom)
end
