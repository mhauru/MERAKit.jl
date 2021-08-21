# Utilities for creating and modifying vector spaces and TensorMaps.
# To be `included` in MERAKit.jl.

"""
An `AbstractTensorMap` from N indices to N indices.
"""
SquareTensorMap{N} = AbstractTensorMap{S1, N, N} where {S1}

"""
A `Union` type of the different `TensorKitManifolds` tangent types: `GrassmannTangent`,
`StiefelTangent`, and `UnitaryTangent`.
"""
Tangent = Union{Grassmann.GrassmannTangent, Stiefel.StiefelTangent, Unitary.UnitaryTangent}

"""
    disentangler_type(::Type{ST}, ::Type{ET}, Tan::Bool)

Given the `IndexSpace` type `ST` and storage type `ET`, return the concrete type of a 2-to-2
disentangler. `Tan` is a flag for whether we want the `TensorMap` type itself (`Tan =
false`) or the corresponding tangent type, i.e. a `StiefelTangent`.

See also: [`ternaryisometry_type`](@ref), [`binaryisometry_type`](@ref)
"""
function disentangler_type(::Type{ST}, ::Type{ET}, Tan::Bool) where {ST, ET}
    TensorType = tensormaptype(ST, 2, 2, ET)
    if Tan
        TA = tensormaptype(ST, 2, 2, ET)
        DisType = Stiefel.StiefelTangent{TensorType, TA}
    else
        DisType = TensorType
    end
    return DisType
end

"""
    ternaryisometry_type(::Type{ST}, ::Type{ET}, Tan::Bool)

Given the `IndexSpace` type `ST` and storage type `ET`, return the concrete type of a 3-to-1
isometry. `Tan` is a flag for whether we want the `TensorMap` type itself (`Tan =
false`) or the corresponding tangent type, i.e. a `GrassmannTangent`.

See also: [`disentangler_type`](@ref), [`binaryisometry_type`](@ref)
"""
function ternaryisometry_type(::Type{ST}, ::Type{ET}, Tan::Bool) where {ST, ET}
    TensorType = tensormaptype(ST, 3, 1, ET)
    if Tan
        TU = tensormaptype(ST, 3, 1, ET)
        SET = isreal(sectortype(ST)) ? real(ET) : ET
        TS = tensormaptype(ST, 1, 1, SET)
        TV = tensormaptype(ST, 1, 1, ET)
        IsoType = Grassmann.GrassmannTangent{TensorType, TU, TS, TV}
    else
        IsoType = TensorType
    end
    return IsoType
end

"""
    binaryisometry_type(::Type{ST}, ::Type{ET}, Tan::Bool)

Given the `IndexSpace` type `ST` and storage type `ET`, return the concrete type of a 2-to-1
isometry. `Tan` is a flag for whether we want the `TensorMap` type itself (`Tan =
false`) or the corresponding tangent type, i.e. a `GrassmannTangent`.

See also: [`disentangler_type`](@ref), [`ternaryinaryisometry_type`](@ref)
"""
function binaryisometry_type(::Type{ST}, ::Type{ET}, Tan::Bool) where {ST, ET}
    TensorType = tensormaptype(ST, 2, 1, ET)
    if Tan
        TU = tensormaptype(ST, 2, 1, ET)
        SET = isreal(sectortype(ST)) ? real(ET) : ET
        TS = tensormaptype(ST, 1, 1, SET)
        TV = tensormaptype(ST, 1, 1, ET)
        IsoType = Grassmann.GrassmannTangent{TensorType, TU, TS, TV}
    else
        IsoType = TensorType
    end
    return IsoType
end

"""
    randomisometry(T, Vout, Vin)

Given an element type `T` and two vector spaces, `Vin` for the domain and `Vout` for the
codomain, return a Haar random isometry.

The implementation uses a QR decomposition of a Gaussian random matrix.
"""
randomisometry(::Type{T}, Vout, Vin) where {T} = TensorMap(randhaar, T, Vout ← Vin)

"""
    initialize_disentangler(T, Vout, Vin, random::Bool)

Initialize a disentangler from `Vin ⊗ Vin` to `Vout ⊗ Vout`, of element type `T`.

The returned tensor is Haar random if `random = true`. If `random = false` and `Vin == Vout`
it is the identity. If `random = false` and `Vin != Vout` it is the tensor product of two
one-site Haar random unitaries.
"""
function initialize_disentangler(::Type{T}, Vout, Vin, random::Bool) where {T}
    if random
        u = randomisometry(T, Vout ⊗ Vout, Vin ⊗ Vin)
    else
        if Vin == Vout
            u = isomorphism(Matrix{T}, Vout ⊗ Vout, Vin ⊗ Vin)
        else
            uhalf = randomisometry(T, Vout, Vin)
            u = uhalf ⊗ uhalf
        end
    end
    return u
end

"""
    support(op)

Return the number of sites/indices `N` that the operator `op` is supported on, assuming it
is an operator from `N` sites to `N` sites.

See also: [`expand_support`](@ref)
"""
support(op::SquareTensorMap{N}) where {N} = N

"""
    expand_support(op, n::Integer)

Given an operator from a `N` indices to `N` indices, expand its support to a larger number
of indices `n` by tensoring with the identity. The different ways of doing the expansion,
e.g. I ⊗ op and op ⊗ I, are averaged over.

See also: [`support`](@ref)
"""
@inline expand_support(op::SquareTensorMap, n::Int) = _expand_support(op, Val(n))

@noinline function _expand_support(op::SquareTensorMap{N}, ::Val{n}) where {N, n}
    if n <= N
        return op
    else
        dom = ProductSpace(ntuple(i->space(op, 1), n)...)
        op2 = fill!(similar(op, dom, dom), 0)
        V = space(op, 1)
        eye = id(storagetype(op), V)
        for k = 0:n-N
            eyes1 = Base.fill_to_length((), eye, Val(k))
            eyes2 = Base.fill_to_length((), eye, Val(n-N-k))
            coeff = factorial(n-N)/(factorial(k)*factorial(n-N-k)) / 2^(n-N)
            axpy!(coeff, ⊗(eyes1..., op, eyes2...), op2)
            # TODO This would generate the uniform sum. See if it changes something.
            # axpy!(1/(n-N+1), ⊗(eyes1..., op, eyes2...), op2)
        end
        return op2
    end
end

"""
    remove_symmetry(V)

Strip a vector space of its symmetry structure, i.e. return the corresponding
`ℂ^n` or `ℝ^n`.
"""
remove_symmetry(V::ElementarySpace{ℝ}) = CartesianSpace(dim(V))
remove_symmetry(V::ElementarySpace{ℂ}) = ComplexSpace(dim(V), isdual(V))

"""
    remove_symmetry(t::AbstractTensorMap)

Strip an `AbstractTensorMap` of its internal symmetries, and return the corresponding
`TensorMap` that operates on `ComplexSpace` or `CartesianSpace`.
"""
function remove_symmetry(t::TensorMap)
    dom = domain(t)
    cod = codomain(t)
    S = field(spacetype(t)) === ℝ ? CartesianSpace : ComplexSpace
    domain_nosym = ProductSpace{S}(map(remove_symmetry, tuple(dom...)))
    codomain_nosym = ProductSpace{S}(map(remove_symmetry, tuple(cod...)))
    t_nosym = TensorMap(convert(Array, t), codomain_nosym ← domain_nosym)
    return t_nosym
end

"""
    expand_vectorspace(V, newdim)

Given a vector space `V` and a dictionary `newdim` of dimensions for the various irrep
sectors, return another vector space of the same type, but with these new dimension. If some
irrep sectors are not in the dictionary, the dimensions of the original space are used.
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

function expand_vectorspace(V::GradedSpace, newdims)
    olddims = Dict(s => dim(V, s) for s in sectors(V))
    sectordict = merge(olddims, newdims)
    return typeof(V)(sectordict; dual = V.dual)
end

# If the first argument given to depseudoserialize is a String, we assume its a
# representation of a an object that can `eval`uated. So we evaluate it and call
# depseudoserialize again.
depseudoserialize(str::String, args...) = depseudoserialize(eval(Meta.parse(str)), args...)

function pseudoserialize(t::T) where T <: TensorMap
    # We make use of the nice fact that many TensorKit objects return on repr strings that
    # are valid syntax to reconstruct these objects.
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

function depseudoserialize(
    ::Type{T}, domstr, codomstr, eltyp, data
) where {T <: AbstractTensorMap}
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
    append_dummy_index(t::TensorMap)

Create a copy of `t`, but with an extra, trivial index as the last index of the domain.
"""
function append_dummy_index(t::TensorMap)
    return TensorMap(t.data, codomain(t), insertunit(domain(t)))
end

"""
    remove_dummy_index(t::TensorMap)

Assuming the last index of the domain of `t` is a trivial dummy index, return a `t` but with
that index removed.
"""
function remove_dummy_index(t::TensorMap{S}) where {S}
    @assert space(t, numind(t)) == oneunit(S)'
    dom = ProductSpace{S}(Base.front(domain(t).spaces))
    return TensorMap(t.data, codomain(t), dom)
end

"""
    pad_with_zeros_to(t::AbstractTensorMap, spacedict::Dict)

Transform `t` to change the vector spaces of its indices, by throwing elements away or
padding the tensor with zeros.

`spacedict` is a dictionary with index labels (1, 2, 3, ...) as keys, and `VectorSpace`s as
values. It tells us which indices should have their space changed, and to what. Instead of a
dictionary, a varargs of `Pair`s `index => vectorspace` also works.

For each index `i`, its current space `Vorig = space(t, i)` and new space
`Vnew = spacedict[i]` should be of the same type. If `Vnew` is strictly larger than `Vold`
then `t` is padded with zeros to fill in the new elements. Otherwise some elements of `t`
will be truncated away.
"""
function pad_with_zeros_to(t::AbstractTensorMap, spacedict::Dict)
    S = spacetype(t)
    newcodomainspaces = tuple(codomain(t)...)
    for i = 1:length(newcodomainspaces)
        if haskey(spacedict, i)
            @assert isdual(spacedict[i]) == isdual(space(t, i))
            # unpredictable behaviour otherwise
            newcodomainspaces = Base.setindex(newcodomainspaces, spacedict[i], i)
        end
    end
    newdomainspaces = tuple(domain(t)...)
    for i = 1:length(newdomainspaces)
        j = i + numout(t)
        if haskey(spacedict, j)
            @assert isdual(spacedict[j]) == isdual(space(t, j))
            # unpredictable behaviour otherwise
            newdomainspaces = Base.setindex(newdomainspaces, spacedict[j]', i)
        end
    end
    newcodomain = ProductSpace{S}(newcodomainspaces)
    newdomain = ProductSpace{S}(newdomainspaces)
    tnew = fill!(similar(t, newcodomain, newdomain), zero(eltype(t)))
    for (f1, f2) in fusiontrees(t)
        a = t[f1, f2]
        anew = tnew[f1, f2]
        axes = Base.OneTo.(min.(size(a), size(anew)))
        copy!(view(anew, axes...), view(a, axes...))
    end
    return tnew
end

pad_with_zeros_to(t::AbstractTensorMap, spaces...) = pad_with_zeros_to(t, Dict(spaces))

"""
    gershgorin_bounds(t::AbstractTensorMap)

For a Hermitian square tensor (a square matrix after fusing the domain and codomain into
single indices), return a lower and upper bound between which all its eigenvalues lie.

This costs O(D^2) time, where D is the matrix dimension.

See also: [`gershgorin_discs`](@ref)
"""
function gershgorin_bounds(t::SquareTensorMap)
    v = real(first(last(first(blocks(t))))) # get first element from first block
    lb, ub = v, v
    largest_bound((lb1, ub1), (lb2, ub2)) = (min(lb1, lb2), max(ub1, ub2))
    for (c, b) in blocks(t)
        lb, ub = largest_bound((lb, ub), gershgorin_bounds(b))
    end
    return lb, ub
end

function gershgorin_bounds(a::Array{S, 2}) where {S}
    nonhermiticity = norm(a - a')/norm(a)
    if nonhermiticity > 1e-12
        msg = "Computing gershgorin_bounds, ignoring non-hermiticity of $(nonhermiticity)."
        @warn(msg)
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
    radii1 = dropdims(sum(abs.(a), dims = 1), dims = 1) .- abs_centres
    radii2 = dropdims(sum(abs.(a), dims = 2), dims = 2) .- abs_centres
    radii = min.(radii1, radii2)
    discs = collect(zip(centres, radii))
    return discs
end

"""
    densitymatrix_entropy(rho)

Compute the von Neumann entropy of a density matrix `rho`.
"""
function densitymatrix_entropy(rho)
    eigs = eigh(rho)[1]
    S = -dot(eigs, log(eigs))
    return S
end

"""
    precondition_tangent(X::Tangent, rho::AbstractTensorMap,
                         delta = precondition_regconst(X))

Precondition the tangent vector `X` with the metrix g(X, Y) =  Tr[X' Y rho], where `rho`
positive definite tensor.

This works for `X` being a `StiefelTangent`, `GrassmannTangent`, or `UnitaryTangent`.

Preconditioning requires inverting `rho`. `delta` is a threshold parameter for regularising
that inverse. The regularised inverse is S -> 1/sqrt(S^2 + delta^2).

See also: [`precondition_regconst`](@ref)
"""
function precondition_tangent(
    X::Stiefel.StiefelTangent, rho::AbstractTensorMap, delta = precondition_regconst(X)
)
    W, A, Z = X.W, X.A, X.Z
    E, U = eigh(rho)
    Einv = reginv_E(E, delta)
    rhoinv = U * Einv * U'
    Z_prec = projectcomplement!(Z * rhoinv, W)
    A_prec = projectantihermitian!(symmetric_sylvester(E, U, 2*A, delta))
    return Stiefel.StiefelTangent(W, A_prec, Z_prec)
end

function precondition_tangent(
    X::Grassmann.GrassmannTangent, rho::AbstractTensorMap, delta = precondition_regconst(X)
)
    W, Z = X.W, X.Z
    E, U = eigh(rho)
    Einv = reginv_E(E, delta)
    rhoinv = U * Einv * U'
    Z_prec = projectcomplement!(Z * rhoinv, W)
    return Grassmann.GrassmannTangent(W, Z_prec)
end

function precondition_tangent(
    X::Unitary.UnitaryTangent, rho::AbstractTensorMap, delta = precondition_regconst(X)
)
    W, A = X.W, X.A
    E, U = eigh(rho)
    A_prec = projectantihermitian!(symmetric_sylvester(E, U, 2*A, delta))
    return Unitary.UnitaryTangent(W, A_prec)
end

function reginv_E(E, delta)
    E_isreal = eltype(E) <: Real && isreal(sectortype(E))
    Ereg = sqrt(E^2 + delta^2*id(domain(E)))
    if E_isreal
        return inv(real(Ereg))
    else
        return inv(Ereg)
    end
end

"""
    precondition_regconst(X::Tangent)

The default regularisation constant to use when inverting the density matrix `rho` in
preconditioning.

See also: [`precondition_tangent`](@ref)
"""
function precondition_regconst(X::Tangent)
    delta = sqrt(eps(real(float(one(eltype(X.W))))))
    delta = max(delta, norm(X) / 100.0)
    return delta
end

"""
    symmetric_sylvester(E, U, C, delta)

Solve the Sylvester equation `A*X + X*A = C`, where we know that `A = A'` and the arguments
`E` and `U` are the eigenvalue decomposition of `A = U*E*U'`. This requires performing a
matrix inversion of `1 ⊗ A + A ⊗ 1`, and that inverse is regularised as
`X -> 1/sqrt(X^2 + delta^2)`.
"""
function symmetric_sylvester(
    E::AbstractTensorMap, U::AbstractTensorMap, C::AbstractTensorMap, delta
)
    cod = domain(C)
    dom = codomain(C)
    sylAB(c) = symmetric_sylvester(diag(block(E, c)), block(U, c), block(C, c), delta)
    data = TensorKit.SectorDict(c => sylAB(c) for c in blocksectors(cod ← dom))
    return TensorMap(data, cod ← dom)
end

function symmetric_sylvester(E::AbstractVector, U::AbstractMatrix, C::AbstractMatrix, delta)
    temp1 = typeof(C)(undef, size(C))
    temp2 = typeof(C)(undef, size(C))
    mul!(temp1, U', C)
    mul!(temp2, temp1, U)
    for i in CartesianIndices(temp2)
        temp2[i] /= sqrt((E[i[1]] + E[i[2]])^2 + delta^2)
    end
    mul!(temp1, U, temp2)
    mul!(temp2, temp1, U')
    return temp2
end
