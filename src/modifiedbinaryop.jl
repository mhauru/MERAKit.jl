# To be included in MERAKit.jl.

"""
    ModifiedBinaryOp{T}

The operator type for `ModifiedBinaryLayer`.

Due to the structure of a `ModifiedBinaryMERA`, there's an alternating pattern of two-site
translation invariance at each layer. Because of this, every operator at a layer is defined
by two tensors, one for each possible position. `ModifiedBinaryOp` encapsulates this. The
two positions are called `mid` and `gap`, where `mid` refers to the position that has a
disentangler directly below it, and `gap` to the position where the disentangler below is
missing. Both `op.mid` and `op.gap` are of type `T`.
"""
struct ModifiedBinaryOp{T}
    mid::T
    gap::T
end

ModifiedBinaryOp(op::T) where T = ModifiedBinaryOp{T}(op, op)
ModifiedBinaryOp(op::ModifiedBinaryOp) = op  # Makes some method signatures simpler to write
Base.convert(::Type{<: ModifiedBinaryOp}, op::AbstractTensorMap) = ModifiedBinaryOp(op)

function Base.convert(::Type{ModifiedBinaryOp{T}}, op::ModifiedBinaryOp) where {T}
    return ModifiedBinaryOp{T}(convert(T, op.mid), convert(T, op.gap))
end

Base.iterate(op::ModifiedBinaryOp) = (op.mid, Val(1))
Base.iterate(op::ModifiedBinaryOp, state::Val{1}) = (op.gap, Val(2))
Base.iterate(op::ModifiedBinaryOp, state::Val{2}) = nothing
Base.length(op::ModifiedBinaryOp) = 2

Base.eltype(op::ModifiedBinaryOp) = promote_type((eltype(x) for x in op)...)
Base.copy(op::ModifiedBinaryOp) = ModifiedBinaryOp((deepcopy(x) for x in op)...)
Base.adjoint(op::ModifiedBinaryOp) = ModifiedBinaryOp((x' for x in op)...)
Base.imag(op::ModifiedBinaryOp) = ModifiedBinaryOp((imag(x) for x in op)...)
Base.real(op::ModifiedBinaryOp) = ModifiedBinaryOp((real(x) for x in op)...)
# The below could just as well use op.gap.
Base.one(op::ModifiedBinaryOp) = ModifiedBinaryOp(one(op.mid))

TensorKit.storagetype(::Type{ModifiedBinaryOp{T}}) where {T} = storagetype(T)

function pad_with_zeros_to(op::ModifiedBinaryOp, args...)
    return ModifiedBinaryOp((pad_with_zeros_to(x, args...) for x in op)...)
end

function gershgorin_bounds(op::ModifiedBinaryOp)
    lb_mid, ub_mid = gershgorin_bounds(op.mid)
    lb_gap, ub_gap = gershgorin_bounds(op.gap)
    return min(lb_mid, lb_gap), max(ub_mid, ub_gap)
end

support(op::ModifiedBinaryOp) = support(op.mid)  # Could equally well be op.gap.

function expand_support(op::ModifiedBinaryOp, n::Integer)
    mid = expand_support(op.mid, n)
    gap = expand_support(op.gap, n)
    return ModifiedBinaryOp(mid, gap)
end

function Base.similar(op::ModifiedBinaryOp, ::Type{elementT} = eltype(op)) where {elementT}
    mid = similar(op.mid, elementT)
    gap = similar(op.gap, elementT)
    return ModifiedBinaryOp(mid, gap)
end

TensorKit.space(op::ModifiedBinaryOp) = space(op.gap)
TensorKit.space(op::ModifiedBinaryOp, n) = space(op.gap, n)
TensorKit.domain(op::ModifiedBinaryOp) = domain(op.gap)
TensorKit.codomain(op::ModifiedBinaryOp) = codomain(op.gap)

function Base.:(==)(op1::ModifiedBinaryOp, op2::ModifiedBinaryOp)
    return (op1.mid == op2.mid) && (op1.gap == op2.gap)
end

function Base.isapprox(op1::ModifiedBinaryOp, op2::ModifiedBinaryOp)
    return (op1.mid ≈ op2.mid) && (op1.gap ≈ op2.gap)
end

# Pass element-wise arithmetic down onto the AbstractTensorMaps. Promote AbstractTensorMaps
# to ModifiedBinaryOps if necessary.
for op in (:+, :-, :/, :*)
    eval(:(
        Base.$(op)(x::ModifiedBinaryOp, y::ModifiedBinaryOp)
        = ModifiedBinaryOp(($(op)(xi, yi) for (xi, yi) in zip(x, y))...)
    ))
    eval( :(
        Base.$(op)(x::AbstractTensorMap, y::ModifiedBinaryOp)
        = $(op)(ModifiedBinaryOp(x), y)
    ))
    eval(:(
        Base.$(op)(x::ModifiedBinaryOp, y::AbstractTensorMap)
        = $(op)(x, ModifiedBinaryOp(y))
    ))
end

Base.:*(op::ModifiedBinaryOp, a::Number) = ModifiedBinaryOp(op.mid * a, op.gap * a)
Base.:*(a::Number, op::ModifiedBinaryOp) = op*a
Base.:/(op::ModifiedBinaryOp, a::Number) = ModifiedBinaryOp(op.mid / a, op.gap / a)

function Base.copy!(op1::ModifiedBinaryOp, op2::ModifiedBinaryOp)
    copy!(op1.mid, op2.mid)
    copy!(op1.gap, op2.gap)
    return op1
end

function Base.fill!(op::ModifiedBinaryOp, a::Number)
    fill!(op.mid, a)
    fill!(op.gap, a)
    return op
end

function LinearAlgebra.dot(op1::ModifiedBinaryOp, op2::ModifiedBinaryOp)
    dotmid = dot(op1.mid, op2.mid)
    dotgap = dot(op1.gap, op2.gap)
    return (dotmid + dotgap) / 2.0
end

function LinearAlgebra.dot(op1::ModifiedBinaryOp, t2::AbstractTensorMap)
    return dot(op1, ModifiedBinaryOp(t2))
end

function LinearAlgebra.dot(t1::AbstractTensorMap, op2::ModifiedBinaryOp)
    return dot(ModifiedBinaryOp(t1), op2)
end

function LinearAlgebra.rmul!(op::ModifiedBinaryOp, a::Number)
    rmul!(op.mid, a)
    rmul!(op.gap, a)
    return op
end

function LinearAlgebra.lmul!(a::Number, op::ModifiedBinaryOp)
    lmul!(a, op.mid)
    lmul!(a, op.gap)
    return op
end

function BLAS.axpby!(a::Number, X::ModifiedBinaryOp, b::Number, Y::ModifiedBinaryOp)
    axpby!(a, X.mid, b, Y.mid)
    axpby!(a, X.gap, b, Y.gap)
    return Y
end

function BLAS.axpby!(a::Number, X::AbstractTensorMap, b::Number, Y::ModifiedBinaryOp)
    return axpby!(a, ModifiedBinaryOp(X), b, Y)
end

function BLAS.axpby!(a::Number, X::ModifiedBinaryOp, b::Number, Y::AbstractTensorMap)
    return axpby!(a, X, b, ModifiedBinaryOp(Y))
end

function BLAS.axpy!(a::Number, X::ModifiedBinaryOp, Y::ModifiedBinaryOp)
    axpy!(a, X.mid, Y.mid)
    axpy!(a, X.gap, Y.gap)
    return Y
end

function BLAS.axpy!(a::Number, X::AbstractTensorMap, Y::ModifiedBinaryOp)
    return axpy!(a, ModifiedBinaryOp(X), Y)
end

function BLAS.axpy!(a::Number, X::ModifiedBinaryOp, Y::AbstractTensorMap)
    return axpy!(a, X, ModifiedBinaryOp(Y))
end

function LinearAlgebra.mul!(C::ModifiedBinaryOp, A::ModifiedBinaryOp, B::Number)
    mul!(C.mid, A.mid, B)
    mul!(C.gap, A.gap, B)
    return C
end

function LinearAlgebra.mul!(C::ModifiedBinaryOp, A::Number, B::ModifiedBinaryOp)
    mul!(C.mid, A, B.mid)
    mul!(C.gap, A, B.gap)
    return C
end

function LinearAlgebra.mul!(C::AbstractTensorMap, A::ModifiedBinaryOp, B::Number)
    return mul!(ModifiedBinaryOp(C), A, B)
end

function LinearAlgebra.mul!(C::ModifiedBinaryOp, A::AbstractTensorMap, B::Number)
    return mul!(C, ModifiedBinaryOp(A), B)
end

function LinearAlgebra.mul!(C::AbstractTensorMap, A::Number, B::ModifiedBinaryOp)
    return mul!(ModifiedBinaryOp(C), A, B)
end

function LinearAlgebra.mul!(C::ModifiedBinaryOp, A::Number, B::AbstractTensorMap)
    return mul!(C, A, ModifiedBinaryOp(B))
end

LinearAlgebra.tr(op::ModifiedBinaryOp) = (tr(op.mid) + tr(op.gap)) / 2.0

# The entropy of the density matrix is the average over the two different density matrices.
function densitymatrix_entropy(rho::ModifiedBinaryOp)
    rho_mid, rho_gap = rho
    S_mid, S_gap = densitymatrix_entropy(rho_mid), densitymatrix_entropy(rho_gap)
    S = (S_mid + S_gap) / 2.0
    return S
end
