# ModifiedBinaryLayer and ModifiedBinaryMERA types, and methods thereof.
# To be `included` in MERA.jl.

# # # The core stuff

# Index numbering convention is as follows, where the physical indices are at the bottom:
# Disentangler:
#  3|   4|
#  +------+
#  |  u   |
#  +------+
#  1|   2|
#
# Isometries:
#    3|
#  +------+
#  |  w   |
#  +------+
#  1|   2|
#
#  The isometries are called left and right, or wl and wr, and their location is with
#  respect to the disentangler below them:
#  |     |
#  wl   wr
#  | \ / |
#  |  u  |
#  | / \ |

struct ModifiedBinaryLayer{ST, ET, Tan} <: SimpleLayer
    disentangler::Any
    isometry_left::Any
    isometry_right::Any

    function ModifiedBinaryLayer{ST, ET, Tan}(disentangler, isometry_left, isometry_right
                                             ) where {ST, ET, Tan}
        DisType = disentangler_type(ST, ET, Tan)
        IsoType = binaryisometry_type(ST, ET, Tan)
        disconv = convert(DisType, disentangler)::DisType
        isoleftconv = convert(IsoType, isometry_left)::IsoType
        isorightconv = convert(IsoType, isometry_right)::IsoType
        return new{ST, ET, Tan}(disconv, isoleftconv, isorightconv)
    end
end

function ModifiedBinaryLayer(disentangler::DisType, isometry_left::IsoType,
                             isometry_right::IsoType
                            ) where {ST,
                                     DisType <: AbstractTensorMap{ST, 2, 2},
                                     IsoType <: AbstractTensorMap{ST, 2, 1}}
    ET = eltype(DisType)
    @assert eltype(IsoType) === ET
    return ModifiedBinaryLayer{ST, ET, false}(disentangler, isometry_left, isometry_right)
end

function ModifiedBinaryLayer(disentangler::DisTanType, isometry_left::IsoTanType,
                             isometry_right::IsoTanType
                            ) where {ST,
                                     DisType <: AbstractTensorMap{ST, 2, 2},
                                     IsoType <: AbstractTensorMap{ST, 2, 1},
                                     DisTanType <: Stiefel.StiefelTangent{DisType},
                                     IsoTanType <: Grassmann.GrassmannTangent{IsoType}}
    ET = eltype(DisType)
    @assert eltype(IsoType) === ET
    return ModifiedBinaryLayer{ST, ET, true}(disentangler, isometry_left, isometry_right)
end

function Base.getproperty(l::ModifiedBinaryLayer{ST, ET, Tan}, sym::Symbol
                         ) where {ST, ET, Tan}
    if sym === :disentangler
        T = disentangler_type(ST, ET, Tan)
    elseif sym === :isometry_left || sym === :isometry_right
        T = binaryisometry_type(ST, ET, Tan)
    else
        T = Any
    end
    return getfield(l, sym)::T
end

ModifiedBinaryMERA{N} = GenericMERA{N, T, O} where {T <: ModifiedBinaryLayer, O}

# Given an instance of a type like ModifiedBinaryLayer{TensorMap, TensorMap, TensorMap},
# return the unparametrised type ModifiedBinaryLayer.
layertype(::ModifiedBinaryLayer) = ModifiedBinaryLayer
layertype(::Type{T}) where T <: ModifiedBinaryMERA = ModifiedBinaryLayer

# Implement the iteration and indexing interfaces. Allows things like `u, wl, wr = layer`.
Base.iterate(layer::ModifiedBinaryLayer) = (layer.disentangler, Val(1))
Base.iterate(layer::ModifiedBinaryLayer, ::Val{1}) = (layer.isometry_left, Val(2))
Base.iterate(layer::ModifiedBinaryLayer, ::Val{2}) = (layer.isometry_right, Val(3))
Base.iterate(layer::ModifiedBinaryLayer, ::Val{3}) = nothing
Base.length(layer::ModifiedBinaryLayer) = 3

"""
The ratio by which the number of sites changes when we go down through this layer.
"""
scalefactor(::Type{<:ModifiedBinaryLayer}) = 2
scalefactor(::Type{ModifiedBinaryMERA}) = scalefactor(ModifiedBinaryLayer)

get_disentangler(m::ModifiedBinaryMERA, depth) = get_layer(m, depth).disentangler
get_isometry_left(m::ModifiedBinaryMERA, depth) = get_layer(m, depth).isometry_left
get_isometry_right(m::ModifiedBinaryMERA, depth) = get_layer(m, depth).isometry_right

function set_disentangler!(m::ModifiedBinaryMERA, u, depth; kwargs...)
    wl = get_isometry_left(m, depth)
    wr = get_isometry_right(m, depth)
    return set_layer!(m, (u, wl, wr), depth; kwargs...)
end

function set_isometry_left!(m::ModifiedBinaryMERA, wl, depth; kwargs...)
    u = get_disentangler(m, depth)
    wr = get_isometry_right(m, depth)
    return set_layer!(m, (u, wl, wr), depth; kwargs...)
end

function set_isometry_right!(m::ModifiedBinaryMERA, wr, depth; kwargs...)
    u = get_disentangler(m, depth)
    wl = get_isometry_left(m, depth)
    return set_layer!(m, (u, wl, wr), depth; kwargs...)
end

causal_cone_width(::Type{<:ModifiedBinaryLayer}) = 2

outputspace(layer::ModifiedBinaryLayer) = space(layer.disentangler, 1)
inputspace(layer::ModifiedBinaryLayer) = space(layer.isometry_left, 3)'
internalspace(layer::ModifiedBinaryLayer) = space(layer.isometry_right, 1)
internalspace(m::ModifiedBinaryMERA, depth) = internalspace(get_layer(m, depth))

"""
Return a new layer where the isometries have been padded with zeros to change the input
(top) vector space to be V_new.
"""
function expand_inputspace(layer::ModifiedBinaryLayer, V_new)
    u, wl, wr = layer
    wl = pad_with_zeros_to(wl, 3 => V_new')
    wr = pad_with_zeros_to(wr, 3 => V_new')
    return ModifiedBinaryLayer(u, wl, wr)
end

"""
Return a new layer where the disentanglers and isometries have been padded with zeros to
change the output (bottom) vector space to be V_new.
"""
function expand_outputspace(layer::ModifiedBinaryLayer, V_new)
    u, wl, wr = layer
    u = pad_with_zeros_to(u, 1 => V_new, 2 => V_new)
    wl = pad_with_zeros_to(wl, 1 => V_new)
    wr = pad_with_zeros_to(wr, 2 => V_new)
    return ModifiedBinaryLayer(u, wl, wr)
end

"""
Return a new layer where the disentanglers and isometries have been padded with zeros to
change the internal vector space to be V_new.
"""
function expand_internalspace(layer::ModifiedBinaryLayer, V_new)
    u, wl, wr = layer
    u = pad_with_zeros_to(u, 3 => V_new', 4 => V_new')
    wl = pad_with_zeros_to(wl, 2 => V_new)
    wr = pad_with_zeros_to(wr, 1 => V_new)
    return ModifiedBinaryLayer(u, wl, wr)
end

"""
Return a layer with random tensors, with `Vin` and `Vout` as the input and output spaces.
The optionalargument `Vint` is the output bond dimension of the disentangler. If
`random_disentangler=true`, the disentangler is also a random unitary, if `false` (default),
it is the identity or the product of two single-site isometries, depending on if `u` is
supposed to be unitary or isometric. `T` is the data type for the tensors, by default
`ComplexF64`.
"""
function randomlayer(::Type{ModifiedBinaryLayer}, T, Vin, Vout, Vint=Vout;
                     random_disentangler=false)
    wl = randomisometry(T, Vout ⊗ Vint, Vin)
    # We make the initial guess be reflection symmetric, since that's often true of the
    # desired MERA too (at least if random_disentangler is false, but we do it every time
    # any way).
    wr = deepcopy(permute(wl, (2,1), (3,)))
    u = initialize_disentangler(T, Vout, Vint, random_disentangler)
    return ModifiedBinaryLayer(u, wl, wr)
end

# # # ModifiedBinaryOp
# Due to the structure of the ModifiedBinaryMERA, there's an alternating pattern of two-site
# translation invariance at each layer. Because of this, every operator at a layer is
# defined by two tensors, one for each possible position. We create a `struct` for
# encapsulating this, so that all the necessary operations like trace and matrix product can
# be defined for such operators. We call the two positions `mid` and `gap`, where `mid`
# refers to the position that has a disentangler directly below it, and `gap` to the
# position where the disentangler below is missing.

struct ModifiedBinaryOp{T}
    mid::T
    gap::T
end

ModifiedBinaryOp(op::T) where T = ModifiedBinaryOp{T}(op, op)
ModifiedBinaryOp(op::ModifiedBinaryOp) = op  # makes some method signatures simpler to write
Base.convert(::Type{<: ModifiedBinaryOp}, op::AbstractTensorMap) = ModifiedBinaryOp(op)

function Base.convert(::Type{ModifiedBinaryOp{T}}, op::ModifiedBinaryOp) where {T}
    return ModifiedBinaryOp{T}(convert(T, op.mid), convert(T, op.gap))
end

function operatortype(::Type{ModifiedBinaryLayer{ST, ET, false}}
                     ) where {ST, ET}
    return ModifiedBinaryOp{tensortype(ST, Val(2), Val(2), ET)}
end
operatortype(::Type{ModifiedBinaryLayer{ST, ET, true}}) where {ST, ET} = Nothing

Base.iterate(op::ModifiedBinaryOp) = (op.mid, Val(1))
Base.iterate(op::ModifiedBinaryOp, state::Val{1}) = (op.gap, Val(2))
Base.iterate(op::ModifiedBinaryOp, state::Val{2}) = nothing
Base.length(op::ModifiedBinaryOp) = 2

Base.eltype(op::ModifiedBinaryOp) = promote_type((eltype(x) for x in op)...)
Base.copy(op::ModifiedBinaryOp) = ModifiedBinaryOp((deepcopy(x) for x in op)...)
Base.adjoint(op::ModifiedBinaryOp) = ModifiedBinaryOp((x' for x in op)...)
Base.imag(op::ModifiedBinaryOp) = ModifiedBinaryOp((imag(x) for x in op)...)
Base.real(op::ModifiedBinaryOp) = ModifiedBinaryOp((real(x) for x in op)...)

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

function Base.similar(op::ModifiedBinaryOp, ::Type{elementT}=eltype(op)) where {elementT}
    mid = similar(op.mid, elementT)
    gap = similar(op.gap, elementT)
    return ModifiedBinaryOp(mid, gap)
end

TensorKit.space(op::ModifiedBinaryOp) = space(op.gap)
TensorKit.domain(op::ModifiedBinaryOp) = domain(op.gap)
TensorKit.codomain(op::ModifiedBinaryOp) = codomain(op.gap)

# TODO This whole thing is very messy and ad hoc, with the interplay of TensorMaps and
# ModifiedBinaryOps.

# Pass element-wise arithmetic down onto the AbstractTensorMaps. Promote AbstractTensorMaps
# to ModifiedBinaryOps if necessary.
for op in (:+, :-, :/, :*)
    eval(:(Base.$(op)(x::ModifiedBinaryOp, y::ModifiedBinaryOp)
           = ModifiedBinaryOp(($(op)(xi, yi) for (xi, yi) in zip(x, y))...)))
    eval(:(Base.$(op)(x::AbstractTensorMap, y::ModifiedBinaryOp) = $(op)(ModifiedBinaryOp(x), y)))
    eval(:(Base.$(op)(x::ModifiedBinaryOp, y::AbstractTensorMap) = $(op)(x, ModifiedBinaryOp(y))))
end

Base.:*(op::ModifiedBinaryOp, a::Number) = ModifiedBinaryOp(op.mid * a, op.gap * a)
Base.:*(a::Number, op::ModifiedBinaryOp) = op*a
Base.:/(op::ModifiedBinaryOp, a::Number) = ModifiedBinaryOp(op.mid / a, op.gap / a)

function Base.copyto!(op1::ModifiedBinaryOp, op2::ModifiedBinaryOp)
    copyto!(op1.mid, op2.mid)
    copyto!(op1.gap, op2.gap)
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

function LinearAlgebra.mul!(C::ModifiedBinaryOp, A::ModifiedBinaryOp, B::ModifiedBinaryOp,
                            α, β)
    mul!(C.mid, A.mid, B.mid, α, β)
    mul!(C.gap, A.gap, B.gap, α, β)
    return C
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

function scalingoperator_initialguess(m::ModifiedBinaryMERA, args...)
    # Call the method of the supertype GenericMERA.
    argtypes = (typeof(x) for x in args)
    rho = invoke(scalingoperator_initialguess, Tuple{GenericMERA, argtypes...}, m, args...)
    return ModifiedBinaryOp(rho)
end

# The entropy of the density matrix is the average over the two different density matrices.
function densitymatrix_entropy(rho::ModifiedBinaryOp)
    rho_mid, rho_gap = rho
    S_mid, S_gap = densitymatrix_entropy(rho_mid), densitymatrix_entropy(rho_gap)
    S = (S_mid + S_gap) / 2.0
    return S
end

"""
Return the operator that is the fixed point of the average ascending superoperator of this
layer, normalised to have norm 1.
"""
function ascending_fixedpoint(layer::ModifiedBinaryLayer)
    V = inputspace(layer)
    width = causal_cone_width(typeof(layer))
    Vtotal = ⊗(Iterators.repeated(V, width)...)::ProductSpace{typeof(V), width}
    eye = id(Vtotal) / sqrt(dim(Vtotal))
    return ModifiedBinaryOp(sqrt(8.0/5.0) * eye, sqrt(2.0/5.0) * eye)
end

function gradient(layer::ModifiedBinaryLayer, env::ModifiedBinaryLayer;
                  isometrymanifold=:grassmann, metric=:euclidean)
    u, wl, wr = layer
    uenv, wlenv, wrenv = env
    # The environment is the partial derivative. We need to turn that into a tangent vector
    # of the Stiefel manifold point u or w.
    # The factor of two is from the partial_x + i partial_y derivative of the cost function,
    # and how it depends on both v and v^dagger.
    ugrad = Stiefel.project!(2*uenv, u; metric=metric)
    if isometrymanifold === :stiefel
        wlgrad = Stiefel.project!(2*wlenv, wl; metric=metric)
        wrgrad = Stiefel.project!(2*wrenv, wr; metric=metric)
    elseif isometrymanifold === :grassmann
        wlgrad = Grassmann.project!(2*wlenv, wl)
        wrgrad = Grassmann.project!(2*wrenv, wr)
    else
        msg = "Unknown isometrymanifold $(isometrymanifold)"
        throw(ArgumentError(msg))
    end
    return ModifiedBinaryLayer(ugrad, wlgrad, wrgrad)
end

function precondition_tangent(layer::ModifiedBinaryLayer, tan::ModifiedBinaryLayer, rho)
    u, wl, wr = layer
    utan, wltan, wrtan = tan
    @tensor rho_wl_mid[-1; -11] := rho.mid[-1 1; -11 1]
    @tensor rho_wl_gap[-1; -11] := rho.gap[1 -1; 1 -11]
    rho_wl = (rho_wl_mid + rho_wl_gap) / 2.0
    @tensor rho_wr_mid[-1; -11] := rho.mid[1 -1; 1 -11]
    @tensor rho_wr_gap[-1; -11] := rho.gap[-1 1; -11 1]
    rho_wr = (rho_wr_mid + rho_wr_gap) / 2.0
    @tensor(rho_u[-1 -2; -11 -12] :=
            wl'[12; 1 -11] * wr'[22; -12 2] *
            rho.mid[11 21; 12 22] *
            wl[1 -1; 11] * wr[-2 2; 21])
    utan_prec = precondition_tangent(utan, rho_u)
    wltan_prec = precondition_tangent(wltan, rho_wl)
    wrtan_prec = precondition_tangent(wrtan, rho_wr)
    return ModifiedBinaryLayer(utan_prec, wltan_prec, wrtan_prec)
end

# # # Invariants

"""
Check the compatibility of the legs connecting the disentanglers and the isometries.
Return true/false.
"""
function space_invar_intralayer(layer::ModifiedBinaryLayer)
    u, wl, wr = layer
    matching_bonds = [(space(u, 3)', space(wl, 2)),
                      (space(u, 4)', space(wr, 1))]
    allmatch = all([==(pair...) for pair in matching_bonds])
    # Check that the dimensions are such that isometricity can hold.
    for v in layer
        codom, dom = fuse(codomain(v)), fuse(domain(v))
        allmatch = allmatch && infinum(dom, codom) == dom
    end
    return allmatch
end

"""
Check the compatibility of the legs connecting the isometries of the first layer to the
disentanglers of the layer above it. Return true/false.
"""
function space_invar_interlayer(layer::ModifiedBinaryLayer, next_layer::ModifiedBinaryLayer)
    u, wl, wr = layer
    unext, wlnext, wrnext = next_layer
    matching_bonds = [(space(wl, 3)', space(unext, 1)),
                      (space(wl, 3)', space(unext, 2)),
                      (space(wr, 3)', space(unext, 1)),
                      (space(wr, 3)', space(unext, 2))]
    allmatch = all([==(pair...) for pair in matching_bonds])
    return allmatch
end

# # # Ascending and descending superoperators

function ascend_left(op::ModifiedBinaryOp{T}, layer::ModifiedBinaryLayer
                    ) where T <: SquareTensorMap{2}
    u, wl, wr = layer
    op_mid, op_gap = op
    # Cost: 2X^7 + 3X^6 + 1X^5
    @tensor(
            scaled_op[-100 -200; -300 -400] :=
            wl[7 8; -300] * wr[10 1; -400] *
            u[6 5; 8 10] *
            op_gap[3 4; 7 6] *
            u'[2 9; 4 5] *
            wl'[-100; 3 2] * wr'[-200; 9 1]
           )
    return scaled_op
end

function ascend_right(op::ModifiedBinaryOp{T}, layer::ModifiedBinaryLayer
                     ) where T <: SquareTensorMap{2}
    u, wl, wr = layer
    op_mid, op_gap = op
    # Cost: 2X^7 + 3X^6 + 1X^5
    @tensor(
            scaled_op[-100 -200; -300 -400] :=
            wl[1 10; -300] * wr[8 7; -400] *
            u[5 6; 10 8] *
            op_gap[4 3; 6 7] *
            u'[9 2; 5 4] *
            wl'[-100; 1 9] * wr'[-200; 2 3]
           )
    return scaled_op
end

function ascend_mid(op::ModifiedBinaryOp{T}, layer::ModifiedBinaryLayer
                   ) where T <: SquareTensorMap{2}
    u, wl, wr = layer
    op_mid, op_gap = op
    # Cost: 4X^6 + 2X^5
    @tensor(
            scaled_op[-100 -200; -300 -400] :=
            wl[12 23; -300] * wr[21 11; -400] *
            u[3 4; 23 21] *
            op_mid[1 2; 3 4] *
            u'[24 22; 1 2] *
            wl'[-100; 12 24] * wr'[-200; 22 11]
           )
    return scaled_op
end

function ascend_between(op::ModifiedBinaryOp{T}, layer::ModifiedBinaryLayer
                       ) where T <: SquareTensorMap{2}
    u, wl, wr = layer
    op_mid, op_gap = op
    # Cost: 2X^6 + 2X^5
    @tensor(
            scaled_op[-100 -200; -300 -400] :=
            wr[12 23; -300] * wl[21 11; -400] *
            op_mid[24 22; 23 21] *
            wr'[-100; 12 24] * wl'[-200; 22 11]
           )
    return scaled_op
end

"""
Ascend a two-site `op` from the bottom of the given layer to the top.
"""
function ascend(op::ModifiedBinaryOp{T}, layer::ModifiedBinaryLayer
               ) where T <: SquareTensorMap{2}
    l = ascend_left(op, layer)
    r = ascend_right(op, layer)
    m = ascend_mid(op, layer)
    b = ascend_between(op, layer)
    scaled_op_mid = (l+r+m) / 2.0
    scaled_op_gap = b / 2.0
    scaled_op = ModifiedBinaryOp(scaled_op_mid, scaled_op_gap)
    return scaled_op
end

function ascend_left(op::ModifiedBinaryOp{T}, layer::ModifiedBinaryLayer
                    ) where {S1, T <: AbstractTensorMap{S1,2,3}}
    u, wl, wr = layer
    op_mid, op_gap = op
    # Cost: 2X^7 + 3X^6 + 1X^5
    @tensor(
            scaled_op[-100 -200; -300 -400 -1000] :=
            wl[7 8; -300] * wr[10 1; -400] *
            u[6 5; 8 10] *
            op_gap[3 4; 7 6 -1000] *
            u'[2 9; 4 5] *
            wl'[-100; 3 2] * wr'[-200; 9 1]
           )
    return scaled_op
end

function ascend_right(op::ModifiedBinaryOp{T}, layer::ModifiedBinaryLayer
                     ) where {S1, T <: AbstractTensorMap{S1,2,3}}
    u, wl, wr = layer
    op_mid, op_gap = op
    # Cost: 2X^7 + 3X^6 + 1X^5
    @tensor(
            scaled_op[-100 -200; -300 -400 -1000] :=
            wl[1 10; -300] * wr[8 7; -400] *
            u[5 6; 10 8] *
            op_gap[4 3; 6 7 -1000] *
            u'[9 2; 5 4] *
            wl'[-100; 1 9] * wr'[-200; 2 3]
           )
    return scaled_op
end

function ascend_mid(op::ModifiedBinaryOp{T}, layer::ModifiedBinaryLayer
                   ) where {S1, T <: AbstractTensorMap{S1,2,3}}
    u, wl, wr = layer
    op_mid, op_gap = op
    # Cost: 4X^6 + 2X^5
    @tensor(
            scaled_op[-100 -200; -300 -400 -1000] :=
            wl[12 23; -300] * wr[21 11; -400] *
            u[3 4; 23 21] *
            op_mid[1 2; 3 4 -1000] *
            u'[24 22; 1 2] *
            wl'[-100; 12 24] * wr'[-200; 22 11]
           )
    return scaled_op
end

function ascend_between(op::ModifiedBinaryOp{T}, layer::ModifiedBinaryLayer
                       ) where {S1, T <: AbstractTensorMap{S1,2,3}}
    u, wl, wr = layer
    op_mid, op_gap = op
    # Cost: 2X^6 + 2X^5
    @tensor(
            scaled_op[-100 -200; -300 -400 -1000] :=
            wr[12 23; -300] * wl[21 11; -400] *
            op_mid[24 22; 23 21 -1000] *
            wr'[-100; 12 24] * wl'[-200; 22 11]
           )
    return scaled_op
end

"""
Ascend a two-site `op` from the bottom of the given layer to the top.
"""
function ascend(op::ModifiedBinaryOp{T}, layer::ModifiedBinaryLayer
               ) where {S1, T <: AbstractTensorMap{S1,2,3}}
    u, wl, wr = layer
    op_mid, op_gap = op
    l = ascend_left(op, layer)
    r = ascend_right(op, layer)
    m = ascend_mid(op, layer)
    b = ascend_between(op, layer)
    scaled_op_mid = (l+r+m) / 2.0
    scaled_op_gap = b / 2.0
    scaled_op = ModifiedBinaryOp(scaled_op_mid, scaled_op_gap)
    return scaled_op
end

function ascend(op::ModifiedBinaryOp{T}, layer::ModifiedBinaryLayer
               ) where {T <: SquareTensorMap{1}}
    op = expand_support(op, causal_cone_width(ModifiedBinaryLayer))
    return ascend(op, layer)
end

function ascend(op::T, layer::ModifiedBinaryLayer
               ) where {S1, T <: Union{SquareTensorMap{1},
                                       SquareTensorMap{2},
                                       AbstractTensorMap{S1,2,3}}}
    op = ModifiedBinaryOp(op)
    return ascend(op, layer)
end

function descend_left(rho::ModifiedBinaryOp, layer::ModifiedBinaryLayer)
    u, wl, wr = layer
    rho_mid, rho_gap = rho
    # Cost: 2X^7 + 3X^6 + 1X^5
    @tensor(
            scaled_rho[-100 -200; -300 -400] :=
            u'[5 6; -400 7] *
            wl'[4; -300 5] * wr'[2; 6 1] *
            rho_mid[9 3; 4 2] *
            wl[-100 10; 9] * wr[8 1; 3] *
            u[-200 7; 10 8]
           )
    return scaled_rho
end

function descend_right(rho::ModifiedBinaryOp, layer::ModifiedBinaryLayer)
    u, wl, wr = layer
    rho_mid, rho_gap = rho
    # Cost: 2X^7 + 3X^6 + 1X^5
    @tensor(
            scaled_rho[-100 -200; -300 -400] :=
            u'[6 5; 7 -300] *
            wl'[2; 1 6] * wr'[4; 5 -400] *
            rho_mid[3 9; 2 4] *
            wl[1 8; 3] * wr[10 -200; 9] *
            u[7 -100; 8 10]
           )
    return scaled_rho
end

function descend_mid(rho::ModifiedBinaryOp, layer::ModifiedBinaryLayer)
    u, wl, wr = layer
    rho_mid, rho_gap = rho
    # Cost: 4X^6 + 2X^5
    @tensor(
            scaled_rho[-100 -200; -300 -400] :=
            u'[9 10; -300 -400] *
            wl'[5; 4 9] * wr'[2; 10 1] *
            rho_mid[6 3; 5 2] *
            wl[4 7; 6] * wr[8 1; 3] *
            u[-100 -200; 7 8]
           )
    return scaled_rho
end

function descend_between(rho::ModifiedBinaryOp, layer::ModifiedBinaryLayer)
    u, wl, wr = layer
    rho_mid, rho_gap = rho
    # Cost: 2X^6 + 2X^5
    @tensor(
            scaled_rho[-100 -200; -300 -400] :=
            wr'[5; 4 -300] * wl'[2; -400 1] *
            rho_gap[6 3; 5 2] *
            wr[4 -100; 6] * wl[-200 1; 3]
           )
    return scaled_rho
end

"""
Decend a two-site `rho` from the top of the given layer to the bottom.
"""
function descend(rho::ModifiedBinaryOp, layer::ModifiedBinaryLayer)
    u, wl, wr = layer
    rho_mid, rho_gap = rho
    l = descend_left(rho, layer)
    r = descend_right(rho, layer)
    m = descend_mid(rho, layer)
    b = descend_between(rho, layer)
    scaled_rho_mid = (m + b) / 2.0
    scaled_rho_gap = (l + r) / 2.0
    scaled_rho = ModifiedBinaryOp(scaled_rho_mid, scaled_rho_gap)
    return scaled_rho
end

function descend(op::AbstractTensorMap, layer::ModifiedBinaryLayer)
    return descend(ModifiedBinaryOp(op), layer)
end

# # # Optimization

"""
Compute the environments of all the tensors in the layer, and return them as a Layer.
"""
function environment(layer::ModifiedBinaryLayer, op, rho; vary_disentanglers=true)
    if vary_disentanglers
        env_u = environment_disentangler(op, layer, rho)
    else
        # The adjoint is just for type stability.
        env_u = zero(layer.disentangler')'
    end
    env_wl = environment_isometry_left(op, layer, rho)
    env_wr = environment_isometry_right(op, layer, rho)
    return ModifiedBinaryLayer(env_u, env_wl, env_wr)
end

"""
Return a new layer that minimizes the expectation value with respect to the environment
`env`.
"""
function minimize_expectation_ev(layer::ModifiedBinaryLayer, env::ModifiedBinaryLayer, pars;
                                 vary_disentanglers=true)
    u = (vary_disentanglers ?  projectisometric(env.disentangler; alg=Polar())
         : layer.disentangler)
    wl = projectisometric(env.isometry_left; alg=Polar())
    wr = projectisometric(env.isometry_right; alg=Polar())
    return ModifiedBinaryLayer(u, wl, wr)
end

"""
Return the environment for a disentangler.
"""
function environment_disentangler(h::ModifiedBinaryOp, layer::ModifiedBinaryLayer,
                                  rho::ModifiedBinaryOp)
    u, wl, wr = layer
    h_mid, h_gap = h
    rho_mid, rho_gap = rho
    # Cost: 2X^7 + 3X^6 + 1X^5
    @tensor(
            envl[-100 -200; -300 -400] :=
            rho_mid[10 3; 4 2] *
            wl[9 -300; 10] * wr[-400 1; 3] *
            h_gap[7 8; 9 -100] *
            u'[5 6; 8 -200] *
            wl'[4; 7 5] * wr'[2; 6 1]
           )

    # Cost: 2X^7 + 3X^6 + 1X^5
    @tensor(
            envr[-100 -200; -300 -400] :=
            rho_mid[3 10; 2 4] *
            wl[1 -300; 3] * wr[-400 9; 10] *
            h_gap[8 7; -200 9] *
            u'[6 5; -100 8] *
            wl'[2; 1 6] * wr'[4; 5 7]
           )

    # Cost: 4X^6 + 2X^5
    @tensor(
            envm[-100 -200; -300 -400] :=
            rho_mid[5 3; 6 4] *
            wl[1 -300; 5] * wr[-400 2; 3] *
            h_mid[9 10; -100 -200] *
            u'[7 8; 9 10] *
            wl'[6; 1 7] * wr'[4; 8 2]
           )

    env = (envl + envr + envm) / 4.0
    # Complex conjugate.
    env = permute(env', (3,4), (1,2))
    return env
end

function environment_disentangler(h::SquareTensorMap{2}, layer::ModifiedBinaryLayer, rho)
    h = ModifiedBinaryOp(expand_support(h, causal_cone_width(ModifiedBinaryLayer)))
    return environment_disentangler(h, layer, rho)
end

function environment_disentangler(h::SquareTensorMap{1}, layer::ModifiedBinaryLayer, rho)
    h = ModifiedBinaryOp(expand_support(h, causal_cone_width(ModifiedBinaryLayer)))
    return environment_disentangler(h, layer, rho)
end

"""
Return the environment for the left isometry.
"""
function environment_isometry_left(h::ModifiedBinaryOp, layer, rho::ModifiedBinaryOp)
    u, wl, wr = layer
    h_mid, h_gap = h
    rho_mid, rho_gap = rho
    # Cost: 2X^7 + 3X^6 + 1X^5
    @tensor(
            envl[-100 -200; -300] :=
            rho_mid[-300 2; 4 3] *
            wr[11 1; 2] *
            u[9 10; -200 11] *
            h_gap[7 8; -100 9] *
            u'[5 6; 8 10] *
            wl'[4; 7 5] * wr'[3; 6 1]
           )

    # Cost: 2X^7 + 3X^6 + 1X^5
    @tensor(
            envr[-100 -200; -300] :=
            rho_mid[-300 9; 10 8] *
            wr[7 6; 9] *
            u[5 4; -200 7] *
            h_gap[3 2; 4 6] *
            u'[11 1; 5 3] *
            wl'[10; -100 11] * wr'[8; 1 2]
           )

    # Cost: 4X^6 + 2X^5
    @tensor(
            envm[-100 -200; -300] :=
            rho_mid[-300 8; 10 9] *
            wr[6 5; 8] *
            u[1 2; -200 6] *
            h_mid[3 4; 1 2] *
            u'[11 7; 3 4] *
            wl'[10; -100 11] * wr'[9; 7 5]
           )

    # Cost: 2X^6 + 2X^5
    @tensor(
            envb[-100 -200; -300] :=
            rho_gap[4 -300; 5 7] *
            wr[1 2; 4] *
            h_mid[3 6; 2 -100] *
            wr'[5; 1 3] * wl'[7; 6 -200]
           )

    env = (envl + envr + envm + envb) / 4.0
    # Complex conjugate.
    env = permute(env', (2,3), (1,))
    return env
end

function environment_isometry_left(h::SquareTensorMap{2}, layer::ModifiedBinaryLayer,
                              rho::ModifiedBinaryOp)
    h = ModifiedBinaryOp(expand_support(h, causal_cone_width(ModifiedBinaryLayer)))
    return environment_isometry_left(h, layer, rho)
end

function environment_isometry_left(h::SquareTensorMap{1}, layer::ModifiedBinaryLayer,
                              rho::ModifiedBinaryOp)
    h = ModifiedBinaryOp(expand_support(h, causal_cone_width(ModifiedBinaryLayer)))
    return environment_isometry_left(h, layer, rho)
end

"""
Return the environment for the right isometry.
"""
function environment_isometry_right(h::ModifiedBinaryOp, layer, rho::ModifiedBinaryOp)
    u, wl, wr = layer
    h_mid, h_gap = h
    rho_mid, rho_gap = rho
    # Cost: 2X^7 + 3X^6 + 1X^5
    @tensor(
            envl[-100 -200; -300] :=
            rho_mid[9 -300; 8 10] *
            wl[6 7; 9] *
            u[4 5; 7 -100] *
            h_gap[2 3; 6 4] *
            u'[1 11; 3 5] *
            wl'[8; 2 1] * wr'[10; 11 -200]
           )

    # Cost: 2X^7 + 3X^6 + 1X^5
    @tensor(
            envr[-100 -200; -300] :=
            rho_mid[2 -300; 3 4] *
            wl[1 11; 2] *
            u[10 9; 11 -100] *
            h_gap[8 7; 9 -200] *
            u'[6 5; 10 8] *
            wl'[3; 1 6] * wr'[4; 5 7]
           )

    # Cost: 4X^6 + 2X^5
    @tensor(
            envm[-100 -200; -300] :=
            rho_mid[8 -300; 9 10] *
            wl[5 6; 8] *
            u[2 1; 6 -100] *
            h_mid[4 3; 2 1] *
            u'[7 11; 4 3] *
            wl'[9; 5 7] * wr'[10; 11 -200]
           )

    # Cost: 2X^6 + 2X^5
    @tensor(
            envb[-100 -200; -300] :=
            rho_gap[-300 4; 7 5] *
            wl[2 1; 4] *
            h_mid[6 3; -200 2] *
            wr'[7; -100 6] * wl'[5; 3 1]
           )

    env = (envl + envr + envm + envb) / 4.0
    # Complex conjugate.
    env = permute(env', (2,3), (1,))
    return env
end

function environment_isometry_right(h::SquareTensorMap{2}, layer::ModifiedBinaryLayer,
                              rho::ModifiedBinaryOp)
    h = ModifiedBinaryOp(expand_support(h, causal_cone_width(ModifiedBinaryLayer)))
    return environment_isometry_right(h, layer, rho)
end

function environment_isometry_right(h::SquareTensorMap{1}, layer::ModifiedBinaryLayer,
                              rho::ModifiedBinaryOp)
    h = ModifiedBinaryOp(expand_support(h, causal_cone_width(ModifiedBinaryLayer)))
    return environment_isometry_right(h, layer, rho)
end

