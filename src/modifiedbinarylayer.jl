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

struct ModifiedBinaryLayer <: SimpleLayer
    disentangler
    isometry_left
    isometry_right
end

ModifiedBinaryMERA = GenericMERA{ModifiedBinaryLayer}

# Implement the iteration and indexing interfaces. Allows things like `u, wl, wr = layer`.
Base.iterate(layer::ModifiedBinaryLayer) = (layer.disentangler, 1)
function Base.iterate(layer::ModifiedBinaryLayer, state)
    state == 1 && return (layer.isometry_left, 2)
    state == 2 && return (layer.isometry_right, 3)
    return nothing
end
Base.length(layer::ModifiedBinaryLayer) = 3
Base.firstindex(layer::ModifiedBinaryLayer) = 1
Base.lastindex(layer::ModifiedBinaryLayer) = 3
function Base.getindex(layer::ModifiedBinaryLayer, i)
    i == 1 && return layer.disentangler
    i == 2 && return layer.isometry_left
    i == 3 && return layer.isometry_right
    throw(BoundsError(layer, i))
end

"""
The ratio by which the number of sites changes when go down through this layer.
"""
scalefactor(::Type{ModifiedBinaryMERA}) = 2

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

causal_cone_width(::Type{ModifiedBinaryLayer}) = 2

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
function randomlayer(::Type{ModifiedBinaryLayer}, Vin, Vout, Vint=Vout;
                     random_disentangler=false, T=ComplexF64)
    wl = randomisometry(Vout ⊗ Vint, Vin, T)
    # We make the initial guess be reflection symmetric, since that's often true of the
    # desired MERA too (at least if random_disentangler is false, but we do it every time
    # any way).
    wr = deepcopy(permute(wl, (2,1), (3,)))
    u = initialize_disentangler(Vout, Vint, random_disentangler, T)
    return ModifiedBinaryLayer(u, wl, wr)
end

# # # ModifiedBinaryOp
# Due to the structure of the ModifiedBinaryMERA, there's an alternating pattern of two-site
# translation invariance at each layer. Because of this, every operator at a layer is
# defined by two tensors, one for each possible position. We create a `struct` for
# encapsulating this, so that all the necessary operations like trace and matrix product can
# be defined for such operators. We call the two positions `mid` and `gap`, where `mid`
# refers to the a position that has disentangler directly below it, and `gap` to the
# position where the disentangler below is missing.

struct ModifiedBinaryOp{T}
    mid::T
    gap::T
end

ModifiedBinaryOp(op::AbstractTensorMap) = ModifiedBinaryOp(op, op)
Base.convert(::Type{ModifiedBinaryOp}, op::AbstractTensorMap) = ModifiedBinaryOp(op)

Base.iterate(op::ModifiedBinaryOp) = (op.mid, 1)
Base.iterate(op::ModifiedBinaryOp, state) = state == 1 ? (op.gap, 2) : nothing
Base.length(op::ModifiedBinaryOp) = 2
Base.firstindex(op::ModifiedBinaryOp) = 1
Base.lastindex(op::ModifiedBinaryOp) = 2

function Base.getindex(op::ModifiedBinaryOp, i)
    i == 1 && return op.mid
    i == 2 && return op.gap
    throw(BoundsError(op, i))
end

Base.eltype(op::ModifiedBinaryOp) = reduce(promote_type, map(eltype, op))
Base.copy(op::ModifiedBinaryOp) = ModifiedBinaryOp(map(deepcopy, op)...)
Base.adjoint(op::ModifiedBinaryOp) = ModifiedBinaryOp(map(adjoint, op)...)
Base.imag(op::ModifiedBinaryOp) = ModifiedBinaryOp(map(imag, op)...)
Base.real(op::ModifiedBinaryOp) = ModifiedBinaryOp(map(real, op)...)

function pad_with_zeros_to(op::ModifiedBinaryOp, args...)
    return ModifiedBinaryOp(map(x -> pad_with_zeros_to(x, args...), op)...)
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

function Base.similar(op::ModifiedBinaryOp, element_type=eltype(op))
    mid = similar(op.mid, element_type)
    gap = similar(op.gap, element_type)
    return ModifiedBinaryOp(mid, gap)
end

TensorKit.space(op::ModifiedBinaryOp) = space(op.gap)
TensorKit.domain(op::ModifiedBinaryOp) = domain(op.gap)
TensorKit.codomain(op::ModifiedBinaryOp) = codomain(op.gap)

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

function BLAS.axpy!(a::Number, X::ModifiedBinaryOp, Y::ModifiedBinaryOp)
    axpy!(a, X.mid, Y.mid)
    axpy!(a, X.gap, Y.gap)
    return Y
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

LinearAlgebra.tr(op::ModifiedBinaryOp) = (tr(op.mid) + tr(op.gap)) / 2.0

# The initial guesses for finding the scale invariant density matrix and scaling operators
# need to be wrapped in ModifiedBinaryOp.
function thermal_densitymatrix(m::ModifiedBinaryMERA, args...)
    # Call the method of the supertype GenericMERA.
    argtypes = map(typeof, args)
    rho = invoke(thermal_densitymatrix, Tuple{GenericMERA, argtypes...}, m, args...)
    return ModifiedBinaryOp(rho)
end

function scalingoperator_initialguess(m::ModifiedBinaryMERA, args...)
    # Call the method of the supertype GenericMERA.
    argtypes = map(typeof, args)
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
    Vtotal = reduce(⊗, repeat([V], width))
    eye = id(Vtotal) / sqrt(dim(Vtotal))
    return ModifiedBinaryOp(sqrt(8.0/5.0) * eye, sqrt(2.0/5.0) * eye)
end

function gradient(layer::ModifiedBinaryLayer, env::ModifiedBinaryLayer; isometrymanifold=:grassmann,
                  metric=:euclidean)
    u, wl, wr = layer
    uenv, wlenv, wrenv = env
    # The environment is the partial derivative. We need to turn that into a tangent vector
    # of the Stiefel manifold point u or w.
    # TODO Where exactly does this factor of 2 come from again? The conjugate part?
    ugrad = Stiefel.project!(2*uenv, u; metric=metric)
    wlgrad = manifoldmodule(isometrymanifold).project!(2*wlenv, wl; metric=metric)
    wrgrad = manifoldmodule(isometrymanifold).project!(2*wrenv, wr; metric=metric)
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
        allmatch = allmatch && min(dom, codom) == dom
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

"""
Ascend a threesite `op` from the bottom of the given layer to the top.
"""
function ascend(op::ModifiedBinaryOp{T}, layer::ModifiedBinaryLayer, pos=:avg
               ) where T <: SquareTensorMap{2}
    u, wl, wr = layer
    op_mid, op_gap = op
    if in(pos, (:left, :l, :L))
        # Cost: 2X^7 + 3X^6 + 1X^5
        @tensor(
                scaled_op[-100 -200; -300 -400] :=
                wl[7 8; -300] * wr[10 1; -400] *
                u[6 5; 8 10] *
                op_gap[3 4; 7 6] *
                u'[2 9; 4 5] *
                wl'[-100; 3 2] * wr'[-200; 9 1]
               )
    elseif in(pos, (:right, :r, :R))
        # Cost: 2X^7 + 3X^6 + 1X^5
        @tensor(
                scaled_op[-100 -200; -300 -400] :=
                wl[1 10; -300] * wr[8 7; -400] *
                u[5 6; 10 8] *
                op_gap[4 3; 6 7] *
                u'[9 2; 5 4] *
                wl'[-100; 1 9] * wr'[-200; 2 3]
               )
    elseif in(pos, (:middle, :m, :M))
        # Cost: 4X^6 + 2X^5
        @tensor(
                scaled_op[-100 -200; -300 -400] :=
                wl[12 23; -300] * wr[21 11; -400] *
                u[3 4; 23 21] *
                op_mid[1 2; 3 4] *
                u'[24 22; 1 2] *
                wl'[-100; 12 24] * wr'[-200; 22 11]
               )
    elseif in(pos, (:between, :b, :B))
        # Cost: 2X^6 + 2X^5
        @tensor(
                scaled_op[-100 -200; -300 -400] :=
                wr[12 23; -300] * wl[21 11; -400] *
                op_mid[24 22; 23 21] *
                wr'[-100; 12 24] * wl'[-200; 22 11]
               )
    elseif in(pos, (:a, :avg, :average))
        l = ascend(op, layer, :left)
        r = ascend(op, layer, :right)
        m = ascend(op, layer, :middle)
        b = ascend(op, layer, :between)
        scaled_op_mid = (l+r+m) / 2.0
        scaled_op_gap = b / 2.0
        scaled_op = ModifiedBinaryOp(scaled_op_mid, scaled_op_gap)
    else
        throw(ArgumentError("Unknown position (should be :l, :r, :m, :b, or :avg)."))
    end
    return scaled_op
end


"""
Ascend a threesite `op` from the bottom of the given layer to the top.
"""
function ascend(op::ModifiedBinaryOp{T}, layer::ModifiedBinaryLayer, pos=:avg
               ) where {S1, T <: AbstractTensorMap{S1,2,3}}
    u, wl, wr = layer
    op_mid, op_gap = op
    if in(pos, (:left, :l, :L))
        # Cost: 2X^7 + 3X^6 + 1X^5
        @tensor(
                scaled_op[-100 -200; -300 -400 -1000] :=
                wl[7 8; -300] * wr[10 1; -400] *
                u[6 5; 8 10] *
                op_gap[3 4; 7 6 -1000] *
                u'[2 9; 4 5] *
                wl'[-100; 3 2] * wr'[-200; 9 1]
               )
    elseif in(pos, (:right, :r, :R))
        # Cost: 2X^7 + 3X^6 + 1X^5
        @tensor(
                scaled_op[-100 -200; -300 -400 -1000] :=
                wl[1 10; -300] * wr[8 7; -400] *
                u[5 6; 10 8] *
                op_gap[4 3; 6 7 -1000] *
                u'[9 2; 5 4] *
                wl'[-100; 1 9] * wr'[-200; 2 3]
               )
    elseif in(pos, (:middle, :m, :M))
        # Cost: 4X^6 + 2X^5
        @tensor(
                scaled_op[-100 -200; -300 -400 -1000] :=
                wl[12 23; -300] * wr[21 11; -400] *
                u[3 4; 23 21] *
                op_mid[1 2; 3 4 -1000] *
                u'[24 22; 1 2] *
                wl'[-100; 12 24] * wr'[-200; 22 11]
               )
    elseif in(pos, (:between, :b, :B))
        # Cost: 2X^6 + 2X^5
        @tensor(
                scaled_op[-100 -200; -300 -400 -1000] :=
                wr[12 23; -300] * wl[21 11; -400] *
                op_mid[24 22; 23 21 -1000] *
                wr'[-100; 12 24] * wl'[-200; 22 11]
               )
    elseif in(pos, (:a, :avg, :average))
        l = ascend(op, layer, :left)
        r = ascend(op, layer, :right)
        m = ascend(op, layer, :middle)
        b = ascend(op, layer, :between)
        scaled_op_mid = (l+r+m) / 2.0
        scaled_op_gap = b / 2.0
        scaled_op = ModifiedBinaryOp(scaled_op_mid, scaled_op_gap)
    else
        throw(ArgumentError("Unknown position (should be :l, :r, :m, :b, or :avg)."))
    end
    return scaled_op
end

function ascend(op::SquareTensorMap{2}, layer::ModifiedBinaryLayer, pos=:avg)
    op = ModifiedBinaryOp(op)
    return ascend(op, layer, pos)
end

function ascend(op::SquareTensorMap{1}, layer::ModifiedBinaryLayer, pos=:avg)
    op = ModifiedBinaryOp(expand_support(op, causal_cone_width(ModifiedBinaryLayer)))
    return ascend(op, layer, pos)
end

function ascend(op::AbstractTensorMap{S1,2,3}, layer::ModifiedBinaryLayer, pos=:avg
               ) where {S1}
    op = ModifiedBinaryOp(op)
    return ascend(op, layer, pos)
end

"""
Decend a threesite `rho` from the top of the given layer to the bottom.
"""
function descend(rho::ModifiedBinaryOp, layer::ModifiedBinaryLayer, pos=:avg)
    u, wl, wr = layer
    rho_mid, rho_gap = rho
    if in(pos, (:left, :l, :L))
        # Cost: 2X^7 + 3X^6 + 1X^5
        @tensor(
                scaled_rho[-100 -200; -300 -400] :=
                u'[5 6; -400 7] *
                wl'[4; -300 5] * wr'[2; 6 1] *
                rho_mid[9 3; 4 2] *
                wl[-100 10; 9] * wr[8 1; 3] *
                u[-200 7; 10 8]
               )
    elseif in(pos, (:right, :r, :R))
        # Cost: 2X^7 + 3X^6 + 1X^5
        @tensor(
                scaled_rho[-100 -200; -300 -400] :=
                u'[6 5; 7 -300] *
                wl'[2; 1 6] * wr'[4; 5 -400] *
                rho_mid[3 9; 2 4] *
                wl[1 8; 3] * wr[10 -200; 9] *
                u[7 -100; 8 10]
               )
    elseif in(pos, (:middle, :m, :M))
        # Cost: 4X^6 + 2X^5
        @tensor(
                scaled_rho[-100 -200; -300 -400] :=
                u'[9 10; -300 -400] *
                wl'[5; 4 9] * wr'[2; 10 1] *
                rho_mid[6 3; 5 2] *
                wl[4 7; 6] * wr[8 1; 3] *
                u[-100 -200; 7 8]
               )
    elseif in(pos, (:between, :b, :B))
        # Cost: 2X^6 + 2X^5
        @tensor(
                scaled_rho[-100 -200; -300 -400] :=
                wr'[5; 4 -300] * wl'[2; -400 1] *
                rho_gap[6 3; 5 2] *
                wr[4 -100; 6] * wl[-200 1; 3]
               )
    elseif in(pos, (:a, :avg, :average))
        l = descend(rho, layer, :l)
        r = descend(rho, layer, :r)
        m = descend(rho, layer, :m)
        b = descend(rho, layer, :b)
        scaled_rho_mid = (m + b) / 2.0
        scaled_rho_gap = (l + r) / 2.0
        scaled_rho = ModifiedBinaryOp(scaled_rho_mid, scaled_rho_gap)
    else
        throw(ArgumentError("Unknown position (should be :l, :r, or :avg)."))
    end
    return scaled_rho
end

function descend(op::AbstractTensorMap, layer::ModifiedBinaryLayer, pos=:avg)
    return descend(ModifiedBinaryOp(op), layer, pos)
end

# # # Optimization

"""
Compute the environments of all the tensors in the layer, and return them as a Layer.
"""
function environment(layer::ModifiedBinaryLayer, op, rho; vary_disentanglers=true)
    if vary_disentanglers
        env_u = environment_disentangler(op, layer, rho)
    else
        env_u = zero(layer.disentangler)
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

