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
Base.eltype(::Type{ModifiedBinaryLayer}) = TensorMap
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
    u = pad_with_zeros_to(u, 1 => V_new, 2 => V_new, 3 => V_new', 4 => V_new')
    wl = pad_with_zeros_to(wl, 1 => V_new, 2 => V_new)
    wr = pad_with_zeros_to(wr, 1 => V_new, 2 => V_new)
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

ModifiedBinaryOp(op::TensorMap) = ModifiedBinaryOp(op, op)
Base.convert(::Type{ModifiedBinaryOp}, op::TensorMap) = ModifiedBinaryOp(op)

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

# Pass element-wise arithmetic down onto the TensorMaps. Promote TensorMaps to
# ModifiedBinaryOps if necessary.
for op in (:+, :-, :/, :*)
    eval(:(Base.$(op)(x::ModifiedBinaryOp, y::ModifiedBinaryOp)
           = ModifiedBinaryOp(($(op)(xi, yi) for (xi, yi) in zip(x, y))...)))
    eval(:(Base.$(op)(x::TensorMap, y::ModifiedBinaryOp) = $(op)(ModifiedBinaryOp(x), y)))
    eval(:(Base.$(op)(x::ModifiedBinaryOp, y::TensorMap) = $(op)(x, ModifiedBinaryOp(y))))
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


# # # Stiefel manifold functions

function stiefel_gradient(h, rho, layer::ModifiedBinaryLayer, pars; vary_disentanglers=true)
    if vary_disentanglers
        uenv = environment_disentangler(h, layer, rho)
    else
        # TODO We could save some subleading computations by not running the whole machinery
        # when uenv .== 0, but this implementation is much simpler.
        u = layer.disentangler
        uenv = TensorMap(zeros, eltype(layer), codomain(u) ← domain(u))
    end
    wlenv = environment_isometry_left(h, layer, rho)
    wrenv = environment_isometry_right(h, layer, rho)
    u, wl, wr = layer
    # The environment is the partial derivative. We need to turn that into a tangent vector
    # of the Stiefel manifold point u or w.
    if pars[:metric] === :canonical
        projection = stiefel_projection_canonical
    elseif pars[:metric] === :euclidean
        projection = stiefel_projection_euclidean
    end
    ugrad = projection(u, uenv)
    wlgrad = projection(wl, wlenv)
    wrgrad = projection(wr, wrenv)
    return ModifiedBinaryLayer(ugrad, wlgrad, wrgrad)
end

function stiefel_geodesic(l::ModifiedBinaryLayer, ltan::ModifiedBinaryLayer, alpha::Number)
    u, utan = stiefel_geodesic_isometry(l.disentangler, ltan.disentangler, alpha)
    wl, wltan = stiefel_geodesic_isometry(l.isometry_left, ltan.isometry_left, alpha)
    wr, wrtan = stiefel_geodesic_isometry(l.isometry_right, ltan.isometry_right, alpha)
    return ModifiedBinaryLayer(u, wl, wr), ModifiedBinaryLayer(utan, wltan, wrtan)
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
               ) where {S1, T <: TensorMap{S1,2,3}}
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

function ascend(op::TensorMap{S1,2,3}, layer::ModifiedBinaryLayer, pos=:avg) where {S1}
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

function descend(op::TensorMap, layer::ModifiedBinaryLayer, pos=:avg)
    return descend(ModifiedBinaryOp(op), layer, pos)
end

# # # Optimization

"""
Loop over the tensors of the layer, optimizing each one in turn to minimize the expecation
value of `h`. `rho` is the density matrix right above this layer.

Three parameters are expected to be in the dictionary `pars`:
    :layer_iters, for how many times to loop over the tensors within a layer,
    :disentangler_iters, for how many times to loop over the disentangler,
    :isometry_iters, for how many times to loop over the isometry.
"""
function minimize_expectation_ev(h, layer::ModifiedBinaryLayer, rho, pars;
                                 vary_disentanglers=true)
    gradnorm_u, gradnorm_wl, gradnorm_wr = 0.0, 0.0, 0.0
    for i in 1:pars[:layer_iters]
        if vary_disentanglers
            for j in 1:pars[:disentangler_iters]
                layer, gradnorm_u = minimize_expectation_ev_disentangler(h, layer, rho)
            end
        end
        for j in 1:pars[:isometry_iters]
            layer, gradnorm_wl = minimize_expectation_ev_isometry_left(h, layer, rho)
            layer, gradnorm_wr = minimize_expectation_ev_isometry_right(h, layer, rho)
        end
    end
    gradnorm = sqrt(gradnorm_u^2 + gradnorm_wl^2 + gradnorm_wr^2)
    return layer, gradnorm
end

"""
Return a new layer, where the disentangler has been changed to the locally optimal one to
minimize the expectation of a threesite operator `h`.
"""
function minimize_expectation_ev_disentangler(h, layer::ModifiedBinaryLayer, rho)
    uold, wlold, wrold = layer
    env = environment_disentangler(h, layer, rho)
    U, S, Vt = tsvd(env, (1,2), (3,4))
    u = U * Vt
    # Compute the Stiefel manifold norm of the gradient. Used as a convergence measure.
    uoldenv = uold' * env
    @tensor crossterm[] := uoldenv[1 2; 3 4] * uoldenv[3 4; 1 2]
    gradnorm = sqrt(abs(norm(env)^2 - real(TensorKit.scalar(crossterm))))
    return ModifiedBinaryLayer(u, wlold, wrold), gradnorm
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
Return a new layer, where the left isometry has been changed to the locally optimal one to
minimize the expectation of a threesite operator `h`.
"""
function minimize_expectation_ev_isometry_left(h, layer::ModifiedBinaryLayer, rho)
    uold, wlold, wrold = layer
    env = environment_isometry_left(h, layer, rho)
    U, S, Vt = tsvd(env, (1,2), (3,))
    wl = U * Vt
    # Compute the Stiefel manifold norm of the gradient. Used as a convergence measure.
    wloldenv = wlold' * env
    @tensor crossterm[] := wloldenv[1; 2] * wloldenv[2; 1]
    gradnorm = sqrt(abs(norm(env)^2 - real(TensorKit.scalar(crossterm))))
    return ModifiedBinaryLayer(uold, wl, wrold), gradnorm
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
Return a new layer, where the right isometry has been changed to the locally optimal one to
minimize the expectation of a threesite operator `h`.
"""
function minimize_expectation_ev_isometry_right(h, layer::ModifiedBinaryLayer, rho)
    uold, wlold, wrold = layer
    env = environment_isometry_right(h, layer, rho)
    U, S, Vt = tsvd(env, (1,2), (3,))
    wr = U * Vt
    # Compute the Stiefel manifold norm of the gradient. Used as a convergence measure.
    wroldenv = wrold' * env
    @tensor crossterm[] := wroldenv[1; 2] * wroldenv[2; 1]
    gradnorm = sqrt(abs(norm(env)^2 - real(TensorKit.scalar(crossterm))))
    return ModifiedBinaryLayer(uold, wlold, wr), gradnorm
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

