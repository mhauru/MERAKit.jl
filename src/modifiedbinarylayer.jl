# ModifiedBinaryLayer and ModifiedBinaryMERA types, and methods thereof.
# To be `included` in MERA.jl.

"""
    ModifiedBinaryLayer{ST, ET, Tan} <: SimpleLayer

The type for layers of a modified binary MERA.

Each layer consists of three tensors, a 2-to-2 disentangler, often called `u`, and two 2-to-1
isometries, often called `wl` and `wr`, for left and right. Their relative locations are
```
|     |
wl   wr
| \\ / |
|  u  |
| / \\ |
```

The type parameters are `ST` for space type, e.g. `ComplexSpace` or `SU2Space`; `ET` for
element type, e.g. `Complex{Float64}`; and `Tan` for whether this layer is a tangent layer
or not.  If `Tan = false`, the layer is question is an actual MERA layer. If `Tan = true` it
consists, instead of the actual tensors, of Stiefel/Grassmann tangent vectors of these
tensors.

Index numbering convention is as follows, where the physical indices are at the bottom:
Disentangler:
```
 3|   4|
 +------+
 |  u   |
 +------+
 1|   2|
```

Isometries:
```
   3|
 +------+
 |  w   |
 +------+
 1|   2|
```
"""
struct ModifiedBinaryLayer{ST, ET, Tan} <: SimpleLayer
    # Even though the types are here marked as any, they are restricted to be specific,
    # concrete types dependent on ST, ET and Tan, both in the constructor and in
    # getproperty.
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

"""
    ModifiedBinaryMERA{N}

A modified binary MERA is a MERA consisting of `ModifiedBinaryLayer`s.
"""
ModifiedBinaryMERA{N} = GenericMERA{N, T, O} where {T <: ModifiedBinaryLayer, O}
#Base.show(io::IO, ::Type{ModifiedBinaryMERA}) = print(io, "ModifiedBinaryMERA")
#function Base.show(io::IO, ::Type{ModifiedBinaryMERA{N}}) where {N}
#    return print(io, "ModifiedBinaryMERA{($N)}")
#end

# Given an instance of a type like ModifiedBinaryLayer{ComplexSpace, Float64, true},
# return the unparametrised type ModifiedBinaryLayer.
layertype(::ModifiedBinaryLayer) = ModifiedBinaryLayer
layertype(::Type{T}) where T <: ModifiedBinaryMERA = ModifiedBinaryLayer

function operatortype(::Type{ModifiedBinaryLayer{ST, ET, false}}
                     ) where {ST, ET}
    return ModifiedBinaryOp{tensormaptype(ST, 2, 2, ET)}
end
operatortype(::Type{ModifiedBinaryLayer{ST, ET, true}}) where {ST, ET} = Nothing

Base.eltype(::Type{ModifiedBinaryLayer{ST, ET, Tan}}) where {ST, ET, Tan} = ET
Base.eltype(l::ModifiedBinaryLayer{ST, ET, Tan}) where {ST, ET, Tan} = ET

# Implement the iteration and indexing interfaces. Allows things like `u, wl, wr = layer`.
_tuple(layer::ModifiedBinaryLayer) =
    (layer.disentangler, layer.isometry_left, layer.isometry_right)

scalefactor(::Type{<:ModifiedBinaryLayer}) = 2
scalefactor(::Type{ModifiedBinaryMERA}) = scalefactor(ModifiedBinaryLayer)

causal_cone_width(::Type{<:ModifiedBinaryLayer}) = 2

outputspace(layer::ModifiedBinaryLayer) = space(layer.disentangler, 1)
inputspace(layer::ModifiedBinaryLayer) = space(layer.isometry_left, 3)'
internalspace(layer::ModifiedBinaryLayer) = space(layer.isometry_right, 1)
internalspace(m::ModifiedBinaryMERA, depth) = internalspace(get_layer(m, depth))

function expand_inputspace(layer::ModifiedBinaryLayer, V_new)
    u, wl, wr = layer
    wl = pad_with_zeros_to(wl, 3 => V_new')
    wr = pad_with_zeros_to(wr, 3 => V_new')
    return ModifiedBinaryLayer(u, wl, wr)
end

function expand_outputspace(layer::ModifiedBinaryLayer, V_new)
    u, wl, wr = layer
    u = pad_with_zeros_to(u, 1 => V_new, 2 => V_new)
    wl = pad_with_zeros_to(wl, 1 => V_new)
    wr = pad_with_zeros_to(wr, 2 => V_new)
    return ModifiedBinaryLayer(u, wl, wr)
end

function expand_internalspace(layer::ModifiedBinaryLayer, V_new)
    u, wl, wr = layer
    u = pad_with_zeros_to(u, 3 => V_new', 4 => V_new')
    wl = pad_with_zeros_to(wl, 2 => V_new)
    wr = pad_with_zeros_to(wr, 1 => V_new)
    return ModifiedBinaryLayer(u, wl, wr)
end

function randomlayer(::Type{ModifiedBinaryLayer}, ::Type{T}, Vin, Vout, Vint=Vout;
                     random_disentangler=false) where {T}
    wl = randomisometry(T, Vout ⊗ Vint, Vin)
    # We make the initial guess be reflection symmetric, since that's often true of the
    # desired MERA too (at least if random_disentangler is false, but we do it every time
    # any way).
    wr = deepcopy(permute(wl, (2,1), (3,)))
    u = initialize_disentangler(T, Vout, Vint, random_disentangler)
    return ModifiedBinaryLayer(u, wl, wr)
end

function ascending_fixedpoint(layer::ModifiedBinaryLayer)
    V = inputspace(layer)
    width = causal_cone_width(typeof(layer))
    Vtotal = ⊗(Iterators.repeated(V, width)...)::ProductSpace{typeof(V), width}
    eye = id(Vtotal) / sqrt(dim(Vtotal))
    return ModifiedBinaryOp(sqrt(8.0/5.0) * eye, sqrt(2.0/5.0) * eye)
end

function gradient(layer::ModifiedBinaryLayer, env::ModifiedBinaryLayer;
                  metric=:euclidean)
    u, wl, wr = layer
    uenv, wlenv, wrenv = env
    # The environment is the partial derivative. We need to turn that into a tangent vector
    # of the Stiefel manifold point u or w.
    # The factor of two is from the partial_x + i partial_y derivative of the cost function,
    # and how it depends on both v and v^dagger.
    ugrad = Stiefel.project!(2*uenv, u; metric=metric)
    wlgrad = Grassmann.project!(2*wlenv, wl)
    wrgrad = Grassmann.project!(2*wrenv, wr)
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

function space_invar_intralayer(layer::ModifiedBinaryLayer)
    u, wl, wr = layer
    matching_bonds = ((space(u, 3)', space(wl, 2)),
                      (space(u, 4)', space(wr, 1)))
    allmatch = all([==(pair...) for pair in matching_bonds])
    # Check that the dimensions are such that isometricity can hold.
    for v in layer
        codom, dom = fuse(codomain(v)), fuse(domain(v))
        allmatch = allmatch && infinum(dom, codom) == dom
    end
    return allmatch
end

function space_invar_interlayer(layer::ModifiedBinaryLayer, next_layer::ModifiedBinaryLayer)
    u, wl, wr = layer
    unext, wlnext, wrnext = next_layer
    matching_bonds = ((space(wl, 3)', space(unext, 1)),
                      (space(wl, 3)', space(unext, 2)),
                      (space(wr, 3)', space(unext, 1)),
                      (space(wr, 3)', space(unext, 2)))
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

function minimize_expectation_ev(layer::ModifiedBinaryLayer, env::ModifiedBinaryLayer;
                                 vary_disentanglers=true)
    u = (vary_disentanglers ?  projectisometric(env.disentangler; alg=Polar())
         : layer.disentangler)
    wl = projectisometric(env.isometry_left; alg=Polar())
    wr = projectisometric(env.isometry_right; alg=Polar())
    return ModifiedBinaryLayer(u, wl, wr)
end

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
