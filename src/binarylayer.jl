# BinaryLayer and BinaryMERA types, and methods thereof.
# To be `included` in MERA.jl.

"""
    BinaryLayer{ST, ET, Tan} <: SimpleLayer

The type for layers of a binary MERA.

Each layer consists of two tensors, a 2-to-2 disentangler, often called `u`, and a 2-to-1
isometry, often called `w`.

The type parameters are `ST` for space type, e.g. `ComplexSpace` or `SU2Space`; `ET` for
element type, e.g. `Complex{Float64}`; and `Tan` for whether this layer is a tangent layer
or not.  If `Tan = false`, the layer in question is an actual MERA layer. If `Tan = true` it
consists, instead of the actual tensors, of Stiefel/Grassmann tangent vectors of these
tensors.

`BinaryLayer` implements the iteration interface, returning first the disentangler, then the
isometry.

Index numbering convention is as follows, where the physical indices are at the bottom:
Disentangler:
```
 3│    4│
 ┌┴─────┴┐
 │   u   │
 └┬─────┬┘
 1│    2│
```

Isometry:
```
    3│
 ┌───┴───┐
 │   w   │
 └┬─────┬┘
 1│    2│
```
"""
struct BinaryLayer{ST, ET, Tan} <: SimpleLayer
    # Even though the types are here marked as any, they are restricted to be specific,
    # concrete types dependent on ST, ET and Tan, both in the constructor and in
    # getproperty.
    disentangler::Any
    isometry::Any

    function BinaryLayer{ST, ET, Tan}(disentangler, isometry) where {ST, ET, Tan}
        DisType = disentangler_type(ST, ET, Tan)
        IsoType = binaryisometry_type(ST, ET, Tan)
        disconv = convert(DisType, disentangler)::DisType
        isoconv = convert(IsoType, isometry)::IsoType
        return new{ST, ET, Tan}(disconv, isoconv)
    end
end

function BinaryLayer(disentangler::DisType, isometry::IsoType) where {
    ST,
    DisType <: AbstractTensorMap{ST, 2, 2},
    IsoType <: AbstractTensorMap{ST, 2, 1}
}
    ET = eltype(DisType)
    @assert eltype(IsoType) === ET
    return BinaryLayer{ST, ET, false}(disentangler, isometry)
end

function BinaryLayer(disentangler::DisTanType, isometry::IsoTanType) where {
    ST,
    DisType <: AbstractTensorMap{ST, 2, 2},
    IsoType <: AbstractTensorMap{ST, 2, 1},
    DisTanType <: Stiefel.StiefelTangent{DisType},
    IsoTanType <: Grassmann.GrassmannTangent{IsoType},
}
    ET = eltype(DisType)
    @assert eltype(IsoType) === ET
    return BinaryLayer{ST, ET, true}(disentangler, isometry)
end

function Base.getproperty(l::BinaryLayer{ST, ET, Tan}, sym::Symbol) where {ST, ET, Tan}
    if sym === :disentangler
        T = disentangler_type(ST, ET, Tan)
    elseif sym === :isometry
        T = binaryisometry_type(ST, ET, Tan)
    else
        T = Any
    end
    return getfield(l, sym)::T
end

"""
    BinaryMERA{N}

A binary MERA is a MERA consisting of `BinaryLayer`s.
"""
BinaryMERA{N} = GenericMERA{N, T, O} where {T <: BinaryLayer, O}
layertype(::Type{BinaryMERA}) = BinaryLayer
#Base.show(io::IO, ::Type{BinaryMERA}) = print(io, "BinaryMERA")
#Base.show(io::IO, ::Type{BinaryMERA{N}}) where {N} = print(io, "BinaryMERA{($N)}")

# Implement the iteration and indexing interfaces. Allows things like `u, w = layer`.
# See simplelayer.jl for details.
_tuple(layer::BinaryLayer) = (layer.disentangler, layer.isometry)

function operatortype(::Type{BinaryLayer{ST, ET, false}}) where {ST, ET}
    return tensormaptype(ST, 3, 3, ET)
end
operatortype(::Type{BinaryLayer{ST, ET, true}}) where {ST, ET} = Nothing

TensorKit.spacetype(::Type{BinaryLayer{ST, ET, Tan}}) where {ST, ET, Tan} = ST

scalefactor(::Type{<:BinaryLayer}) = 2

causal_cone_width(::Type{<:BinaryLayer}) = 3

Base.eltype(::Type{BinaryLayer{ST, ET, Tan}}) where {ST, ET, Tan} = ET

outputspace(layer::BinaryLayer) = space(layer.disentangler, 1)
inputspace(layer::BinaryLayer) = space(layer.isometry, 3)'
internalspace(layer::BinaryLayer) = space(layer.isometry, 1)
internalspace(m::BinaryMERA, depth) = internalspace(getlayer(m, depth))

function expand_inputspace(layer::BinaryLayer, V_new)
    u, w = layer
    w = pad_with_zeros_to(w, 3 => V_new')
    return BinaryLayer(u, w)
end

function expand_outputspace(layer::BinaryLayer, V_new)
    u, w = layer
    u = pad_with_zeros_to(u, 1 => V_new, 2 => V_new)
    return BinaryLayer(u, w)
end

function expand_internalspace(layer::BinaryLayer, V_new)
    u, w = layer
    u = pad_with_zeros_to(u, 3 => V_new', 4 => V_new')
    w = pad_with_zeros_to(w, 1 => V_new, 2 => V_new)
    return BinaryLayer(u, w)
end

function randomlayer(
    ::Type{BinaryLayer}, ::Type{T}, Vin, Vout, Vint = Vout; random_disentangler = false
) where {T}
    w = randomisometry(T, Vint ⊗ Vint, Vin)
    u = initialize_disentangler(T, Vout, Vint, random_disentangler)
    return BinaryLayer(u, w)
end

function ascending_fixedpoint(layer::BinaryLayer)
    width = causal_cone_width(layer)
    Vtotal = ⊗(ntuple(n->inputspace(layer), Val(width))...)
    return id(storagetype(operatortype(layer)), Vtotal)
end

function scalingoperator_initialguess(l::BinaryLayer, irreps...)
    width = causal_cone_width(l)
    V = inputspace(l)
    interlayer_space = ⊗(ntuple(n->V, Val(width))...)
    outspace = interlayer_space
    inspace = interlayer_space
    for irrep in irreps
        inspace = inspace ⊗ spacetype(V)(irrep => 1)
    end
    typ = eltype(l)
    t = TensorMap(randn, typ, outspace ← inspace)
    return t
end

function gradient(layer::BinaryLayer, env::BinaryLayer; metric = :euclidean)
    u, w = layer
    uenv, wenv = env
    # The environment is the partial derivative. We need to turn that into a tangent vector
    # of the Stiefel manifold point u or w.
    # The factor of two is from the partial_x + i partial_y derivative of the cost function,
    # and how it depends on both v and v^dagger.
    ugrad = Stiefel.project!(2*uenv, u; metric = metric)
    wgrad = Grassmann.project!(2*wenv, w)
    return BinaryLayer(ugrad, wgrad)
end

function precondition_tangent(layer::BinaryLayer, tan::BinaryLayer, rho)
    u, w = layer
    utan, wtan = tan
    @planar rho_wl[-1; -11] := rho[-1 1 2; -11 1 2]
    @planar rho_wm[-1; -11] := rho[1 -1 2; 1 -11 2]
    @planar rho_wr[-1; -11] := rho[1 2 -1; 1 2 -11]
    rho_w = (rho_wl + rho_wm + rho_wr) / 3.0
    rho_twosite = (rho_twosite_l + rho_twosite_r) / 2.0
    @planar(
        rho_u[-1 -2; -11 -12] :=
        w[1 -1; 11] * w[-2 2; 21] *
        rho_twosite[11 21; 12 22] *
        w'[12; 1 -11] * w'[22; -12 2]
    )
    utan_prec = precondition_tangent(utan, rho_u)
    wtan_prec = precondition_tangent(wtan, rho_w)
    return BinaryLayer(utan_prec, wtan_prec)
end

# # # Invariants

function space_invar_intralayer(layer::BinaryLayer)
    u, w = layer
    matching_bonds = (
        (space(u, 3)', space(w, 2)),
        (space(u, 4)', space(w, 1))
    )
    allmatch = all(pair -> ==(pair...), matching_bonds)
    # Check that the dimensions are such that isometricity can hold.
    allmatch &= all((u, w)) do v
        codom, dom = fuse(codomain(v)), fuse(domain(v))
        infimum(dom, codom) == dom
    end
    return allmatch
end

function space_invar_interlayer(layer::BinaryLayer, next_layer::BinaryLayer)
    u, w = layer.disentangler, layer.isometry
    unext, wnext = next_layer.disentangler, next_layer.isometry
    matching_bonds = (
        (space(w, 3)', space(unext, 1)),
        (space(w, 3)', space(unext, 2))
    )
    allmatch = all(pair -> ==(pair...), matching_bonds)
    return allmatch
end

# # # Ascending and descending superoperators
const BinaryOperator{S} = AbstractTensorMap{S, 3, 3}
const ChargedBinaryOperator{S} = AbstractTensorMap{S, 3, 4}
const DoubleChargedBinaryOperator{S} = AbstractTensorMap{S, 3, 5}

function ascend(
    op::Union{BinaryOperator, ChargedBinaryOperator, DoubleChargedBinaryOperator},
    layer::BinaryLayer
)
    l = ascend_left(op, layer)
    r = ascend_right(op, layer)
    scaled_op = (l+r)/2
    return scaled_op
end

function ascend(
    op::Union{SquareTensorMap{1}, SquareTensorMap{2}}, layer::BinaryLayer
)
    return ascend(expand_support(op, causal_cone_width(BinaryLayer)), layer)
end

# TODO Think about how to best remove the code duplication of having the separate methods
# for ordinary, charged, and double charged operators. Note also that there's a mix of using
# @tensor and @planar.
function ascend_left(op::ChargedBinaryOperator, layer::BinaryLayer)
    u, w = layer
    @tensor(
        scaled_op[-100 -200 -300; -400 -500 -600 -1000] :=
        w'[-100; 5 7] * w'[-200; 13 11] * w'[-300; 17 15] *
        u'[7 13; 3 4] * u'[11 17; 14 12] *
        op[3 4 14; 1 2 10 -1000] *
        u[1 2; 6 9] * u[10 12; 8 16] *
        w[5 6; -400] * w[9 8; -500] * w[16 15; -600]
    )
    return scaled_op
end

function ascend_right(op::ChargedBinaryOperator, layer::BinaryLayer)
    u, w = layer
    @tensor(
        scaled_op[-100 -200 -300; -400 -500 -600 -1000] :=
        w'[-100; 15 17] * w'[-200; 11 13] * w'[-300; 7 5] *
        u'[17 11; 12 14] * u'[13 7; 3 4] *
        op[14 3 4; 10 1 2 -1000] *
        u[12 10; 16 8] * u[1 2; 9 6] *
        w[15 16; -400] * w[8 9; -500] * w[6 5; -600]
    )
    return scaled_op
end

# TODO Figure out how to deal with the extra charge legs in the case of anyonic tensors.
function ascend_left(op::BinaryOperator, layer::BinaryLayer)
    u, w = layer
    @planar(
        scaled_op[-100 -200 -300; -400 -500 -600] :=
        w'[-100; 5 7] * w'[-200; 13 11] * w'[-300; 17 15] *
        u'[7 13; 3 4] * u'[11 17; 14 12] *
        op[3 4 14; 1 2 10] *
        u[1 2; 6 9] * u[10 12; 8 16] *
        w[5 6; -400] * w[9 8; -500] * w[16 15; -600]
    )
    return scaled_op
end

function ascend_right(op::BinaryOperator, layer::BinaryLayer)
    u, w = layer
    @planar(
        scaled_op[-100 -200 -300; -400 -500 -600] :=
        w'[-100; 15 17] * w'[-200; 11 13] * w'[-300; 7 5] *
        u'[17 11; 12 14] * u'[13 7; 3 4] *
        op[14 3 4; 10 1 2] *
        u[12 10; 16 8] * u[1 2; 9 6] *
        w[15 16; -400] * w[8 9; -500] * w[6 5; -600]
    )
    return scaled_op
end

function ascend_left(op::DoubleChargedBinaryOperator, layer::BinaryLayer)
    u, w = layer
    @planar(
        temp1[-1 -2; -3 -4 -5 -6 -7 -8] :=
        op[-1 -2 -8; 1 2 -5 -6 -7] *
        u[1 2; -3 -4]
    )
    temp2 = braid(temp1, (11, 12, 13, 14, 15, 16, 1, 100), (1, 2), (3, 6, 7, 4, 5, 8))
    @planar(
        temp3[-100 -200 -300; -400 -1000 -2000 -500 -600] :=
        w'[-100; 5 7] * w'[-200; 13 11] * w'[-300; 17 15] *
        u'[7 13; 3 4] * u'[11 17; 14 12] *
        temp2[3 4; 6 -1000 -2000 9 10 14] *
        u[10 12; 8 16] *
        w[5 6; -400] * w[9 8; -500] * w[16 15; -600]
    )
    scaled_op = braid(temp3, (11, 12, 13, 14, 1, 100, 15, 16), (1, 2, 3), (4, 7, 8, 5, 6))
    return scaled_op
end

function ascend_right(op::DoubleChargedBinaryOperator, layer::BinaryLayer)
    u, w = layer
    @planar(
        temp1[-1 -2; -3 -4 -5 -6 -7 -8] :=
        op[-3 -1 -2; -4 1 2 -7 -8] *
        u[1 2; -5 -6]
    )
    temp2 = braid(temp1, (11, 12, 13, 14, 15, 16, 1, 100), (1, 2), (3, 4, 5, 7, 8, 6))
    @planar(
        temp3[-100 -200 -300; -400 -500 -1000 -2000 -600] :=
        w'[-100; 15 17] * w'[-200; 11 13] * w'[-300; 7 5] *
        u'[17 11; 12 14] * u'[13 7; 3 4] *
        temp2[3 4; 14 10 9 -1000 -2000 6] *
        u[12 10; 16 8] *
        w[15 16; -400] * w[8 9; -500] * w[6 5; -600]
    )
    scaled_op = braid(temp3, (11, 12, 13, 14, 15, 1, 100, 16), (1, 2, 3), (4, 5, 8, 6, 7))
    return scaled_op
end

function descend_left(rho::BinaryOperator, layer::BinaryLayer)
    u, w = layer
    @planar(
        scaled_rho[-100 -200 -300; -400 -500 -600] :=
        u[-100 -200; 14 15] * u[-300 11; 3 8] *
        w[1 14; 13] * w[15 3; 7] * w[8 4; 6] *
        rho[13 7 6; 12 9 5] *
        w'[12; 1 16] * w'[9; 17 2] * w'[5; 10 4] *
        u'[16 17; -400 -500] * u'[2 10; -600 11]
    )
    return scaled_rho
end

function descend_right(rho::BinaryOperator, layer::BinaryLayer)
    u, w = layer
    @planar(
        scaled_rho[-100 -200 -300; -400 -500 -600] :=
        u[11 -100; 8 3] * u[-200 -300; 15 14] *
        w[4 8; 6] * w[3 15; 7] * w[14 1; 13] *
        rho[6 7 13; 5 9 12] *
        w'[5; 4 10] * w'[9; 2 17] * w'[12; 16 1] *
        u'[10 2; 11 -400] * u'[17 16; -500 -600]
    )
    return scaled_rho
end

function descend(rho::BinaryOperator, layer::BinaryLayer)
    l = descend_left(rho, layer)
    r = descend_right(rho, layer)
    scaled_rho = (l+r)/2
    return scaled_rho
end

# # # Optimization

function environment(op, layer::BinaryLayer, rho; vary_disentanglers = true)
    if vary_disentanglers
        env_u = environment_disentangler(op, layer, rho)
    else
        # The adjoint is just for type stability.
        env_u = zero(layer.disentangler')'
    end
    env_w = environment_isometry(op, layer, rho)
    return BinaryLayer(env_u, env_w)
end

function minimize_expectation_ev(
    layer::BinaryLayer, env::BinaryLayer; vary_disentanglers = true
)
    u = if vary_disentanglers
        projectisometric(env.disentangler; alg = Polar())
    else
        layer.disentangler
    end
    w = projectisometric(env.isometry; alg = Polar())
    return BinaryLayer(u, w)
end

function environment_disentangler(h::BinaryOperator, layer::BinaryLayer, rho)
    u, w = layer
    @planar(
        env1[-1 -2; -3 -4] :=
        rho[17 18 10; 15 14 9] *
        w'[15; 5 6] * w'[14; 16 -3] * w'[9; -4 8] *
        u'[6 16; 1 2] *
        h[1 2 -1; 3 4 13] *
        u[3 4; 7 12] * u[13 -2; 11 19] *
        w[5 7; 17] * w[12 11; 18] * w[19 8; 10]
    )

    @planar(
        env2[-1 -2; -3 -4] :=
        rho[4 15 6; 3 10 5] *
        w'[3; 1 11] * w'[10; 9 -3] * w'[5; -4 2] *
        u'[11 9; 12 19] *
        h[19 -1 -2; 18 7 8] *
        u[12 18; 13 14] * u[7 8; 16 17] *
        w[1 13; 4] * w[14 16; 15] * w[17 2; 6]
    )

    @planar(
        env3[-1 -2; -3 -4] :=
        rho[6 15 4; 5 10 3] *
        w'[5; 2 -3] * w'[10; -4 9] * w'[3; 11 1] *
        u'[9 11; 19 12] *
        h[-1 -2 19; 8 7 18] *
        u[8 7; 17 16] * u[18 12; 14 13] *
        w[2 17; 6] * w[16 14; 15] * w[13 1; 4]
    )

    @planar(
        env4[-1 -2; -3 -4] :=
        rho[10 18 17; 9 14 15] *
        w'[9; 8 -3] * w'[14; -4 16] * w'[15; 6 5] *
        u'[16 6; 2 1] *
        h[-2 2 1; 13 4 3] *
        u[-1 13; 19 11] * u[4 3; 12 7] *
        w[8 19; 10] * w[11 12; 18] * w[7 5; 17]
    )

    env = (env1 + env2 + env3 + env4)/2
    return env
end

function environment_disentangler(
    h::Union{SquareTensorMap{1}, SquareTensorMap{2}}, layer::BinaryLayer, rho
)
    h = expand_support(h, causal_cone_width(BinaryLayer))
    return environment_disentangler(h, layer, rho)
end

function environment_isometry(h::BinaryOperator, layer, rho)
    u, w = layer
    @planar(
        env1[-1 -2; -3] :=
        rho[16 15 19; 18 17 -3] *
        w'[18; 5 6] * w'[17; 9 8] *
        u'[6 9; 2 1] * u'[8 -1; 10 11] *
        h[2 1 10; 4 3 12] *
        u[4 3; 7 14] * u[12 11; 13 20] *
        w[5 7; 16] * w[14 13; 15] * w[20 -2; 19]
    )

    @planar(
        env2[-1 -2; -3] :=
        rho[18 17 19; 16 15 -3] *
        w'[16; 12 13] * w'[15; 5 6] *
        u'[13 5; 9 7] * u'[6 -1; 2 1] *
        h[7 2 1; 8 4 3] *
        u[9 8; 14 11] * u[4 3; 10 20] *
        w[12 14; 18] * w[11 10; 17] * w[20 -2; 19]
    )

    @planar(
        env3[-1 -2; -3] :=
        rho[19 20 15; 18 -3 14] *
        w'[18; 5 6] * w'[14; 17 13] *
        u'[6 -1; 2 1] * u'[-2 17; 12 11] *
        h[2 1 12; 4 3 9] *
        u[4 3; 7 10] * u[9 11; 8 16] *
        w[5 7; 19] * w[10 8; 20] * w[16 13; 15]
    )

    @planar(
        env4[-1 -2; -3] :=
        rho[15 20 19; 14 -3 18] *
        w'[14; 13 17] * w'[18; 6 5] *
        u'[17 -1; 11 12] * u'[-2 6; 1 2] *
        h[12 1 2; 9 3 4] *
        u[11 9; 16 8] * u[3 4; 10 7] *
        w[13 16; 15] * w[8 10; 20] * w[7 5; 19]
    )

    @planar(
        env5[-1 -2; -3] :=
        rho[19 17 18; -3 15 16] *
        w'[15; 6 5] * w'[16; 13 12] *
        u'[-2 6; 1 2] * u'[5 13; 7 9] *
        h[1 2 7; 3 4 8] *
        u[3 4; 20 10] * u[8 9; 11 14] *
        w[-1 20; 19] * w[10 11; 17] * w[14 12; 18]
    )

    @planar(
        env6[-1 -2; -3] :=
        rho[19 15 16; -3 17 18] *
        w'[17; 8 9] * w'[18; 6 5] *
        u'[-2 8; 11 10] * u'[9 6; 1 2] *
        h[10 1 2; 12 3 4] *
        u[11 12; 20 13] * u[3 4; 14 7] *
        w[-1 20; 19] * w[13 14; 15] * w[7 5; 16]
    )

    env = (env1 + env2 + env3 + env4 + env5 + env6)/2
    return env
end

function environment_isometry(
    h::Union{SquareTensorMap{1}, SquareTensorMap{2}}, layer::BinaryLayer, rho
)
    h = expand_support(h, causal_cone_width(BinaryLayer))
    return environment_isometry(h, layer, rho)
end
