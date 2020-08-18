# BinaryLayer and BinaryMERA types, and methods thereof.
# To be `included` in MERA.jl.

"""
    BinaryLayer{ST, ET, Tan} <: SimpleLayer

The type for layers of a binary MERA.

Each layer consists of two tensors, a 2-to-2 disentangler, often called `u`, and a 2-to-1
isometry, often called `w`.

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

Isometry:
```
   3|
 +------+
 |  w   |
 +------+
 1|   2|
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

function BinaryLayer(disentangler::DisType, isometry::IsoType
                     ) where {ST,
                              DisType <: AbstractTensorMap{ST, 2, 2},
                              IsoType <: AbstractTensorMap{ST, 2, 1}}
    ET = eltype(DisType)
    @assert eltype(IsoType) === ET
    return BinaryLayer{ST, ET, false}(disentangler, isometry)
end

function BinaryLayer(disentangler::DisTanType, isometry::IsoTanType
                     ) where {ST,
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
_tuple(layer::BinaryLayer) = (layer.disentangler, layer.isometry)

function operatortype(::Type{BinaryLayer{ST, ET, false}}) where {ST, ET}
    return tensormaptype(ST, 3, 3, ET)
end
operatortype(::Type{BinaryLayer{ST, ET, true}}) where {ST, ET} = Nothing

scalefactor(::Type{<:BinaryLayer}) = 2

causal_cone_width(::Type{<:BinaryLayer}) = 3

Base.eltype(::Type{BinaryLayer{ST, ET, Tan}}) where {ST, ET, Tan} = ET


outputspace(layer::BinaryLayer) = space(layer.disentangler, 1)
inputspace(layer::BinaryLayer) = space(layer.isometry, 3)'
internalspace(layer::BinaryLayer) = space(layer.isometry, 1)
internalspace(m::BinaryMERA, depth) = internalspace(get_layer(m, depth))

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

function randomlayer(::Type{BinaryLayer}, T, Vin, Vout, Vint=Vout;
                     random_disentangler=false)
    w = randomisometry(T, Vint ⊗ Vint, Vin)
    u = initialize_disentangler(T, Vout, Vint, random_disentangler)
    return BinaryLayer(u, w)
end

function ascending_fixedpoint(layer::BinaryLayer)
    V = inputspace(layer)
    width = causal_cone_width(typeof(layer))
    Vtotal = ⊗(Iterators.repeated(V, width)...)::ProductSpace{typeof(V), width}
    eye = id(Vtotal) / sqrt(dim(Vtotal))
    return eye
end

function gradient(layer::BinaryLayer, env::BinaryLayer; metric=:euclidean)
    u, w = layer
    uenv, wenv = env
    # The environment is the partial derivative. We need to turn that into a tangent vector
    # of the Stiefel manifold point u or w.
    # The factor of two is from the partial_x + i partial_y derivative of the cost function,
    # and how it depends on both v and v^dagger.
    ugrad = Stiefel.project!(2*uenv, u; metric=metric)
    wgrad = Grassmann.project!(2*wenv, w)
    return BinaryLayer(ugrad, wgrad)
end

function precondition_tangent(layer::BinaryLayer, tan::BinaryLayer, rho)
    u, w = layer
    utan, wtan = tan
    @tensor rho_wl[-1; -11] := rho[-1 1 2; -11 1 2]
    @tensor rho_wm[-1; -11] := rho[1 -1 2; 1 -11 2]
    @tensor rho_wr[-1; -11] := rho[1 2 -1; 1 2 -11]
    rho_w = (rho_wl + rho_wm + rho_wr) / 3.0
    @tensor rho_twosite_l[-1 -2; -11 -12] := rho[-1 -2 1; -11 -12 1]
    @tensor rho_twosite_r[-1 -2; -11 -12] := rho[-1 -2 1; -11 -12 1]
    rho_twosite = (rho_twosite_l + rho_twosite_r) / 2.0
    @tensor(rho_u[-1 -2; -11 -12] :=
            w'[12; 1 -11] * w'[22; -12 2] *
            rho_twosite[11 21; 12 22] *
            w[1 -1; 11] * w[-2 2; 21])
    utan_prec = precondition_tangent(utan, rho_u)
    wtan_prec = precondition_tangent(wtan, rho_w)
    return BinaryLayer(utan_prec, wtan_prec)
end

# # # Invariants

function space_invar_intralayer(layer::BinaryLayer)
    u, w = layer
    matching_bonds = ((space(u, 3)', space(w, 2)),
                      (space(u, 4)', space(w, 1)))
    allmatch = all(pair->==(pair...), matching_bonds)
    # Check that the dimensions are such that isometricity can hold.
    allmatch &= all((u, w)) do v
        codom, dom = fuse(codomain(v)), fuse(domain(v))
        infinum(dom, codom) == dom
    end
    return allmatch
end

function space_invar_interlayer(layer::BinaryLayer, next_layer::BinaryLayer)
    u, w = layer.disentangler, layer.isometry
    unext, wnext = next_layer.disentangler, next_layer.isometry
    matching_bonds = ((space(w, 3)', space(unext, 1)),
                      (space(w, 3)', space(unext, 2)))
    allmatch = all(pair->==(pair...), matching_bonds)
    return allmatch
end

# # # Ascending and descending superoperators
const BinaryOperator{S} = AbstractTensorMap{S,3,3}
const ChargedBinaryOperator{S} = AbstractTensorMap{S,3,4}

function ascend_left(op::ChargedBinaryOperator, layer::BinaryLayer)
    u, w = layer
    @tensor(
            scaled_op[-100 -200 -300; -400 -500 -600 -1000] :=
            w[5 6; -400] * w[9 8; -500] * w[16 15; -600] *
            u[1 2; 6 9] * u[10 12; 8 16] *
            op[3 4 14; 1 2 10 -1000] *
            u'[7 13; 3 4] * u'[11 17; 14 12] *
            w'[-100; 5 7] * w'[-200; 13 11] * w'[-300; 17 15]
           )
    return scaled_op
end

function ascend_right(op::ChargedBinaryOperator, layer::BinaryLayer)
    u, w = layer
    @tensor(
            scaled_op[-100 -200 -300; -400 -500 -600 -1000] :=
            w[15 16; -400] * w[8 9; -500] * w[6 5; -600] *
            u[12 10; 16 8] * u[1 2; 9 6] *
            op[14 3 4; 10 1 2 -1000] *
            u'[17 11; 12 14] * u'[13 7; 3 4] *
            w'[-100; 15 17] * w'[-200; 11 13] * w'[-300; 7 5]
           )
    return scaled_op
end

function ascend(op::ChargedBinaryOperator, layer::BinaryLayer)
    l = ascend_left(op, layer)
    r = ascend_right(op, layer)
    scaled_op = (l+r)/2
    return scaled_op
end

ascend_left(op::BinaryOperator, layer::BinaryLayer) =
    remove_dummy_index(ascend_left(append_dummy_index(op), layer))

ascend_right(op::BinaryOperator, layer::BinaryLayer) =
    remove_dummy_index(ascend_right(append_dummy_index(op), layer))

ascend(op::BinaryOperator, layer::BinaryLayer) =
    remove_dummy_index(ascend(append_dummy_index(op), layer))

ascend(op::Union{SquareTensorMap{1}, SquareTensorMap{2}}, layer::BinaryLayer) =
    ascend(expand_support(op, causal_cone_width(BinaryLayer)), layer)

function descend_left(rho::BinaryOperator, layer::BinaryLayer)
    u, w = layer
    @tensor(
            scaled_rho[-100 -200 -300; -400 -500 -600] :=
            u'[16 17; -400 -500] * u'[2 10; -600 11] *
            w'[12; 1 16] * w'[9; 17 2] * w'[5; 10 4] *
            rho[13 7 6; 12 9 5] *
            w[1 14; 13] * w[15 3; 7] * w[8 4; 6] *
            u[-100 -200; 14 15] * u[-300 11; 3 8]
           )
    return scaled_rho
end

function descend_right(rho::BinaryOperator, layer::BinaryLayer)
    u, w = layer
    @tensor(
            scaled_rho[-100 -200 -300; -400 -500 -600] :=
            u'[10 2; 11 -400] * u'[17 16; -500 -600] *
            w'[5; 4 10] * w'[9; 2 17] * w'[12; 16 1] *
            rho[6 7 13; 5 9 12] *
            w[4 8; 6] * w[3 15; 7] * w[14 1; 13] *
            u[11 -100; 8 3] * u[-200 -300; 15 14]
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

function environment(layer::BinaryLayer, op, rho; vary_disentanglers=true)
    if vary_disentanglers
        env_u = environment_disentangler(op, layer, rho)
    else
        # The adjoint is just for type stability.
        env_u = zero(layer.disentangler')'
    end
    env_w = environment_isometry(op, layer, rho)
    return BinaryLayer(env_u, env_w)
end

function minimize_expectation_ev(layer::BinaryLayer, env::BinaryLayer;
                                 vary_disentanglers=true)
    u = (vary_disentanglers ? projectisometric(env.disentangler; alg=Polar())
         : layer.disentangler)
    w = projectisometric(env.isometry; alg=Polar())
    return BinaryLayer(u, w)
end

function environment_disentangler(h::BinaryOperator, layer::BinaryLayer, rho)
    u, w = layer
    @tensor(
            env1[-1 -2; -3 -4] :=
            rho[15 14 9; 17 18 10] *
            w[5 6; 15] * w[16 -3; 14] * w[-4 8; 9] *
            u[1 2; 6 16] *
            h[3 4 13; 1 2 -1] *
            u'[7 12; 3 4] * u'[11 19; 13 -2] *
            w'[17; 5 7] * w'[18; 12 11] * w'[10; 19 8]
           )

    @tensor(
            env2[-1 -2; -3 -4] :=
            rho[3 10 5; 4 15 6] *
            w[1 11; 3] * w[9 -3; 10] * w[-4 2; 5] *
            u[12 19; 11 9] *
            h[18 7 8; 19 -1 -2] *
            u'[13 14; 12 18] * u'[16 17; 7 8] *
            w'[4; 1 13] * w'[15; 14 16] * w'[6; 17 2]
           )

    @tensor(
            env3[-1 -2; -3 -4] :=
            rho[5 10 3; 6 15 4] *
            w[2 -3; 5] * w[-4 9; 10] * w[11 1; 3] *
            u[19 12; 9 11] *
            h[8 7 18; -1 -2 19] *
            u'[17 16; 8 7] * u'[14 13; 18 12] *
            w'[6; 2 17] * w'[15; 16 14] * w'[4; 13 1]
           )

    @tensor(
            env4[-1 -2; -3 -4] :=
            rho[9 14 15; 10 18 17] *
            w[8 -3; 9] * w[-4 16; 14] * w[6 5; 15] *
            u[2 1; 16 6] *
            h[13 4 3; -2 2 1] *
            u'[19 11; -1 13] * u'[12 7; 4 3] *
            w'[10; 8 19] * w'[18; 11 12] * w'[17; 7 5]
           )

    env = (env1 + env2 + env3 + env4)/2
    # Complex conjugate.
    env = permute(env', (3,4), (1,2))
    return env
end

function environment_disentangler(h::Union{SquareTensorMap{1}, SquareTensorMap{2}},
                                    layer::BinaryLayer, rho)
    h = expand_support(h, causal_cone_width(BinaryLayer))
    return environment_disentangler(h, layer, rho)
end

function environment_isometry(h::BinaryOperator, layer, rho)
    u, w = layer
    @tensor(
            env1[-1 -2; -3] :=
            rho[18 17 -3; 16 15 19] *
            w[5 6; 18] * w[9 8; 17] *
            u[2 1; 6 9] * u[10 11; 8 -1] *
            h[4 3 12; 2 1 10] *
            u'[7 14; 4 3] * u'[13 20; 12 11] *
            w'[16; 5 7] * w'[15; 14 13] * w'[19; 20 -2]
           )

    @tensor(
            env2[-1 -2; -3] :=
            rho[16 15 -3; 18 17 19] *
            w[12 13; 16] * w[5 6; 15] *
            u[9 7; 13 5] * u[2 1; 6 -1] *
            h[8 4 3; 7 2 1] *
            u'[14 11; 9 8] * u'[10 20; 4 3] *
            w'[18; 12 14] * w'[17; 11 10] * w'[19; 20 -2]
           )

    @tensor(
            env3[-1 -2; -3] :=
            rho[18 -3 14; 19 20 15] *
            w[5 6; 18] * w[17 13; 14] *
            u[2 1; 6 -1] * u[12 11; -2 17] *
            h[4 3 9; 2 1 12] *
            u'[7 10; 4 3] * u'[8 16; 9 11] *
            w'[19; 5 7] * w'[20; 10 8] * w'[15; 16 13]
           )

    @tensor(
            env4[-1 -2; -3] :=
            rho[14 -3 18; 15 20 19] *
            w[13 17; 14] * w[6 5; 18] *
            u[11 12; 17 -1] * u[1 2; -2 6] *
            h[9 3 4; 12 1 2] *
            u'[16 8; 11 9] * u'[10 7; 3 4] *
            w'[15; 13 16] * w'[20; 8 10] * w'[19; 7 5]
           )

    @tensor(
            env5[-1 -2; -3] :=
            rho[-3 15 16; 19 17 18] *
            w[6 5; 15] * w[13 12; 16] *
            u[1 2; -2 6] * u[7 9; 5 13] *
            h[3 4 8; 1 2 7] *
            u'[20 10; 3 4] * u'[11 14; 8 9] *
            w'[19; -1 20] * w'[17; 10 11] * w'[18; 14 12]
           )

    @tensor(
            env6[-1 -2; -3] :=
            rho[-3 17 18; 19 15 16] *
            w[8 9; 17] * w[6 5; 18] *
            u[11 10; -2 8] * u[1 2; 9 6] *
            h[12 3 4; 10 1 2] *
            u'[20 13; 11 12] * u'[14 7; 3 4] *
            w'[19; -1 20] * w'[15; 13 14] * w'[16; 7 5]
           )

    env = (env1 + env2 + env3 + env4 + env5 + env6)/2
    # Complex conjugate.
    env = permute(env', (2,3), (1,))
    return env
end

function environment_isometry(h::Union{SquareTensorMap{1}, SquareTensorMap{2}},
                                layer::BinaryLayer, rho)
    h = expand_support(h, causal_cone_width(BinaryLayer))
    return environment_isometry(h, layer, rho)
end
