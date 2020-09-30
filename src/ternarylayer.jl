# TernaryLayer and TernaryMERA types, and methods thereof.
# To be `included` in MERA.jl.

"""
    TernaryLayer{ST, ET, Tan} <: SimpleLayer

The type for layers of a ternary MERA.

Each layer consists of two tensors, a 2-to-2 disentangler, often called `u`, and a 3-to-1
isometry, often called `w`.

The type parameters are `ST` for space type, e.g. `ComplexSpace` or `SU2Space`; `ET` for
element type, e.g. `Complex{Float64}`; and `Tan` for whether this layer is a tangent layer
or not.  If `Tan = false`, the layer is question is an actual MERA layer. If `Tan = true` it
consists, instead of the actual tensors, of Stiefel/Grassmann tangent vectors of these
tensors.

`TernaryLayer` implements the iteration interface, returning first the disentangler, then
the isometry.

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
    4│
 ┌───┴───┐
 │   w   │
 └┬──┬──┬┘
 1│ 2│ 3│
```
"""
struct TernaryLayer{ST, ET, Tan} <: SimpleLayer
    # Even though the types are here marked as any, they are restricted to be specific,
    # concrete types dependent on ST, ET and Tan, both in the constructor and in
    # getproperty.
    disentangler::Any
    isometry::Any

    function TernaryLayer{ST, ET, Tan}(disentangler, isometry) where {ST, ET, Tan}
        DisType = disentangler_type(ST, ET, Tan)
        IsoType = ternaryisometry_type(ST, ET, Tan)
        disconv = convert(DisType, disentangler)::DisType
        isoconv = convert(IsoType, isometry)::IsoType
        return new{ST, ET, Tan}(disconv, isoconv)
    end
end

function TernaryLayer(disentangler::DisType, isometry::IsoType
                     ) where {ST,
                              DisType <: AbstractTensorMap{ST, 2, 2},
                              IsoType <: AbstractTensorMap{ST, 3, 1}}
    ET = eltype(DisType)
    @assert eltype(IsoType) === ET
    return TernaryLayer{ST, ET, false}(disentangler, isometry)
end

function TernaryLayer(disentangler::DisTanType, isometry::IsoTanType
                     ) where {ST,
                              DisType <: AbstractTensorMap{ST, 2, 2},
                              IsoType <: AbstractTensorMap{ST, 3, 1},
                              DisTanType <: Stiefel.StiefelTangent{DisType},
                              IsoTanType <: Grassmann.GrassmannTangent{IsoType},
                             }
    ET = eltype(DisType)
    @assert eltype(IsoType) === ET
    return TernaryLayer{ST, ET, true}(disentangler, isometry)
end

function Base.getproperty(l::TernaryLayer{ST, ET, Tan}, sym::Symbol) where {ST, ET, Tan}
    if sym === :disentangler
        T = disentangler_type(ST, ET, Tan)
    elseif sym === :isometry
        T = ternaryisometry_type(ST, ET, Tan)
    else
        T = Any
    end
    return getfield(l, sym)::T
end

"""
    TernaryMERA{N}

A ternary MERA is a MERA consisting of `TernaryLayer`s.
"""
TernaryMERA{N} = GenericMERA{N, T, O} where {T <: TernaryLayer, O}
layertype(::Type{TernaryMERA}) = TernaryLayer
#Base.show(io::IO, ::Type{TernaryMERA}) = print(io, "TernaryMERA")
#Base.show(io::IO, ::Type{TernaryMERA{N}}) where {N} = print(io, "TernaryMERA{($N)}")

# Implement the iteration and indexing interfaces.
# See simplelayer.jl for details.
_tuple(layer::TernaryLayer) = (layer.disentangler, layer.isometry)

function operatortype(::Type{TernaryLayer{ST, ET, false}}) where {ST, ET}
    return tensormaptype(ST, 2, 2, ET)
end
operatortype(::Type{TernaryLayer{ST, ET, true}}) where {ST, ET} = Nothing

TensorKit.spacetype(::Type{TernaryLayer{ST, ET, Tan}}) where {ST, ET, Tan} = ST

scalefactor(::Type{<:TernaryLayer}) = 3

causal_cone_width(::Type{<:TernaryLayer}) = 2

Base.eltype(::Type{TernaryLayer{ST, ET, Tan}}) where {ST, ET, Tan} = ET

outputspace(layer::TernaryLayer) = space(layer.disentangler, 1)
inputspace(layer::TernaryLayer) = space(layer.isometry, 4)'
internalspace(layer::TernaryLayer) = space(layer.isometry, 1)
internalspace(m::TernaryMERA, depth) = internalspace(getlayer(m, depth))

function expand_inputspace(layer::TernaryLayer, V_new)
    u, w = layer
    w = pad_with_zeros_to(w, 4 => V_new')
    return TernaryLayer(u, w)
end

function expand_outputspace(layer::TernaryLayer, V_new)
    u, w = layer
    u = pad_with_zeros_to(u, 1 => V_new, 2 => V_new)
    w = pad_with_zeros_to(w, 2 => V_new)
    return TernaryLayer(u, w)
end

function expand_internalspace(layer::TernaryLayer, V_new)
    u, w = layer
    u = pad_with_zeros_to(u, 3 => V_new', 4 => V_new')
    w = pad_with_zeros_to(w, 1 => V_new, 3 => V_new)
    return TernaryLayer(u, w)
end

function randomlayer(::Type{TernaryLayer}, ::Type{T}, Vin, Vout, Vint = Vout;
                     random_disentangler = false) where {T}
    w = randomisometry(T, Vint ⊗ Vout ⊗ Vint, Vin)
    u = initialize_disentangler(T, Vout, Vint, random_disentangler)
    return TernaryLayer(u, w)
end

function ascending_fixedpoint(layer::TernaryLayer)
    width = causal_cone_width(layer)
    Vtotal = ⊗(ntuple(n->inputspace(layer), Val(width))...)
    return id(storagetype(operatortype(layer)), Vtotal)
end

function scalingoperator_initialguess(l::TernaryLayer, irreps...)
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

function gradient(layer::TernaryLayer, env::TernaryLayer; metric = :euclidean)
    u, w = layer
    uenv, wenv = env
    # The environment is the partial derivative. We need to turn that into a tangent vector
    # of the Stiefel manifold point u or w.
    # The factor of two is from the partial_x + i partial_y derivative of the cost function,
    # and how it depends on both v and v^dagger.
    ugrad = Stiefel.project!(2*uenv, u; metric = metric)
    wgrad = Grassmann.project!(2*wenv, w)
    return TernaryLayer(ugrad, wgrad)
end

function precondition_tangent(layer::TernaryLayer, tan::TernaryLayer, rho)
    u, w = layer
    utan, wtan = tan
    # TODO This is just a silly hack to work around the fact that I haven't yet made trace!
    # work with anyonic tensors. trace! calls permute of double fusion trees, which isn't
    # well-defined for anyons.
    V = space(rho, 1)
    eye = isomorphism(Matrix{eltype(rho)}, V', V')
    @tensor rho_wl[-1; -11] := eye[1; 2] * rho[-1 1; -11 2]
    @tensor rho_wr[-1; -11] := rho[1 -1; 2 -11] * eye[1; 2]
    #@tensor rho_wl[-1; -11] := rho[-1 1; -11 1]
    #@tensor rho_wr[-1; -11] := rho[1 -1; 1 -11]
    rho_w = (rho_wl + rho_wr) / 2.0
    @tensor(rho_u[-1 -2; -11 -12] :=
            w[1 2 -1; 11] * w[-2 3 4; 21] *
            rho[11 21; 12 22] *
            w'[12; 1 2 -11] * w'[22; -12 3 4])
    utan_prec = precondition_tangent(utan, rho_u)
    wtan_prec = precondition_tangent(wtan, rho_w)
    return TernaryLayer(utan_prec, wtan_prec)
end

# # # Invariants

function space_invar_intralayer(layer::TernaryLayer)
    u, w = layer
    matching_bonds = ((space(u, 3)', space(w, 3)),
                      (space(u, 4)', space(w, 1)))
    allmatch = all(pair -> ==(pair...), matching_bonds)
    # Check that the dimensions are such that isometricity can hold.
    allmatch &= all((u,w)) do v
        codom, dom = fuse(codomain(v)), fuse(domain(v))
        infimum(dom, codom) == dom
    end
    return allmatch
end

function space_invar_interlayer(layer::TernaryLayer, next_layer::TernaryLayer)
    u, w = layer.disentangler, layer.isometry
    unext, wnext = next_layer.disentangler, next_layer.isometry
    matching_bonds = ((space(w, 4)', space(unext, 1)),
                      (space(w, 4)', space(unext, 2)),
                      (space(w, 4)', space(wnext, 2)))
    allmatch = all(pair -> ==(pair...), matching_bonds)
    return allmatch
end

# # # Ascending and descending superoperators

"""
Return the ascending superoperator of the one site in the middle of the isometries in a
TernaryMERA, as a TensorMap. Unlike most ascending superoperators, this one is actually
affordable to construct as a full tensor.
"""
ascending_superop_onesite(m::TernaryMERA) = ascending_superop_onesite(getlayer(m, Inf))

function ascending_superop_onesite(layer::TernaryLayer)
    w = layer.isometry
    @tensor superop[-1 -2; -11 -12] := w'[-11; 1 -1 2] * w[1 -2 2; -12]
    return superop
end

const TernaryOperator{S} = AbstractTensorMap{S, 2, 2}
const ChargedTernaryOperator{S} = AbstractTensorMap{S, 2, 3}
const DoubleChargedTernaryOperator{S} = AbstractTensorMap{S, 2, 4}

function ascend_left(op::ChargedTernaryOperator, layer::TernaryLayer)
    u, w = layer
    # Cost: 2X^8 + 2X^7 + 2X^6
    @tensor(
            scaled_op[-100 -200; -300 -400 -1000] :=
            w'[-100; 51 31 21] * w'[-200; 55 11 12] *
            u'[21 55; 32 42] *
            op[31 32; 52 41 -1000] *
            u[41 42; 53 54] *
            w[51 52 53; -300] * w[54 11 12; -400]
           )
    return scaled_op
end

function ascend_right(op::ChargedTernaryOperator, layer::TernaryLayer)
    u, w = layer
    # Cost: 2X^8 + 2X^7 + 2X^6
    @tensor(
            scaled_op[-100 -200; -300 -400 -1000] :=
            w'[-100; 11 12 64] * w'[-200; 21 41 62] *
            u'[64 21; 51 31] *
            op[31 41; 52 61 -1000] *
            u[51 52; 65 63] *
            w[11 12 65; -300] * w[63 61 62; -400]
           )
    return scaled_op
end

function ascend_mid(op::ChargedTernaryOperator, layer::TernaryLayer)
    u, w = layer
    # Cost: 6X^6
    @tensor(
            scaled_op[-100 -200; -300 -400 -1000] :=
            w'[-100; 31 32 42] * w'[-200; 52 21 22] *
            u'[42 52; 11 12] *
            op[11 12; 1 2 -1000] *
            u[1 2; 41 51] *
            w[31 32 41; -300] * w[51 21 22; -400]
           )
    return scaled_op
end

function ascend(op::Union{TernaryOperator,
                          ChargedTernaryOperator,
                          DoubleChargedTernaryOperator},
                layer::TernaryLayer) where {S1}
    u, w = layer
    l = ascend_left(op, layer)
    r = ascend_right(op, layer)
    m = ascend_mid(op, layer)
    scaled_op = (l+r+m)/3
    return scaled_op
end

ascend_left(op::TernaryOperator, layer::TernaryLayer) =
    remove_dummy_index(ascend_left(append_dummy_index(op), layer))

ascend_right(op::TernaryOperator, layer::TernaryLayer) =
    remove_dummy_index(ascend_right(append_dummy_index(op), layer))

ascend_mid(op::TernaryOperator, layer::TernaryLayer) =
    remove_dummy_index(ascend_mid(append_dummy_index(op), layer))

ascend(op::SquareTensorMap{1}, layer::TernaryLayer) =
    ascend(expand_support(op, causal_cone_width(TernaryLayer)), layer)

# TODO Figure out how to deal with the extra charge legs in the case of anyonic tensors.
function ascend_left(op::TernaryOperator, layer::TernaryLayer{GradedSpace[FibonacciAnyon]})
    u, w = layer
    # Cost: 2X^8 + 2X^7 + 2X^6
    @tensor(
            scaled_op[-100 -200; -300 -400] :=
            w'[-100; 51 31 21] * w'[-200; 55 11 12] *
            u'[21 55; 32 42] *
            op[31 32; 52 41] *
            u[41 42; 53 54] *
            w[51 52 53; -300] * w[54 11 12; -400]
           )
    return scaled_op
end

function ascend_right(op::TernaryOperator, layer::TernaryLayer{GradedSpace[FibonacciAnyon]})
    u, w = layer
    # Cost: 2X^8 + 2X^7 + 2X^6
    @tensor(
            scaled_op[-100 -200; -300 -400] :=
            w'[-100; 11 12 64] * w'[-200; 21 41 62] *
            u'[64 21; 51 31] *
            op[31 41; 52 61] *
            u[51 52; 65 63] *
            w[11 12 65; -300] * w[63 61 62; -400]
           )
    return scaled_op
end

function ascend_mid(op::TernaryOperator, layer::TernaryLayer{GradedSpace[FibonacciAnyon]})
    u, w = layer
    # Cost: 6X^6
    @tensor(
            scaled_op[-100 -200; -300 -400] :=
            w'[-100; 31 32 42] * w'[-200; 52 21 22] *
            u'[42 52; 11 12] *
            op[11 12; 1 2] *
            u[1 2; 41 51] *
            w[31 32 41; -300] * w[51 21 22; -400]
           )
    return scaled_op
end

function ascend_left(op::DoubleChargedTernaryOperator,
                     layer::TernaryLayer{GradedSpace[FibonacciAnyon]})
    u, w = layer
    # Cost: 2X^8 + 2X^7 + 2X^6
    @tensor(
            temp1[-1 -2 -3 -4; -11 -12 -13 -14] :=
            w'[-3; -2 31 21] *
            u'[21 -4; 32 -14] *
            op[31 32; -1 -11 -12 -13]
           )
    temp2 = braid(temp1, (11, 12, 13, 14, 15, 1, 100, 16), (1, 2, 3, 7, 6, 4), (5, 8))
    @tensor(
            temp3[-100 -1000 -2000 -200; -300 -400] :=
            w'[-200; 55 11 12] *
            temp2[52 51 -100 -1000 -2000 55; 41 42] *
            u[41 42; 53 54] *
            w[51 52 53; -300] * w[54 11 12; -400]
           )
    scaled_op = braid(temp3, (11, 100, 1, 12, 13, 14), (1, 4), (5, 6, 3, 2))
    return scaled_op
end

function ascend_right(op::DoubleChargedTernaryOperator,
                      layer::TernaryLayer{GradedSpace[FibonacciAnyon]})
    u, w = layer
    # Cost: 2X^8 + 2X^7 + 2X^6
    @tensor(
            temp1[-1 -2 -3 -4 -5 -6; -11 -12] :=
            w'[-2; 21 41 -3] *
            u'[-1 21; -11 31] *
            op[31 41; -12 -6 -5 -4]
           )
    temp2 = braid(temp1, (11, 12, 13, 100, 1, 14, 15, 16), (1, 2, 4, 5, 3, 6), (7, 8))
    @tensor(
            temp3[-100 -200 -1000 -2000; -300 -400] :=
            w'[-100; 11 12 64] *
            temp2[64 -200 -1000 -2000 62 61; 51 52] *
            u[51 52; 65 63] *
            w[11 12 65; -300] * w[63 61 62; -400]
           )
    scaled_op = braid(temp3, (11, 12, 100, 1, 13, 14), (1, 2), (5, 6, 4, 3))
    return scaled_op
end

function ascend_mid(op::DoubleChargedTernaryOperator,
                    layer::TernaryLayer{GradedSpace[FibonacciAnyon]})
    u, w = layer
    # Cost: 6X^6
    @tensor(
            temp1[-1 -2; -3 -4 -5 -6] :=
            u'[-1 -2; 11 12] *
            op[11 12; 1 2 -5 -6] *
            u[1 2; -3 -4]
           )
    temp2 = braid(temp1, (11, 12, 13, 14, 1, 100), (1, 6, 5, 2), (3, 4))
    @tensor(
            temp3[-100 -1000 -2000 -200; -300 -400] :=
            w'[-100; 31 32 42] * w'[-200; 52 21 22] *
            temp2[42 -1000 -2000 52; 41 51] *
            w[31 32 41; -300] * w[51 21 22; -400]
           )
    scaled_op = braid(temp3, (11, 100, 1, 12, 13, 14), (1, 4), (5, 6, 3, 2))
    return scaled_op
end

function descend_left(rho::TernaryOperator, layer::TernaryLayer)
    u, w = layer
    # Cost: 2X^8 + 2X^7 + 2X^6
    @tensor(
            scaled_rho[-100 -200; -300 -400] :=
            u[-200 63; 41 31] *
            w[52 -100 41; 42] * w[31 11 12; 22] *
            rho[42 22; 51 21] *
            w'[51; 52 -300 61] * w'[21; 62 11 12] *
            u'[61 62; -400 63]
           )
    return scaled_rho
end

function descend_right(rho::TernaryOperator, layer::TernaryLayer)
    u, w = layer
    # Cost: 2X^8 + 2X^7 + 2X^6
    @tensor(
            scaled_rho[-100 -200; -300 -400] :=
            u[63 -100; 41 31] *
            w[11 12 41; 22] * w[31 -200 52; 42] *
            rho[22 42; 21 51] *
            w'[21; 11 12 62] * w'[51; 61 -400 52] *
            u'[62 61; 63 -300]
           )
    return scaled_rho
end

function descend_mid(rho::TernaryOperator, layer::TernaryLayer)
    u, w = layer
    # Cost: 6X^6
    @tensor(
            scaled_rho[-100 -200; -300 -400] :=
            u[-100 -200; 51 52] *
            w[11 12 51; 22] * w[52 31 32; 42] *
            rho[22 42; 21 41] *
            w'[21; 11 12 61] * w'[41; 62 31 32] *
            u'[61 62; -300 -400]
           )
    return scaled_rho
end

function descend(rho::TernaryOperator, layer::TernaryLayer)
    u, w = layer
    l = descend_left(rho, layer)
    r = descend_right(rho, layer)
    m = descend_mid(rho, layer)
    scaled_rho = (l+r+m)/3
    return scaled_rho
end

# # # Optimization

function environment(op, layer::TernaryLayer, rho; vary_disentanglers = true)
    if vary_disentanglers
        env_u = environment_disentangler(op, layer, rho)
    else
        # The adjoint is just for type stability.
        env_u = zero(layer.disentangler')'
    end
    env_w = environment_isometry(op, layer, rho)
    return TernaryLayer(env_u, env_w)
end

function minimize_expectation_ev(layer::TernaryLayer, env::TernaryLayer;
                                 vary_disentanglers = true)
    u = (vary_disentanglers ? projectisometric(env.disentangler; alg = Polar())
         : layer.disentangler)
    w = projectisometric(env.isometry; alg = Polar())
    return TernaryLayer(u, w)
end

function environment_disentangler(h::TernaryOperator, layer, rho)
    u, w = layer
    # Cost: 2X^8 + 2X^7 + 2X^6
    @tensor(
            env1[-1 -2; -3 -4] :=
            rho[31 21; 63 22] *
            w'[63; 61 62 -3] * w'[22; -4 11 12] *
            h[62 -1; 51 52] *
            u[52 -2; 41 42] *
            w[61 51 41; 31] * w[42 11 12; 21]
           )

    # Cost: 6X^6
    @tensor(
            env2[-1 -2; -3 -4] :=
            rho[41 51; 42 52] *
            w'[42; 21 22 -3] * w'[52; -4 31 32] *
            h[-1 -2; 11 12] *
            u[11 12; 61 62] *
            w[21 22 61; 41] * w[62 31 32; 51]
           )

    # Cost: 2X^8 + 2X^7 + 2X^6
    @tensor(
            env3[-1 -2; -3 -4] :=
            rho[21 31; 22 63] *
            w'[22; 12 11 -3] * w'[63; -4 62 61] *
            h[-2 62; 52 51] *
            u[-1 52; 42 41] *
            w[12 11 42; 21] * w[41 51 61; 31]
           )

    env = (env1 + env2 + env3)/3
    return env
end

function environment_isometry(h::TernaryOperator, layer, rho)
    u, w = layer
    # Cost: 2X^8 + 2X^7 + 2X^6
    @tensor(
            env1[-1 -2 -3; -4] :=
            rho[81 84; 82 -4] *
            w'[82; 62 61 63] *
            u'[63 -1; 51 52] *
            h[61 51; 41 42] *
            u[42 52; 31 83] *
            w[62 41 31; 81] * w[83 -2 -3; 84]
           )

    # Cost: 6X^6
    @tensor(
            env2[-1 -2 -3; -4] :=
            rho[41 62; 42 -4] *
            w'[42; 11 12 51] *
            u'[51 -1; 21 22] *
            h[21 22; 31 32] *
            u[31 32; 52 61] *
            w[11 12 52; 41] * w[61 -2 -3; 62]
           )

    # Cost: 2X^8 + 2X^7 + 2X^6
    @tensor(
            env3[-1 -2 -3; -4] :=
            rho[31 33; 32 -4] *
            w'[32; 21 11 73] *
            u'[73 -1; 72 71] *
            h[71 -2; 62 61] *
            u[72 62; 51 41] *
            w[21 11 51; 31] * w[41 61 -3; 33]
           )

    # Cost: 2X^8 + 2X^7 + 2X^6
    @tensor(
            env4[-1 -2 -3; -4] :=
            rho[33 31; -4 32] *
            w'[32; 73 11 21] *
            u'[-3 73; 71 72] *
            h[-2 71; 61 62] *
            u[62 72; 41 51] *
            w[-1 61 41; 33] * w[51 11 21; 31]
           )

    # Cost: 6X^6
    @tensor(
            env5[-1 -2 -3; -4] :=
            rho[62 41; -4 42] *
            w'[42; 51 12 11] *
            u'[-3 51; 22 21] *
            h[22 21; 32 31] *
            u[32 31; 61 52] *
            w[-1 -2 61; 62] * w[52 12 11; 41]
           )

    # Cost: 2X^8 + 2X^7 + 2X^6
    @tensor(
            env6[-1 -2 -3; -4] :=
            rho[84 81; -4 82] *
            w'[82; 63 61 62] *
            u'[-3 63; 52 51] *
            h[51 61; 42 41] *
            u[52 42; 83 31] *
            w[-1 -2 83; 84] * w[31 41 62; 81]
           )

    env = (env1 + env2 + env3 + env4 + env5 + env6)/3
    return env
end
