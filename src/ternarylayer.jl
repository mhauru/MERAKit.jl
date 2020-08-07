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
    4|
 +-------+
 |   w   |
 +-------+
 1| 2| 3|
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
Base.show(io::IO, ::Type{TernaryMERA}) = print(io, "TernaryMERA")
Base.show(io::IO, ::Type{TernaryMERA{N}}) where {N} = print(io, "TernaryMERA{($N)}")

# Given an instance of a type like TernaryLayer{ComplexSpace, Float64, true},
# return the unparametrised type TernaryLayer.
layertype(::TernaryLayer) = TernaryLayer
layertype(::Type{T}) where T <: TernaryMERA = TernaryLayer

function operatortype(::Type{TernaryLayer{ST, ET, false}}) where {ST, ET}
    return tensortype(ST, Val(2), Val(2), ET)
end
operatortype(::Type{TernaryLayer{ST, ET, true}}) where {ST, ET} = Nothing

Base.eltype(::Type{TernaryLayer{ST, ET, Tan}}) where {ST, ET, Tan} = ET
Base.eltype(l::TernaryLayer{ST, ET, Tan}) where {ST, ET, Tan} = ET

function Base.convert(::Type{TernaryLayer{T1, T2}}, l::TernaryLayer) where {T1, T2}
    return TernaryLayer(convert(T1, l.disentangler), convert(T2, l.isometry))
end

# Implement the iteration and indexing interfaces.
Base.iterate(layer::TernaryLayer) = (layer.disentangler, Val(1))
Base.iterate(layer::TernaryLayer, ::Val{1}) = (layer.isometry, Val(2))
Base.iterate(layer::TernaryLayer, ::Val{2}) = nothing
Base.length(layer::TernaryLayer) = 2

scalefactor(::Type{<:TernaryLayer}) = 3
scalefactor(::Type{TernaryMERA}) = scalefactor(TernaryLayer)

get_disentangler(m::TernaryMERA, depth) = get_layer(m, depth).disentangler
get_isometry(m::TernaryMERA, depth) = get_layer(m, depth).isometry

function set_disentangler!(m::TernaryMERA, u, depth; kwargs...)
    w = get_isometry(m, depth)
    return set_layer!(m, (u, w), depth; kwargs...)
end

function set_isometry!(m::TernaryMERA, w, depth; kwargs...)
    u = get_disentangler(m, depth)
    return set_layer!(m, (u, w), depth; kwargs...)
end

causal_cone_width(::Type{<:TernaryLayer}) = 2

outputspace(layer::TernaryLayer) = space(layer.disentangler, 1)
inputspace(layer::TernaryLayer) = space(layer.isometry, 4)'
internalspace(layer::TernaryLayer) = space(layer.isometry, 1)
internalspace(m::TernaryMERA, depth) = internalspace(get_layer(m, depth))

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

function randomlayer(::Type{TernaryLayer}, T, Vin, Vout, Vint=Vout;
                     random_disentangler=false)
    w = randomisometry(T, Vint ⊗ Vout ⊗ Vint, Vin)
    u = initialize_disentangler(T, Vout, Vint, random_disentangler)
    return TernaryLayer(u, w)
end

function ascending_fixedpoint(layer::TernaryLayer)
    V = inputspace(layer)
    width = causal_cone_width(typeof(layer))
    Vtotal = ⊗(Iterators.repeated(V, width)...)::ProductSpace{typeof(V), width}
    eye = id(Vtotal) / sqrt(dim(Vtotal))
    return eye
end

function gradient(layer::TernaryLayer, env::TernaryLayer; metric=:euclidean)
    u, w = layer
    uenv, wenv = env
    # The environment is the partial derivative. We need to turn that into a tangent vector
    # of the Stiefel manifold point u or w.
    # The factor of two is from the partial_x + i partial_y derivative of the cost function,
    # and how it depends on both v and v^dagger.
    ugrad = Stiefel.project!(2*uenv, u; metric=metric)
    wgrad = Grassmann.project!(2*wenv, w)
    return TernaryLayer(ugrad, wgrad)
end

function precondition_tangent(layer::TernaryLayer, tan::TernaryLayer, rho)
    u, w = layer
    utan, wtan = tan
    @tensor rho_wl[-1; -11] := rho[-1 1; -11 1]
    @tensor rho_wr[-1; -11] := rho[1 -1; 1 -11]
    rho_w = (rho_wl + rho_wr) / 2.0
    @tensor(rho_u[-1 -2; -11 -12] :=
            w'[12; 1 2 -11] * w'[22; -12 3 4] *
            rho[11 21; 12 22] *
            w[1 2 -1; 11] * w[-2 3 4; 21])
    utan_prec = precondition_tangent(utan, rho_u)
    wtan_prec = precondition_tangent(wtan, rho_w)
    return TernaryLayer(utan_prec, wtan_prec)
end

# # # Invariants

function space_invar_intralayer(layer::TernaryLayer)
    u, w = layer
    matching_bonds = ((space(u, 3)', space(w, 3)),
                      (space(u, 4)', space(w, 1)))
    allmatch = all([==(pair...) for pair in matching_bonds])
    # Check that the dimensions are such that isometricity can hold.
    for v in layer
        codom, dom = fuse(codomain(v)), fuse(domain(v))
        allmatch = allmatch && infinum(dom, codom) == dom
    end
    return allmatch
end

function space_invar_interlayer(layer::TernaryLayer, next_layer::TernaryLayer)
    u, w = layer.disentangler, layer.isometry
    unext, wnext = next_layer.disentangler, next_layer.isometry
    matching_bonds = ((space(w, 4)', space(unext, 1)),
                      (space(w, 4)', space(unext, 2)),
                      (space(w, 4)', space(wnext, 2)))
    allmatch = all([==(pair...) for pair in matching_bonds])
    return allmatch
end

# # # Ascending and descending superoperators

"""
Return the ascending superoperator of the one site in the middle of the isometries in a
TernaryMERA, as a TensorMap. Unlike most ascending superoperators, this one is actually
affordable to construct as a full tensor.
"""
ascending_superop_onesite(m::TernaryMERA) = ascending_superop_onesite(get_layer(m, Inf))

function ascending_superop_onesite(layer::TernaryLayer)
    w = layer.isometry
    @tensor(superop[-1 -2; -11 -12] := w[1 -2 2; -12] * w'[-11; 1 -1 2])
    return superop
end

function ascend_left(op::SquareTensorMap{2}, layer::TernaryLayer)
    u, w = layer
    # Cost: 2X^8 + 2X^7 + 2X^6
    @tensor(
            scaled_op[-100 -200; -300 -400] :=
            w[51 52 53; -300 ] * w[54 11 12; -400] *
            u[41 42; 53 54] *
            op[31 32; 52 41] *
            u'[21 55; 32 42] *
            w'[-100; 51 31 21] * w'[-200; 55 11 12]
           )
    return scaled_op
end

function ascend_right(op::SquareTensorMap{2}, layer::TernaryLayer)
    u, w = layer
    # Cost: 2X^8 + 2X^7 + 2X^6
    @tensor(
            scaled_op[-100 -200; -300 -400] :=
            w[11 12 65; -300] * w[63 61 62; -400] *
            u[51 52; 65 63] *
            op[31 41; 52 61] *
            u'[64 21; 51 31] *
            w'[-100; 11 12 64] * w'[-200; 21 41 62]
           )
    return scaled_op
end

function ascend_mid(op::SquareTensorMap{2}, layer::TernaryLayer)
    u, w = layer
    # Cost: 6X^6
    @tensor(
            scaled_op[-100 -200; -300 -400] :=
            w[31 32 41; -300] * w[51 21 22; -400] *
            u[1 2; 41 51] *
            op[11 12; 1 2] *
            u'[42 52; 11 12] *
            w'[-100; 31 32 42] * w'[-200; 52 21 22]
           )
    return scaled_op
end

function ascend(op::SquareTensorMap{2}, layer::TernaryLayer)
    l = ascend_left(op, layer)
    r = ascend_right(op, layer)
    m = ascend_mid(op, layer)
    scaled_op = (l+r+m)/3.
    return scaled_op
end

# TODO Would there be a nice way of doing this where I wouldn't have to replicate all the
# network contractions? @ncon could do it, but Jutho's testing says it's significantly
# slower. This is only used for diagonalizing in charge sectors, so having tensors with
# non-trivial charge would also solve this.
function ascend_left(op::AbstractTensorMap{S1,2,3}, layer::TernaryLayer) where {S1}
    u, w = layer
    # Cost: 2X^8 + 2X^7 + 2X^6
    @tensor(
            scaled_op[-100 -200; -300 -400 -1000] :=
            w[51 52 53; -300] * w[54 11 12; -400] *
            u[41 42; 53 54] *
            op[31 32; 52 41 -1000] *
            u'[21 55; 32 42] *
            w'[-100; 51 31 21] * w'[-200; 55 11 12]
           )
    return scaled_op
end

function ascend_right(op::AbstractTensorMap{S1,2,3}, layer::TernaryLayer) where {S1}
    u, w = layer
    # Cost: 2X^8 + 2X^7 + 2X^6
    @tensor(
            scaled_op[-100 -200; -300 -400 -1000] :=
            w[11 12 65; -300] * w[63 61 62; -400] *
            u[51 52; 65 63] *
            op[31 41; 52 61 -1000] *
            u'[64 21; 51 31] *
            w'[-100; 11 12 64] * w'[-200; 21 41 62]
           )
    return scaled_op
end

function ascend_mid(op::AbstractTensorMap{S1,2,3}, layer::TernaryLayer) where {S1}
    u, w = layer
    # Cost: 6X^6
    @tensor(
            scaled_op[-100 -200; -300 -400 -1000] :=
            w[31 32 41; -300] * w[51 21 22; -400] *
            u[1 2; 41 51] *
            op[11 12; 1 2 -1000] *
            u'[42 52; 11 12] *
            w'[-100; 31 32 42] * w'[-200; 52 21 22]
           )
    return scaled_op
end

function ascend(op::AbstractTensorMap{S1,2,3}, layer::TernaryLayer) where {S1}
    u, w = layer
    l = ascend_left(op, layer)
    r = ascend_right(op, layer)
    m = ascend_mid(op, layer)
    scaled_op = (l+r+m)/3.
    return scaled_op
end

function ascend(op::SquareTensorMap{1}, layer::TernaryLayer)
    op = expand_support(op, causal_cone_width(TernaryLayer))
    return ascend(op, layer)
end

function descend_left(rho::SquareTensorMap{2}, layer::TernaryLayer)
    u, w = layer
    # Cost: 2X^8 + 2X^7 + 2X^6
    @tensor(
            scaled_rho[-100 -200; -300 -400] :=
            u'[61 62; -400 63] *
            w'[51; 52 -300 61] * w'[21; 62 11 12] *
            rho[42 22; 51 21] *
            w[52 -100 41; 42] * w[31 11 12; 22] *
            u[-200 63; 41 31]
           )
    return scaled_rho
end

function descend_right(rho::SquareTensorMap{2}, layer::TernaryLayer)
    u, w = layer
    # Cost: 2X^8 + 2X^7 + 2X^6
    @tensor(
            scaled_rho[-100 -200; -300 -400] :=
            u'[62 61; 63 -300] *
            w'[21; 11 12 62] * w'[51; 61 -400 52] *
            rho[22 42; 21 51] *
            w[11 12 41; 22] * w[31 -200 52; 42] *
            u[63 -100; 41 31]
           )
    return scaled_rho
end

function descend_mid(rho::SquareTensorMap{2}, layer::TernaryLayer)
    u, w = layer
    # Cost: 6X^6
    @tensor(
            scaled_rho[-100 -200; -300 -400] :=
            u'[61 62; -300 -400] *
            w'[21; 11 12 61] * w'[41; 62 31 32] *
            rho[22 42; 21 41] *
            w[11 12 51; 22] * w[52 31 32; 42] *
            u[-100 -200; 51 52]
           )
    return scaled_rho
end

function descend(rho::SquareTensorMap{2}, layer::TernaryLayer)
    u, w = layer
    l = descend_left(rho, layer)
    r = descend_right(rho, layer)
    m = descend_mid(rho, layer)
    scaled_rho = (l+r+m)/3.
    return scaled_rho
end

# # # Optimization

function environment(layer::TernaryLayer, op, rho; vary_disentanglers=true)
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
                                 vary_disentanglers=true)
    u = (vary_disentanglers ? projectisometric(env.disentangler; alg=Polar())
         : layer.disentangler)
    w = projectisometric(env.isometry; alg=Polar())
    return TernaryLayer(u, w)
end

function environment_disentangler(h::SquareTensorMap{2}, layer, rho)
    u, w = layer
    # Cost: 2X^8 + 2X^7 + 2X^6
    @tensor(
            env1[-1 -2; -3 -4] :=
            rho[63 22; 31 21] *
            w[61 62 -3; 63] * w[-4 11 12; 22] *
            h[51 52; 62 -1] *
            u'[41 42; 52 -2] *
            w'[31; 61 51 41] * w'[21; 42 11 12]
           )

    # Cost: 6X^6
    @tensor(
            env2[-1 -2; -3 -4] :=
            rho[42 52; 41 51] *
            w[21 22 -3; 42] * w[-4 31 32; 52] *
            h[11 12; -1 -2] *
            u'[61 62; 11 12] *
            w'[41; 21 22 61] * w'[51; 62 31 32]
           )

    # Cost: 2X^8 + 2X^7 + 2X^6
    @tensor(
            env3[-1 -2; -3 -4] :=
            rho[22 63; 21 31] *
            w[12 11 -3; 22] * w[-4 62 61; 63] *
            h[52 51; -2 62] *
            u'[42 41; -1 52] *
            w'[21; 12 11 42] * w'[31; 41 51 61]
           )

    env = (env1 + env2 + env3)/3
    # Complex conjugate.
    env = permute(env', (3,4), (1,2))
    return env
end

function environment_isometry(h::SquareTensorMap{2}, layer, rho)
    u, w = layer
    # Cost: 2X^8 + 2X^7 + 2X^6
    @tensor(
            env1[-1 -2 -3; -4] :=
            rho[82 -4; 81 84] *
            w[62 61 63; 82] *
            u[51 52; 63 -1] *
            h[41 42; 61 51] *
            u'[31 83; 42 52] *
            w'[81; 62 41 31] * w'[84; 83 -2 -3]
           )

    # Cost: 6X^6
    @tensor(
            env2[-1 -2 -3; -4] :=
            rho[42 -4; 41 62] *
            w[11 12 51; 42] *
            u[21 22; 51 -1] *
            h[31 32; 21 22] *
            u'[52 61; 31 32] *
            w'[41; 11 12 52] * w'[62; 61 -2 -3]
           )

    # Cost: 2X^8 + 2X^7 + 2X^6
    @tensor(
            env3[-1 -2 -3; -4] :=
            rho[32 -4; 31 33] *
            w[21 11 73; 32] *
            u[72 71; 73 -1] *
            h[62 61; 71 -2] *
            u'[51 41; 72 62] *
            w'[31; 21 11 51] * w'[33; 41 61 -3]
           )

    # Cost: 2X^8 + 2X^7 + 2X^6
    @tensor(
            env4[-1 -2 -3; -4] :=
            rho[-4 32; 33 31] *
            w[73 11 21; 32] *
            u[71 72; -3 73] *
            h[61 62; -2 71] *
            u'[41 51; 62 72] *
            w'[33; -1 61 41] * w'[31; 51 11 21]
           )

    # Cost: 6X^6
    @tensor(
            env5[-1 -2 -3; -4] :=
            rho[-4 42; 62 41] *
            w[51 12 11; 42] *
            u[22 21; -3 51] *
            h[32 31; 22 21] *
            u'[61 52; 32 31] *
            w'[62; -1 -2 61] * w'[41; 52 12 11]
           )

    # Cost: 2X^8 + 2X^7 + 2X^6
    @tensor(
            env6[-1 -2 -3; -4] :=
            rho[-4 82; 84 81] *
            w[63 61 62; 82] *
            u[52 51; -3 63] *
            h[42 41; 51 61] *
            u'[83 31; 52 42] *
            w'[84; -1 -2 83] * w'[81; 31 41 62]
           )

    env = (env1 + env2 + env3 + env4 + env5 + env6)/3
    # Complex conjugate.
    env = permute(env', (2,3,4), (1,))
    return env
end
