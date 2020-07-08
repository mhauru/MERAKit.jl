# BinaryLayer and BinaryMERA types, and methods thereof.
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
# Isometry:
#    3|
#  +------+
#  |  w   |
#  +------+
#  1|   2|

struct BinaryLayer{DisType, IsoType} <: SimpleLayer
    disentangler::DisType
    isometry::IsoType
end

BinaryMERA{N} = GenericMERA{N, T} where T <: BinaryLayer

# Given an instance of a type like BinaryLayer{TensorMap, TensorMap, TensorMap},
# return the unparametrised type BinaryLayer.
layertype(::BinaryLayer) = BinaryLayer
layertype(::Type{T}) where T <: BinaryMERA = BinaryLayer

# Implement the iteration and indexing interfaces. Allows things like `u, w = layer`.
Base.iterate(layer::BinaryLayer) = (layer.disentangler, 1)
Base.iterate(layer::BinaryLayer, state) = state == 1 ? (layer.isometry, 2) : nothing
Base.length(layer::BinaryLayer) = 2
Base.firstindex(layer::BinaryLayer) = 1
Base.lastindex(layer::BinaryLayer) = 2
function Base.getindex(layer::BinaryLayer, i)
    i == 1 && return layer.disentangler
    i == 2 && return layer.isometry
    throw(BoundsError(layer, i))
end

"""
The ratio by which the number of sites changes when you go down through this layer.
"""
scalefactor(::Type{<:BinaryLayer}) = 2
scalefactor(::Type{BinaryMERA}) = scalefactor(BinaryLayer)

get_disentangler(m::BinaryMERA, depth) = get_layer(m, depth).disentangler
get_isometry(m::BinaryMERA, depth) = get_layer(m, depth).isometry

function set_disentangler!(m::BinaryMERA, u, depth; kwargs...)
    w = get_isometry(m, depth)
    return set_layer!(m, (u, w), depth; kwargs...)
end

function set_isometry!(m::BinaryMERA, w, depth; kwargs...)
    u = get_disentangler(m, depth)
    return set_layer!(m, (u, w), depth; kwargs...)
end

causal_cone_width(::Type{<:BinaryLayer}) = 3

outputspace(layer::BinaryLayer) = space(layer.disentangler, 1)
inputspace(layer::BinaryLayer) = space(layer.isometry, 3)'
internalspace(layer::BinaryLayer) = space(layer.isometry, 1)
internalspace(m::BinaryMERA, depth) = internalspace(get_layer(m, depth))

"""
Return a new layer where the isometries have been padded with zeros to change the input
(top) vector space to be V_new.
"""
function expand_inputspace(layer::BinaryLayer, V_new)
    u, w = layer
    w = pad_with_zeros_to(w, 3 => V_new')
    return BinaryLayer(u, w)
end

"""
Return a new layer where the disentanglers and isometries have been padded with zeros to
change the output (bottom) vector space to be V_new.
"""
function expand_outputspace(layer::BinaryLayer, V_new)
    u, w = layer
    u = pad_with_zeros_to(u, 1 => V_new, 2 => V_new)
    return BinaryLayer(u, w)
end

"""
Return a new layer where the disentanglers and isometries have been padded with zeros to
change the internal vector space to be V_new.
"""
function expand_internalspace(layer::BinaryLayer, V_new)
    u, w = layer
    u = pad_with_zeros_to(u, 3 => V_new', 4 => V_new')
    w = pad_with_zeros_to(w, 1 => V_new, 2 => V_new)
    return BinaryLayer(u, w)
end

"""
Return a layer with random tensors, with `Vin` and `Vout` as the input and output spaces.
The optionalargument `Vint` is the output bond dimension of the disentangler. If
`random_disentangler=true`, the disentangler is also a random unitary, if `false` (default),
it is the identity or the product of two single-site isometries, depending on if `u` is
supposed to be unitary or isometric. `T` is the data type for the tensors, by default
`ComplexF64`.
"""
function randomlayer(::Type{BinaryLayer}, Vin, Vout, Vint=Vout; random_disentangler=false,
                     T=ComplexF64)
    w = randomisometry(Vint ⊗ Vint, Vin, T)
    u = initialize_disentangler(Vout, Vint, random_disentangler, T)
    return BinaryLayer(u, w)
end

"""
Return the operator that is the fixed point of the average ascending superoperator of this
layer, normalised to have norm 1.
"""
function ascending_fixedpoint(layer::BinaryLayer)
    V = inputspace(layer)
    width = causal_cone_width(typeof(layer))
    Vtotal = reduce(⊗, repeat([V], width))
    eye = id(Vtotal) / sqrt(dim(Vtotal))
    return eye
end

function gradient(layer::BinaryLayer, env::BinaryLayer; isometrymanifold=:grassmann,
                  metric=:euclidean)
    u, w = layer
    uenv, wenv = env
    # The environment is the partial derivative. We need to turn that into a tangent vector
    # of the Stiefel manifold point u or w.
    # The factor of two is from the partial_x + i partial_y derivative of the cost function,
    # and how it depends on both v and v^dagger.
    ugrad = Stiefel.project!(2*uenv, u; metric=metric)
    if isometrymanifold === :stiefel
        wgrad = Stiefel.project!(2*wenv, w; metric=metric)
    elseif isometrymanifold === :grassmann
        wgrad = Grassmann.project!(2*wenv, w)
    else
        msg = "Unknown isometrymanifold $(isometrymanifold)"
        throw(ArgumentError(msg))
    end
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

"""
Check the compatibility of the legs connecting the disentanglers and the isometries.
Return true/false.
"""
function space_invar_intralayer(layer::BinaryLayer)
    u, w = layer
    matching_bonds = [(space(u, 3)', space(w, 2)),
                      (space(u, 4)', space(w, 1))]
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
function space_invar_interlayer(layer::BinaryLayer, next_layer::BinaryLayer)
    u, w = layer.disentangler, layer.isometry
    unext, wnext = next_layer.disentangler, next_layer.isometry
    matching_bonds = [(space(w, 3)', space(unext, 1)),
                      (space(w, 3)', space(unext, 2))]
    allmatch = all([==(pair...) for pair in matching_bonds])
    return allmatch
end

# # # Ascending and descending superoperators

"""
Ascend a threesite `op` from the bottom of the given layer to the top.
"""
function ascend(op::SquareTensorMap{3}, layer::BinaryLayer, pos=:avg)
    u, w = layer
    if in(pos, (:left, :l, :L))
        @tensor(
                scaled_op[-100 -200 -300; -400 -500 -600] :=
                w[5 6; -400] * w[9 8; -500] * w[16 15; -600] *
                u[1 2; 6 9] * u[10 12; 8 16] *
                op[3 4 14; 1 2 10] *
                u'[7 13; 3 4] * u'[11 17; 14 12] *
                w'[-100; 5 7] * w'[-200; 13 11] * w'[-300; 17 15]
               )
    elseif in(pos, (:right, :r, :R))
        @tensor(
                scaled_op[-100 -200 -300; -400 -500 -600] :=
                w[15 16; -400] * w[8 9; -500] * w[6 5; -600] *
                u[12 10; 16 8] * u[1 2; 9 6] *
                op[14 3 4; 10 1 2] *
                u'[17 11; 12 14] * u'[13 7; 3 4] *
                w'[-100; 15 17] * w'[-200; 11 13] * w'[-300; 7 5]
               )
    elseif in(pos, (:a, :avg, :average))
        l = ascend(op, layer, :left)
        r = ascend(op, layer, :right)
        scaled_op = (l+r)/2.
    else
        throw(ArgumentError("Unknown position (should be :l, :r, or :avg)."))
    end
    return scaled_op
end


# TODO Would there be a nice way of doing this where I wouldn't have to replicate all the
# network contractions? @ncon could do it, but Jutho's testing says it's significantly
# slower. This is only used for diagonalizing in charge sectors, so having tensors with
# non-trivial charge would also solve this.
"""
Ascend a threesite `op` with an extra free leg from the bottom of the given layer to the
top.
"""
function ascend(op::AbstractTensorMap{S1,3,4}, layer::BinaryLayer, pos=:avg) where {S1}
    u, w = layer
    if in(pos, (:left, :l, :L))
        @tensor(
                scaled_op[-100 -200 -300; -400 -500 -600 -1000] :=
                w[5 6; -400] * w[9 8; -500] * w[16 15; -600] *
                u[1 2; 6 9] * u[10 12; 8 16] *
                op[3 4 14; 1 2 10 -1000] *
                u'[7 13; 3 4] * u'[11 17; 14 12] *
                w'[-100; 5 7] * w'[-200; 13 11] * w'[-300; 17 15]
               )
    elseif in(pos, (:right, :r, :R))
        @tensor(
                scaled_op[-100 -200 -300; -400 -500 -600 -1000] :=
                w[15 16; -400] * w[8 9; -500] * w[6 5; -600] *
                u[12 10; 16 8] * u[1 2; 9 6] *
                op[14 3 4; 10 1 2 -1000] *
                u'[17 11; 12 14] * u'[13 7; 3 4] *
                w'[-100; 15 17] * w'[-200; 11 13] * w'[-300; 7 5]
               )
    elseif in(pos, (:a, :avg, :average))
        l = ascend(op, layer, :left)
        r = ascend(op, layer, :right)
        scaled_op = (l+r)/2.
    else
        throw(ArgumentError("Unknown position (should be :l, :r, or :avg)."))
    end
    return scaled_op
end

function ascend(op::SquareTensorMap{2}, layer::BinaryLayer, pos=:avg)
    op = expand_support(op, causal_cone_width(BinaryLayer))
    return ascend(op, layer, pos)
end

function ascend(op::SquareTensorMap{1}, layer::BinaryLayer, pos=:avg)
    op = expand_support(op, causal_cone_width(BinaryLayer))
    return ascend(op, layer, pos)
end

"""
Decend a threesite `rho` from the top of the given layer to the bottom.
"""
function descend(rho::SquareTensorMap{3}, layer::BinaryLayer, pos=:avg)
    u, w = layer
    if in(pos, (:left, :l, :L))
        @tensor(
                scaled_rho[-100 -200 -300; -400 -500 -600] :=
                u'[16 17; -400 -500] * u'[2 10; -600 11] *
                w'[12; 1 16] * w'[9; 17 2] * w'[5; 10 4] *
                rho[13 7 6; 12 9 5] *
                w[1 14; 13] * w[15 3; 7] * w[8 4; 6] *
                u[-100 -200; 14 15] * u[-300 11; 3 8]
               )
    elseif in(pos, (:right, :r, :R))
        @tensor(
                scaled_rho[-100 -200 -300; -400 -500 -600] :=
                u'[10 2; 11 -400] * u'[17 16; -500 -600] *
                w'[5; 4 10] * w'[9; 2 17] * w'[12; 16 1] *
                rho[6 7 13; 5 9 12] *
                w[4 8; 6] * w[3 15; 7] * w[14 1; 13] *
                u[11 -100; 8 3] * u[-200 -300; 15 14]
               )
    elseif in(pos, (:a, :avg, :average))
        l = descend(rho, layer, :left)
        r = descend(rho, layer, :right)
        scaled_rho = (l+r)/2.
    else
        throw(ArgumentError("Unknown position (should be :l, :r, or :avg)."))
    end
    return scaled_rho
end

# # # Optimization

"""
Compute the environments of all the tensors in the layer, and return them as a Layer.
"""
function environment(layer::BinaryLayer, op, rho; vary_disentanglers=true)
    if vary_disentanglers
        env_u = environment_disentangler(op, layer, rho)
    else
        env_u = zero(layer.disentangler)
    end
    env_w = environment_isometry(op, layer, rho)
    return BinaryLayer(env_u, env_w)
end

"""
Return a new layer that minimizes the expectation value with respect to the environment
`env`.
"""
function minimize_expectation_ev(layer::BinaryLayer, env::BinaryLayer, pars;
                                 vary_disentanglers=true)
    u = (vary_disentanglers ? projectisometric(env.disentangler; alg=Polar())
         : layer.disentangler)
    w = projectisometric(env.isometry; alg=Polar())
    return BinaryLayer(u, w)
end

"""
Return the environment for a disentangler.
"""
function environment_disentangler(h::SquareTensorMap{3}, layer::BinaryLayer, rho)
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

function environment_disentangler(h::SquareTensorMap{2}, layer::BinaryLayer, rho)
    h = expand_support(h, causal_cone_width(BinaryLayer))
    return environment_disentangler(h, layer, rho)
end

function environment_disentangler(h::SquareTensorMap{1}, layer::BinaryLayer, rho)
    h = expand_support(h, causal_cone_width(BinaryLayer))
    return environment_disentangler(h, layer, rho)
end

"""
Return the environment for the isometry.
"""
function environment_isometry(h::SquareTensorMap{3}, layer, rho)
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

function environment_isometry(h::SquareTensorMap{2}, layer::BinaryLayer, rho)
    h = expand_support(h, causal_cone_width(BinaryLayer))
    return environment_isometry(h, layer, rho)
end

function environment_isometry(h::SquareTensorMap{1}, layer::BinaryLayer, rho)
    h = expand_support(h, causal_cone_width(BinaryLayer))
    return environment_isometry(h, layer, rho)
end

