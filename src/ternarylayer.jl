# TernaryLayer and TernaryMERA types, and methods thereof.
# To be `included` in MERA.jl.

# TODO We could parametrise this as TernaryLayer{T1, T2}, disentagler::T1, isometry::T2.
# Would this be good, because it increased type stability, or bad because it caused
# unnecessary recompilation?
struct TernaryLayer <: Layer
    disentangler::TensorMap
    isometry::TensorMap
end

TernaryMERA = GenericMERA{TernaryLayer}

Base.convert(::Type{TernaryLayer}, tuple::Tuple) = TernaryLayer(tuple...)

# Implement the iteration and indexing interfaces.
Base.iterate(layer::TernaryLayer) = (layer.disentangler, 1)
Base.iterate(layer::TernaryLayer, state) = state == 1 ? (layer.isometry, 2) : nothing
Base.eltype(::Type{TernaryLayer}) = TensorMap
Base.length(layer::TernaryLayer) = 2
Base.firstindex(layer::TernaryLayer) = 1
Base.lastindex(layer::TernaryLayer) = 2
function Base.getindex(layer::TernaryLayer, i)
    i == 1 && return layer.disentangler
    i == 2 && return layer.isometry
    throw(BoundsError(layer, i))
end

Base.eltype(layer::TernaryLayer) = reduce(promote_type, map(eltype, layer))

"""
The ratio by which the number of sites changes when go down through this layer.
"""
scalefactor(::Type{TernaryMERA}) = 3

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

causal_cone_width(::Type{TernaryLayer}) = 2

"""
Check the compatibility of the legs connecting the disentanglers and the isometries.
Return true/false.
"""
function space_invar_intralayer(layer::TernaryLayer)
    u, w = layer
    matching_bonds = [(space(u, 1), space(w, 4)'),
                      (space(u, 2), space(w, 2)')]
    allmatch = all([==(pair...) for pair in matching_bonds])
    return allmatch
end

"""
Check the compatibility of the legs connecting the isometries of the first layer to the
disentanglers of the layer above it. Return true/false.
"""
function space_invar_interlayer(layer::TernaryLayer, next_layer::TernaryLayer)
    u, w = layer.disentangler, layer.isometry
    unext, wnext = next_layer.disentangler, next_layer.isometry
    matching_bonds = [(space(w, 1), space(unext, 3)'),
                      (space(w, 1), space(unext, 4)')]
    allmatch = all([==(pair...) for pair in matching_bonds])
    return allmatch
end

inputspace(layer::TernaryLayer) = space(layer.disentangler, 3)
outputspace(layer::TernaryLayer) = space(layer.isometry, 1)

"""
Return a new layer where the isometries have been padded with zeros to change the top vector
space to be V_new.
"""
function expand_outputspace(layer::TernaryLayer, V_new)
    u, w = layer
    w = pad_with_zeros_to(w, 1 => V_new)
    return TernaryLayer(u, w)
end

"""
Return a new layer where the disentanglers and isometries have been padded with zeros to
change the bottom vector space to be V_new.
"""
function expand_inputspace(layer::TernaryLayer, V_new)
    u, w = layer
    u = pad_with_zeros_to(u, 1 => V_new, 2 => V_new, 3 => V_new', 4 => V_new')
    w = pad_with_zeros_to(w, 2 => V_new', 3 => V_new', 4 => V_new')
    return TernaryLayer(u, w)
end

"""
Return a layer with random tensors, with `Vin` and `Vout` as the input and output spaces.
If `random_disentangler=true`, the disentangler is also a random unitary, if `false`
(default), it is the identity.
"""
function randomlayer(::Type{TernaryLayer}, Vin, Vout; random_disentangler=false)
    u = (random_disentangler ?
         randomisometry(Vin ⊗ Vin, Vin ⊗ Vin)
         : identitytensor(Vin ⊗ Vin, Vin ⊗ Vin))
    w = randomisometry(Vout, Vin ⊗ Vin ⊗ Vin)
    return TernaryLayer(u, w)
end

pseudoserialize(layer::TernaryLayer) = (repr(TernaryLayer), map(pseudoserialize, layer))
depseudoserialize(::Type{TernaryLayer}, data) = TernaryLayer([depseudoserialize(d...)
                                                              for d in data]...)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Ascending and descending superoperators

"""
Return the ascending superoperator of the one site in the middle of the isometries in a
TernaryMERA, as a TensorMap. Unlike most ascending superoperators, this one is actually
affordable to construct as a full tensor.
"""
ascending_superop_onesite(m::TernaryMERA) = ascending_superop_onesite(get_layer(m, Inf))

function ascending_superop_onesite(layer::TernaryLayer)
    w = layer.isometry
    w_dg = w'
    @tensor(superop[-1,-2,-11,-12] := w[-1,1,-11,2] * w_dg[1,-12,2,-2])
    return superop
end

"""
Ascend a twosite `op` from the bottom of the given layer to the top.
"""
function ascend(op::SquareTensorMap{2}, layer::TernaryLayer, pos=:avg)
    u, w = layer
    u_dg = u'
    w_dg = w'
    if in(pos, (:left, :l, :L))
        # Cost: 2X^8 + 2X^7 + 2X^6
        @tensor(
                scaled_op[-100,-200,-300,-400] :=
                w[-100,51,52,53] * w[-200,54,11,12] *
                u[53,54,41,42] *
                op[52,41,31,32] *
                u_dg[32,42,21,55] *
                w_dg[51,31,21,-300] * w_dg[55,11,12,-400]
               )
    elseif in(pos, (:right, :r, :R))
        # Cost: 2X^8 + 2X^7 + 2X^6
        @tensor(
                scaled_op[-100,-200,-300,-400] :=
                w[-100,11,12,65] * w[-200,63,61,62] *
                u[65,63,51,52] *
                op[52,61,31,41] *
                u_dg[51,31,64,21] *
                w_dg[11,12,64,-300] * w_dg[21,41,62,-400]
               )
    elseif in(pos, (:middle, :mid, :m, :M))
        # Cost: 6X^6
        @tensor(
                scaled_op[-100,-200,-300,-400] :=
                w[-100,31,32,41] * w[-200,51,21,22] *
                u[41,51,1,2] *
                op[1,2,11,12] *
                u_dg[11,12,42,52] *
                w_dg[31,32,42,-300] * w_dg[52,21,22,-400]
               )
    elseif in(pos, (:a, :avg, :average))
        l = ascend(op, layer, :l)
        r = ascend(op, layer, :r)
        m = ascend(op, layer, :m)
        scaled_op = (l+r+m)/3.
    else
        throw(ArgumentError("Unknown position (should be :m, :l, :r, or :avg)."))
    end
    scaled_op = permuteind(scaled_op, (1,2), (3,4))
    return scaled_op
end

# TODO Would there be a nice way of doing this where I wouldn't have to replicate all the
# network contractions? @ncon could do it, but Jutho's testing says it's significantly
# slower.
"""
Ascend a twosite `op` with an extra free leg from the bottom of the given layer to the top.
"""
function ascend(op::TensorMap{S1,2,3,S2,T1,T2,T3}, layer::TernaryLayer, pos=:avg) where {S1, S2, T1, T2, T3}
    u, w = layer
    u_dg = u'
    w_dg = w'
    if in(pos, (:left, :l, :L))
        # Cost: 2X^8 + 2X^7 + 2X^6
        @tensor(
                scaled_op[-100,-200,-300,-400,-1000] :=
                w[-100,51,52,53] * w[-200,54,11,12] *
                u[53,54,41,42] *
                op[52,41,31,32,-1000] *
                u_dg[32,42,21,55] *
                w_dg[51,31,21,-300] * w_dg[55,11,12,-400]
               )
    elseif in(pos, (:right, :r, :R))
        # Cost: 2X^8 + 2X^7 + 2X^6
        @tensor(
                scaled_op[-100,-200,-300,-400,-1000] :=
                w[-100,11,12,65] * w[-200,63,61,62] *
                u[65,63,51,52] *
                op[52,61,31,41,-1000] *
                u_dg[51,31,64,21] *
                w_dg[11,12,64,-300] * w_dg[21,41,62,-400]
               )
    elseif in(pos, (:middle, :mid, :m, :M))
        # Cost: 6X^6
        @tensor(
                scaled_op[-100,-200,-300,-400,-1000] :=
                w[-100,31,32,41] * w[-200,51,21,22] *
                u[41,51,1,2] *
                op[1,2,11,12,-1000] *
                u_dg[11,12,42,52] *
                w_dg[31,32,42,-300] * w_dg[52,21,22,-400]
               )
    elseif in(pos, (:a, :avg, :average))
        l = ascend(op, layer, :l)
        r = ascend(op, layer, :r)
        m = ascend(op, layer, :m)
        scaled_op = (l+r+m)/3.
    else
        throw(ArgumentError("Unknown position (should be :m, :l, :r, or :avg)."))
    end
    scaled_op = permuteind(scaled_op, (1,2), (3,4,5))
    return scaled_op
end


"""
Decend a twosite `rho` from the top of the given layer to the bottom.
"""
function descend(rho::SquareTensorMap{2}, layer::TernaryLayer, pos=:avg)
    u, w = layer
    u_dg = u'
    w_dg = w'
    if in(pos, (:left, :l, :L))
        # Cost: 2X^8 + 2X^7 + 2X^6
        @tensor(
                scaled_rho[-100,-200,-300,-400] :=
                u_dg[-200,63,61,62] *
                w_dg[52,-100,61,51] * w_dg[62,11,12,21] *
                rho[51,21,42,22] *
                w[42,52,-300,41] * w[22,31,11,12] *
                u[41,31,-400,63]
               )
    elseif in(pos, (:right, :r, :R))
        # Cost: 2X^8 + 2X^7 + 2X^6
        @tensor(
                scaled_rho[-100,-200,-300,-400] :=
                u_dg[63,-100,62,61] *
                w_dg[11,12,62,21] * w_dg[61,-200,52,51] *
                rho[21,51,22,42] *
                w[22,11,12,41] * w[42,31,-400,52] *
                u[41,31,63,-300]
               )
    elseif in(pos, (:middle, :mid, :m, :M))
        # Cost: 6X^6
        @tensor(
                scaled_rho[-100,-200,-300,-400] :=
                u_dg[-100,-200,61,62] *
                w_dg[11,12,61,21] * w_dg[62,31,32,41] *
                rho[21,41,22,42] *
                w[22,11,12,51] * w[42,52,31,32] *
                u[51,52,-300,-400]
               )
    elseif in(pos, (:a, :avg, :average))
        l = descend(rho, layer, :l)
        r = descend(rho, layer, :r)
        m = descend(rho, layer, :m)
        scaled_rho = (l+r+m)/3.
    else
        throw(ArgumentError("Unknown position (should be :m, :l, :r, or :avg)."))
    end
    scaled_rho = permuteind(scaled_rho, (1,2), (3,4))
    return scaled_rho
end

"""
Loop over the tensors of the layer, optimizing each one in turn to minimize the expecation
value of `h`. `rho` is the density matrix right above this layer.
"""
function minimize_expectation_layer(h, layer::TernaryLayer, rho, pars;
                                    do_disentanglers=true)
    for i in 1:pars[:layer_iters]
        if do_disentanglers
            for j in 1:pars[:disentangler_iters]
                layer = minimize_expectation_disentangler(h, layer, rho)
            end
        end
        for j in 1:pars[:isometry_iters]
            layer = minimize_expectation_isometry(h, layer, rho)
        end
    end
    return layer
end

"""
Return a new layer, where the disentangler has been changed to the locally optimal one to
minimize the expectation of a twosite operator `h`.
"""
function minimize_expectation_disentangler(h::SquareTensorMap{2}, layer::TernaryLayer, rho)
    u, w = layer
    w_dg = w'
    u_dg = u'
    # Cost: 2X^8 + 2X^7 + 2X^6
    @tensor(
            env1[-1,-2,-3,-4] :=
            rho[31,21,63,22] *
            w[63,61,62,-1] * w[22,-2,11,12] *
            h[62,-3,51,52] *
            u_dg[52,-4,41,42] *
            w_dg[61,51,41,31] * w_dg[42,11,12,21]
           )

    # Cost: 6X^6
    @tensor(
            env2[-1,-2,-3,-4] :=
            rho[41,51,42,52] *
            w[42,21,22,-1] * w[52,-2,31,32] *
            h[-3,-4,11,12] *
            u_dg[11,12,61,62] *
            w_dg[21,22,61,41] * w_dg[62,31,32,51]
           )

    # Cost: 2X^8 + 2X^7 + 2X^6
    @tensor(
            env3[-1,-2,-3,-4] :=
            rho[21,31,22,63] *
            w[22,12,11,-1] * w[63,-2,62,61] *
            h[-4,62,52,51] *
            u_dg[-3,52,42,41] *
            w_dg[12,11,42,21] * w_dg[41,51,61,31]
           )

    env = env1 + env2 + env3
    U, S, Vt = svd(env, (1,2), (3,4))
    @tensor u[-1,-2,-3,-4] := conj(U[-1,-2,1]) * conj(Vt[1,-3,-4])
    u = permuteind(u, (1,2), (3,4))
    return TernaryLayer(u, w)
end

"""
Return a new layer, where the isometry has been changed to the locally optimal one to
minimize the expectation of a twosite operator `h`.
"""
function minimize_expectation_isometry(h::SquareTensorMap{2}, layer::TernaryLayer, rho)
    u, w = layer
    w_dg = w'
    u_dg = u'
    # Cost: 2X^8 + 2X^7 + 2X^6
    @tensor(
            env1[-1,-2,-3,-4] :=
            rho[81,84,82,-1] *
            w[82,62,61,63] *
            u[63,-2,51,52] *
            h[61,51,41,42] *
            u_dg[42,52,31,83] *
            w_dg[62,41,31,81] * w_dg[83,-3,-4,84]
           )

    # Cost: 6X^6
    @tensor(
            env2[-1,-2,-3,-4] :=
            rho[41,62,42,-1] *
            w[42,11,12,51] *
            u[51,-2,21,22] *
            h[21,22,31,32] *
            u_dg[31,32,52,61] *
            w_dg[11,12,52,41] * w_dg[61,-3,-4,62]
           )

    # Cost: 2X^8 + 2X^7 + 2X^6
    @tensor(
            env3[-1,-2,-3,-4] :=
            rho[31,33,32,-1] *
            w[32,21,11,73] *
            u[73,-2,72,71] *
            h[71,-3,62,61] *
            u_dg[72,62,51,41] *
            w_dg[21,11,51,31] * w_dg[41,61,-4,33]
           )

    # Cost: 2X^8 + 2X^7 + 2X^6
    @tensor(
            env4[-1,-2,-3,-4] :=
            rho[33,31,-1,32] *
            w[32,73,11,21] *
            u[-4,73,71,72] *
            h[-3,71,61,62] *
            u_dg[62,72,41,51] *
            w_dg[-2,61,41,33] * w_dg[51,11,21,31]
           )

    # Cost: 6X^6
    @tensor(
            env5[-1,-2,-3,-4] :=
            rho[62,41,-1,42] *
            w[42,51,12,11] *
            u[-4,51,22,21] *
            h[22,21,32,31] *
            u_dg[32,31,61,52] *
            w_dg[-2,-3,61,62] * w_dg[52,12,11,41]
           )

    # Cost: 2X^8 + 2X^7 + 2X^6
    @tensor(
            env6[-1,-2,-3,-4] :=
            rho[84,81,-1,82] *
            w[82,63,61,62] *
            u[-4,63,52,51] *
            h[51,61,42,41] *
            u_dg[52,42,83,31] *
            w_dg[-2,-3,83,84] * w_dg[31,41,62,81]
           )

    env = env1 + env2 + env3 + env4 + env5 + env6
    U, S, Vt = svd(env, (1,), (2,3,4))
    @tensor w[-1,-2,-3,-4] := conj(U[-1,1]) * conj(Vt[1,-2,-3,-4])
    w = permuteind(w, (1,), (2,3,4))
    return TernaryLayer(u, w)
end

