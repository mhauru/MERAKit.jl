# BinaryLayer and BinaryMERA types, and methods thereof.
# To be `included` in MERA.jl.

# TODO We could parametrise this as BinaryLayer{T1, T2}, disentagler::T1, isometry::T2.
# Would this be good, because it increased type stability, or bad because it caused
# unnecessary recompilation?
struct BinaryLayer <: Layer
    disentangler::TensorMap
    isometry::TensorMap
end

BinaryMERA = GenericMERA{BinaryLayer}

Base.convert(::Type{BinaryLayer}, tuple::Tuple) = BinaryLayer(tuple...)

# Implement the iteration and indexing interfaces.
Base.iterate(layer::BinaryLayer) = (layer.disentangler, 1)
Base.iterate(layer::BinaryLayer, state) = state == 1 ? (layer.isometry, 2) : nothing
Base.eltype(::Type{BinaryLayer}) = TensorMap
Base.length(layer::BinaryLayer) = 2
Base.firstindex(layer::BinaryLayer) = 1
Base.lastindex(layer::BinaryLayer) = 2
function Base.getindex(layer::BinaryLayer, i)
    i == 1 && return layer.disentangler
    i == 2 && return layer.isometry
    throw(BoundsError(layer, i))
end

Base.eltype(layer::BinaryLayer) = reduce(promote_type, map(eltype, layer))

"""
The ratio by which the number of sites changes when go down through this layer.
"""
scalefactor(::Type{BinaryMERA}) = 2

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

causal_cone_width(::Type{BinaryLayer}) = 3

"""
Check the compatibility of the legs connecting the disentanglers and the isometries.
Return true/false.
"""
function space_invar_intralayer(layer::BinaryLayer)
    u, w = layer
    matching_bonds = [(space(u, 1), space(w, 3)'),
                      (space(u, 2), space(w, 2)')]
    allmatch = all([==(pair...) for pair in matching_bonds])
    return allmatch
end

"""
Check the compatibility of the legs connecting the isometries of the first layer to the
disentanglers of the layer above it. Return true/false.
"""
function space_invar_interlayer(layer::BinaryLayer, next_layer::BinaryLayer)
    u, w = layer.disentangler, layer.isometry
    unext, wnext = next_layer.disentangler, next_layer.isometry
    matching_bonds = [(space(w, 1), space(unext, 3)'),
                      (space(w, 1), space(unext, 4)')]
    allmatch = all([==(pair...) for pair in matching_bonds])
    return allmatch
end

inputspace(layer::BinaryLayer) = space(layer.disentangler, 3)
outputspace(layer::BinaryLayer) = space(layer.isometry, 1)

"""
Return a new layer where the isometries have been padded with zeros to change the top vector
space to be V_new.
"""
function expand_outputspace(layer::BinaryLayer, V_new)
    u, w = layer
    w = pad_with_zeros_to(w, 1 => V_new)
    return BinaryLayer(u, w)
end

"""
Return a new layer where the disentanglers and isometries have been padded with zeros to
change the bottom vector space to be V_new.
"""
function expand_inputspace(layer::BinaryLayer, V_new)
    u, w = layer
    u = pad_with_zeros_to(u, 1 => V_new, 2 => V_new, 3 => V_new', 4 => V_new')
    w = pad_with_zeros_to(w, 2 => V_new', 3 => V_new')
    return BinaryLayer(u, w)
end

"""Strip a BinaryLayer of its internal symmetries."""
remove_symmetry(layer::BinaryLayer) = BinaryLayer(map(remove_symmetry, layer)...)

"""
Return a layer with random tensors, with `Vin` and `Vout` as the input and output spaces.
If `random_disentangler=true`, the disentangler is also a random unitary, if `false`
(default), it is the identity.
"""
function randomlayer(::Type{BinaryLayer}, Vin, Vout; random_disentangler=false)
    u = (random_disentangler ?
         randomisometry(Vin ⊗ Vin, Vin ⊗ Vin)
         : identitytensor(Vin ⊗ Vin, Vin ⊗ Vin))
    w = randomisometry(Vout, Vin ⊗ Vin)
    return BinaryLayer(u, w)
end

pseudoserialize(layer::BinaryLayer) = (repr(BinaryLayer), map(pseudoserialize, layer))
depseudoserialize(::Type{BinaryLayer}, data) = BinaryLayer([depseudoserialize(d...)
                                                              for d in data]...)

# # # Ascending and descending superoperators

"""
Ascend a threesite `op` from the bottom of the given layer to the top.
"""
function ascend(op::SquareTensorMap{3}, layer::BinaryLayer, pos=:avg)
    u, w = layer
    u_dg = u'
    w_dg = w'
    if in(pos, (:left, :l, :L))
        @tensor(
                scaled_op[-100,-200,-300,-400,-500,-600] :=
                w[-100,5,6] * w[-200,9,8] * w[-300,16,15] *
                u[6,9,1,2] * u[8,16,10,12] *
                op[1,2,10,3,4,14] *
                u_dg[3,4,7,13] * u_dg[14,12,11,17] *
                w_dg[5,7,-400] * w_dg[13,11,-500] * w_dg[17,15,-600]
               )
    elseif in(pos, (:right, :r, :R))
        @tensor(
                scaled_op[-100,-200,-300,-400,-500,-600] :=
                w[-100,15,16] * w[-200,8,9] * w[-300,6,5] *
                u[16,8,12,10] * u[9,6,1,2] *
                op[10,1,2,14,3,4] *
                u_dg[12,14,17,11] * u_dg[3,4,13,7] *
                w_dg[15,17,-400] * w_dg[11,13,-500] * w_dg[7,5,-600]
               )
    elseif in(pos, (:a, :avg, :average))
        l = ascend(op, layer, :left)
        r = ascend(op, layer, :right)
        scaled_op = (l+r)/2.
    else
        throw(ArgumentError("Unknown position (should be :l, :r, or :avg)."))
    end
    scaled_op = permuteind(scaled_op, (1,2,3), (4,5,6))
    return scaled_op
end


# TODO Would there be a nice way of doing this where I wouldn't have to replicate all the
# network contractions? @ncon could do it, but Jutho's testing says it's significantly
# slower.
"""
Ascend a threesite `op` with an extra free leg from the bottom of the given layer to the
top.
"""
function ascend(op::TensorMap{S1,3,4,S2,T1,T2,T3}, layer::BinaryLayer, pos=:avg) where {S1, S2, T1, T2, T3}
    u, w = layer
    u_dg = u'
    w_dg = w'
    if in(pos, (:left, :l, :L))
        @tensor(
                scaled_op[-100,-200,-300,-400,-500,-600,-1000] :=
                w[-100,5,6] * w[-200,9,8] * w[-300,16,15] *
                u[6,9,1,2] * u[8,16,10,12] *
                op[1,2,10,3,4,14,-1000] *
                u_dg[3,4,7,13] * u_dg[14,12,11,17] *
                w_dg[5,7,-400] * w_dg[13,11,-500] * w_dg[17,15,-600]
               )
    elseif in(pos, (:right, :r, :R))
        @tensor(
                scaled_op[-100,-200,-300,-400,-500,-600,-1000] :=
                w[-100,15,16] * w[-200,8,9] * w[-300,6,5] *
                u[16,8,12,10] * u[9,6,1,2] *
                op[10,1,2,14,3,4,-1000] *
                u_dg[12,14,17,11] * u_dg[3,4,13,7] *
                w_dg[15,17,-400] * w_dg[11,13,-500] * w_dg[7,5,-600]
               )
    elseif in(pos, (:a, :avg, :average))
        l = ascend(op, layer, :left)
        r = ascend(op, layer, :right)
        scaled_op = (l+r)/2.
    else
        throw(ArgumentError("Unknown position (should be :l, :r, or :avg)."))
    end
    scaled_op = permuteind(scaled_op, (1,2,3), (4,5,6,7))
    return scaled_op
end

# TODO Write faster versions that actually do only the necessary contractions.
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
    u_dg = u'
    w_dg = w'
    if in(pos, (:left, :l, :L))
        @tensor(
                scaled_rho[-100,-200,-300,-400,-500,-600] :=
                u_dg[-100,-200,16,17] * u_dg[-300,11,2,10] *
                w_dg[1,16,12] * w_dg[17,2,9] * w_dg[10,4,5] *
                rho[12,9,5,13,7,6] *
                w[13,1,14] * w[7,15,3] * w[6,8,4] *
                u[14,15,-400,-500] * u[3,8,-600,11]
               )
    elseif in(pos, (:right, :r, :R))
        @tensor(
                scaled_rho[-100,-200,-300,-400,-500,-600] :=
                u_dg[11,-100,10,2] * u_dg[-200,-300,17,16] *
                w_dg[4,10,5] * w_dg[2,17,9] * w_dg[16,1,12] *
                rho[5,9,12,6,7,13] *
                w[6,4,8] * w[7,3,15] * w[13,14,1] *
                u[8,3,11,-400] * u[15,14,-500,-600]
               )
    elseif in(pos, (:a, :avg, :average))
        l = descend(rho, layer, :left)
        r = descend(rho, layer, :right)
        scaled_rho = (l+r)/2.
    else
        throw(ArgumentError("Unknown position (should be :l, :r, or :avg)."))
    end
    scaled_rho = permuteind(scaled_rho, (1,2,3), (4,5,6))
    return scaled_rho
end

# # # Optimization

"""
Loop over the tensors of the layer, optimizing each one in turn to minimize the expecation
value of `h`. `rho` is the density matrix right above this layer.
"""
function minimize_expectation_layer(h, layer::BinaryLayer, rho, pars;
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
minimize the expectation of a threesite operator `h`.
"""
function minimize_expectation_disentangler(h::SquareTensorMap{3}, layer::BinaryLayer, rho)
    u, w = layer
    w_dg = w'
    u_dg = u'
    @tensor(
            env1[-1,-2,-3,-4] :=
            rho[17,18,10,15,14,9] *
            w[15,5,6] * w[14,16,-1] * w[9,-2,8] *
            u[6,16,1,2] *
            h[1,2,-3,3,4,13] *
            u_dg[3,4,7,12] * u_dg[13,-4,11,19] *
            w_dg[5,7,17] * w_dg[12,11,18] * w_dg[19,8,10]
           )
                
    @tensor(
            env2[-1,-2,-3,-4] :=
            rho[4,15,6,3,10,5] *
            w[3,1,11] * w[10,9,-1] * w[5,-2,2] *
            u[11,9,12,19] *
            h[19,-3,-4,18,7,8] *
            u_dg[12,18,13,14] * u_dg[7,8,16,17] *
            w_dg[1,13,4] * w_dg[14,16,15] * w_dg[17,2,6]
           )
                
    @tensor(
            env3[-1,-2,-3,-4] :=
            rho[6,15,4,5,10,3] *
            w[5,2,-1] * w[10,-2,9] * w[3,11,1] *
            u[9,11,19,12] *
            h[-3,-4,19,8,7,18] *
            u_dg[8,7,17,16] * u_dg[18,12,14,13] *
            w_dg[2,17,6] * w_dg[16,14,15] * w_dg[13,1,4]
           )

    @tensor(
            env4[-1,-2,-3,-4] :=
            rho[10,18,17,9,14,15] *
            w[9,8,-1] * w[14,-2,16] * w[15,6,5] *
            u[16,6,2,1] *
            h[-4,2,1,13,4,3] *
            u_dg[-3,13,19,11] * u_dg[4,3,12,7] *
            w_dg[8,19,10] * w_dg[11,12,18] * w_dg[7,5,17]
           )

    env = env1 + env2 + env3
    U, S, Vt = svd(env, (1,2), (3,4))
    @tensor u[-1,-2,-3,-4] := conj(U[-1,-2,1]) * conj(Vt[1,-3,-4])
    u = permuteind(u, (1,2), (3,4))
    return BinaryLayer(u, w)
end

# TODO Write faster versions that actually do only the necessary contractions.
function minimize_expectation_disentangler(h::SquareTensorMap{2}, layer::BinaryLayer, rho)
    h = expand_support(h, causal_cone_width(BinaryLayer))
    return minimize_expectation_disentangler(h, layer, rho)
end

function minimize_expectation_disentangler(h::SquareTensorMap{1}, layer::BinaryLayer, rho)
    h = expand_support(h, causal_cone_width(BinaryLayer))
    return minimize_expectation_disentangler(h, layer, rho)
end


"""
Return a new layer, where the isometry has been changed to the locally optimal one to
minimize the expectation of a threesite operator `h`.
"""
function minimize_expectation_isometry(h::SquareTensorMap{3}, layer::BinaryLayer, rho)
    u, w = layer
    w_dg = w'
    u_dg = u'
    @tensor(
            env1[-1,-2,-3] :=
            rho[16,15,19,18,17,-1] *
            w[18,5,6] * w[17,9,8] *
            u[6,9,2,1] * u[8,-2,10,11] *
            h[2,1,10,4,3,12] *
            u_dg[4,3,7,14] * u_dg[12,11,13,20] *
            w_dg[5,7,16] * w_dg[14,13,15] * w_dg[20,-3,19]
           )
                
    @tensor(
            env2[-1,-2,-3] :=
            rho[18,17,19,16,15,-1] *
            w[16,12,13] * w[15,5,6] *
            u[13,5,9,7] * u[6,-2,2,1] *
            h[7,2,1,8,4,3] *
            u_dg[9,8,14,11] * u_dg[4,3,10,20] *
            w_dg[12,14,18] * w_dg[11,10,17] * w_dg[20,-3,19]
           )

    @tensor(
            env3[-1,-2,-3] :=
            rho[19,20,15,18,-1,14] *
            w[18,5,6] * w[14,17,13] *
            u[6,-2,2,1] * u[-3,17,12,11] *
            h[2,1,12,4,3,9] *
            u_dg[4,3,7,10] * u_dg[9,11,8,16] *
            w_dg[5,7,19] * w_dg[10,8,20] * w_dg[16,13,15]
           )

    @tensor(
            env4[-1,-2,-3] :=
            rho[15,20,19,14,-1,18] *
            w[14,13,17] * w[18,6,5] *
            u[17,-2,11,12] * u[-3,6,1,2] *
            h[12,1,2,9,3,4] *
            u_dg[11,9,16,8] * u_dg[3,4,10,7] *
            w_dg[13,16,15] * w_dg[8,10,20] * w_dg[7,5,19]
           )

    @tensor(
            env5[-1,-2,-3] :=
            rho[19,17,18,-1,15,16] *
            w[15,6,5] * w[16,13,12] *
            u[-3,6,1,2] * u[5,13,7,9] *
            h[1,2,7,3,4,8] *
            u_dg[3,4,20,10] * u_dg[8,9,11,14] *
            w_dg[-2,20,19] * w_dg[10,11,17] * w_dg[14,12,18]
           )

    @tensor(
            env6[-1,-2,-3] :=
            rho[19,15,16,-1,17,18] *
            w[17,8,9] * w[18,6,5] *
            u[-3,8,11,10] * u[9,6,1,2] *
            h[10,1,2,12,3,4] *
            u_dg[11,12,20,13] * u_dg[3,4,14,7] *
            w_dg[-2,20,19] * w_dg[13,14,15] * w_dg[7,5,16]
           )

    env = env1 + env2 + env3 + env4 + env5 + env6
    U, S, Vt = svd(env, (1,), (2,3))
    @tensor w[-1,-2,-3] := conj(U[-1,1]) * conj(Vt[1,-2,-3])
    w = permuteind(w, (1,), (2,3))
    return BinaryLayer(u, w)
end

# TODO Write faster versions that actually do only the necessary contractions.
function minimize_expectation_isometry(h::SquareTensorMap{2}, layer::BinaryLayer, rho)
    h = expand_support(h, causal_cone_width(BinaryLayer))
    return minimize_expectation_isometry(h, layer, rho)
end

function minimize_expectation_isometry(h::SquareTensorMap{1}, layer::BinaryLayer, rho)
    h = expand_support(h, causal_cone_width(BinaryLayer))
    return minimize_expectation_isometry(h, layer, rho)
end

