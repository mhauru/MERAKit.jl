using TensorKit
using KrylovKit
using Printf
using LinearAlgebra
using Logging


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# The data type

mutable struct MERA
    uw_list::Vector{Tuple{TensorMap, TensorMap}}
    top::Tensor
    depth::Int

    function MERA(uw_list, top)
        m = new(uw_list, top, length(uw_list))
        space_invar(m)
        return m
    end
end


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Utility functions

# TODO Could we remove these functions?
function build_u_dg(u)
    u_dg = u'
    return u_dg
end

# TODO Could we remove these functions?
function build_w_dg(w)
    w_dg = w'
    return w_dg
end

function get_uw(m, layer)
    return m.uw_list[layer]
end

function get_u(m, layer)
    return get_uw(m, layer)[1]
end

function get_w(m, layer)
    return get_uw(m, layer)[2]
end

function get_top(m)
    return m.top
end

function set_uw!(m, u, w, layer)
    m.uw_list[layer] = (u, w)
    space_invar(m)
    return m
end

function set_u!(m, u, layer)
    return set_uw!(m, u, m.uw_list[layer][2], layer)
end

function set_w!(m, w, layer)
    return set_uw!(m, m.uw_list[layer][1], w, layer)
end

function set_top!(m, top)
    m.top = top
    space_invar(m)
    return m
end

function addlayer!(m, u, w, layer)
    insert!(m.uw_list, layer, (u, w))
    m.depth += 1
    space_invar(m)
    depth_invar(m)
    return m
end

function removelayer!(m, layer)
    deleat!(m.uw_list, layer)
    m.depth -= 1
    space_invar(m)
    depth_invar(m)
    return m
end

function getspace(m, layer)
    if layer == m.depth+1
        V = space(m.top, 1)'
    elseif layer <= m.depth
        u = m.uw_list[layer][1]
        V = space(u, 3)'
    else
        errmsg = "layer > depth of MERA"
        throw(ArgumentError(errmsg))
    end
    return V
end


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Invariants

function depth_invar(m)
    if m.depth != length(m.uw_list)
        errmsg = "Depth of MERA does not match number of layers."
        throw(ArgumentError(errmsg))
    end
    return true
end

function space_invar(m)
    uw_list = m.uw_list
    u, w = uw_list[1]
    for i in 2:m.depth
        unext, wnext = uw_list[i]
        if !space_invar_intralayer(u, w)
            errmsg = "Mismatching bonds in MERA within layer $(i-1)."
            throw(ArgumentError(errmsg))
        end
        if !space_invar_interlayer(w, unext)
            errmsg = "Mismatching bonds in MERA between layers $(i-1) and $i."
            throw(ArgumentError(errmsg))
        end
        u, w = unext, wnext
    end

    if !space_invar_intralayer(u, w)
        errmsg = "Mismatching bonds in MERA within layer $(m.depth)."
        throw(ArgumentError(errmsg))
    end

    if !space_invar_top(w, m.top)
        errmsg = "Mismatching bonds in MERA for the top tensor."
        throw(ArgumentError(errmsg))
    end
    return true
end

function space_invar_intralayer(u, w)
    matching_bonds = [(space(u, 1), space(w, 3)'),
                      (space(u, 2), space(w, 2)')]
    allmatch = all([==(pair...) for pair in matching_bonds])
    return allmatch
end

function space_invar_interlayer(w, unext)
    matching_bonds = [(space(w, 1), space(unext, 3)'),
                      (space(w, 1), space(unext, 4)')]
    allmatch = all([==(pair...) for pair in matching_bonds])
    return allmatch
end

function space_invar_top(w, top)
    allmatch = (space(w, 1)' == space(top, 1) == space(top, 2) == space(top, 3))
    return allmatch
end


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Functions for creating and replacing tensors

function randomisometry(Vout, Vin)
    temp = TensorMap(rand, ComplexF64, Vout ← Vin)
    U, S, Vt = svd(temp)
    u = U * Vt
    return u
end

function randomtop(V)
    top = Tensor(rand, ComplexF64, V' ⊗ V' ⊗ V')
    top /= norm(top)
    return top
end

function randomlayer(Vin, Vout)
    u = randomisometry(Vin ⊗ Vin, Vin ⊗ Vin)
    w = randomisometry(Vout, Vin ⊗ Vin)
    return u, w
end

function addrandomlayer!(m, layer, Vin=getspace(m, layer))
    Vout = getspace(m, layer)
    u, w = randomlayer(Vin, Vout)
    addlayer!(m, u, w, layer)
    return m
end

function randomizelayer!(m, layer)
    Vin = getspace(m, layer)
    Vout = getspace(m, layer+1)
    u, w = randomlayer(Vin, Vout)
    set_uw!(m, u, w, layer)
    return m
end

function randomizetop!(m)
    V = getspace(m, m.depth+1)
    top = randomtop(V)
    set_top!(m, top)
    return m
end

function build_random_MERA(V, layers)
    Vs = repeat([V], layers+1)
    return build_random_MERA(Vs)
end

function build_random_MERA(Vs)
    layers = length(Vs)-1
    uw_list = []
    for i in 1:layers
        V = Vs[i]
        Vnext = Vs[i+1]
        u, w = randomlayer(V, Vnext)
        push!(uw_list, (u, w))
    end
    top = randomtop(Vs[length(Vs)])
    m = MERA(uw_list, top)
    return m
end


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Ascending and descending superoperators

function binary_ascend_threesite(op, u, w; pos=:avg)
    u_dg = build_u_dg(u)
    w_dg = build_w_dg(w)
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
        l = binary_ascend_threesite(op, u, w; pos=:l)
        r = binary_ascend_threesite(op, u, w; pos=:r)
        scaled_op = (l+r)/2.
    else
        throw(ArgumentError("Unknown position (should be l, r or avg)."))
    end
    return scaled_op
end


function binary_descend_threesite(rho, u, w; pos=:avg)
    u_dg = build_u_dg(u)
    w_dg = build_w_dg(w)
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
        l = binary_descend_threesite(rho, u, w; pos=:l)
        r = binary_descend_threesite(rho, u, w; pos=:r)
        scaled_rho = (l+r)/2.
    else
        throw(ArgumentError("Unknown position (should be :l, :r or :avg)."))
    end
    return scaled_rho
end


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Scaling functions

function asc_threesite(op, mera; endscale=mera.depth, startscale=1)
    if endscale < startscale
        throw(ArgumentError("endscale < startscale"))
    elseif endscale > startscale
        op = asc_threesite(op, mera; endscale=endscale-1,
                           startscale=startscale)
        u, w = mera.uw_list[endscale-1]
        op = binary_ascend_threesite(op, u, w; pos=:avg)
    end
    return op
end

function desc_threesite(op, mera; endscale=1,
                        startscale=mera.depth+1)
    if endscale > startscale
        throw(ArgumentError("endscale > startscale"))
    elseif endscale < startscale
        op = desc_threesite(op, mera; endscale=endscale+1,
                            startscale=startscale)
        u, w = mera.uw_list[endscale]
        op = binary_descend_threesite(op, u, w; pos=:avg)
    end
    return op
end

function build_rho(mera, scale)
    depth = mera.depth
    top = mera.top
    @tensor rho[-1,-2,-3,-11,-12,-13] := conj(top[-1,-2,-3]) * top[-11,-12,-13]
    rho = desc_threesite(rho, mera; endscale=scale, startscale=depth+1)
    return rho
end

function buildrhos(m, lowest_to_generate=1)
    rhos = []
    rho = build_rho(m, m.depth+1)
    for l in m.depth:-1:lowest_to_generate
        push!(rhos, rho)
        rho = desc_threesite(rho, m, endscale=l, startscale=l+1)
    end
    rhos = reverse(rhos)
    return rhos
end


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Evaluation

function expect(op, mera; opscale=1, evalscale=mera.depth+1)
    rho = build_rho(mera, evalscale)
    op = asc_threesite(op, mera; startscale=opscale, endscale=evalscale)
    # TODO this averaging shouldn't be hardcoded.
    opavg = (op
            + permuteind(op, (2,3,1,5,6,4))
            + permuteind(op, (3,1,2,6,4,5))
           )/3
    @tensor value_tens[] := rho[1,2,3,11,12,13] * opavg[11,12,13,1,2,3]
    # TODO Probably shouldn't have to explicitly mention TensorKit.
    value = scalar(value_tens)
    if abs(imag(value)/value) > 1e-13
        @warn("Non-real expectation value: $value")
    end
    value = real(value)
    return value
end


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Optimization

function minimize_expectation!(m, h, pars; lowest_to_optimize=1,
                               normalization=identity)
    println("Optimizing a MERA with $(m.depth) layers,"*
            " keeping the lowest $(lowest_to_optimize-1) fixed.")
    horig = asc_threesite(h, m; endscale=lowest_to_optimize)
    energy = Inf
    energy_change = Inf
    counter = 0
    while (abs(energy_change) > pars[:energy_delta]
           && counter < pars[:energy_maxiter])
        h = horig
        rhos = buildrhos(m, lowest_to_optimize)

        counter += 1
        oldenergy = energy
        for l in lowest_to_optimize:m.depth
            rho = rhos[l-lowest_to_optimize+1]
            u, w = get_uw(m, l)
            u, w = minimize_expectation_uw(h, u, w, rho, pars)
            set_uw!(m, u, w, l)
            h = asc_threesite(h, m; startscale=l, endscale=l+1)
        end
        top = minimize_expectation_top(h, pars)
        set_top!(m, top)
        energy = expect(h, m, opscale=m.depth+1, evalscale=m.depth+1)
        energy = normalization(energy)
        energy_change = (energy - oldenergy)/energy
        @printf("Energy = %.5e,    change = %.5e,    counter = %d\n",
                energy, energy_change, counter)
    end
    return m
end

function minimize_expectation_uw(h, u, w, rho, pars)
    for i in 1:pars[:uw_iters]
        for j in 1:pars[:u_iters]
            u = minimize_expectation_u(h, u, w, rho)
        end
        for j in 1:pars[:w_iters]
            w = minimize_expectation_w(h, u, w, rho)
        end
    end
    return u, w
end

function minimize_expectation_u(h, u, w, rho)
    w_dg = build_w_dg(w)
    u_dg = build_u_dg(u)
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

    env = env1 + env2 + env3 + env4
    U, S, Vt = svd(env, (1,2), (3,4))
    @tensor u[-1,-2,-3,-4] := conj(U[-1,-2,1]) * conj(Vt[1,-3,-4])
    u = permuteind(u, (1,2), (3,4))
    return u
end

function minimize_expectation_w(h, u, w, rho)
    w_dg = build_w_dg(w)
    u_dg = build_u_dg(u)
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
    return w
end

function minimize_expectation_top(h, pars)
    havg = (h
            + permuteind(h, (2,3,1,5,6,4))
            + permuteind(h, (3,1,2,6,4,5))
           )/3
    top = eigsolve(v -> @tensor(res[-1,-2,-3]
                                := havg[1,2,3,-1,-2,-3] * v[1,2,3]),
                   Tensor(randn, ComplexF64, space(havg, (1,2,3))');
                   ishermitian=true
                  )[2][1]
    return top
end

