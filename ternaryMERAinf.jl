module TernaryMERAInf

using TensorKit
using KrylovKit
using Printf
using LinearAlgebra
using Logging

export get_uw, get_u, get_w
export build_rho, build_rhos, build_random_MERA, release_transitionlayer!
export num_translayers, expect, randomlayer!, minimize_expectation!

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# The data type

mutable struct MERA
    uw_list::Vector{Tuple{TensorMap, TensorMap}}

    function MERA(uw_list)
        m = new(uw_list)
        space_invar(m)
        return m
    end
end


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Utility functions

function num_translayers(m)
    return length(m.uw_list)-1
end

function get_uw(m, layer)
    if layer > num_translayers(m)
        return m.uw_list[end]
    else
        return m.uw_list[layer]
    end
end

function get_u(m, layer)
    return get_uw(m, layer)[1]
end

function get_w(m, layer)
    return get_uw(m, layer)[2]
end

function set_uw!(m, u, w, layer)
    if layer > num_translayers(m)
        m.uw_list[end] = (u, w)
    else
        m.uw_list[layer] = (u, w)
    end
    space_invar(m)
    return m
end

function set_u!(m, u, layer)
    return set_uw!(m, u, get_w(m, layer), layer)
end

function set_w!(m, w, layer)
    return set_uw!(m, get_u(m, layer), w, layer)
end

function release_transitionlayer!(m)
    u, w = get_uw(m, Inf)
    u, w = copy(u), copy(w)
    push!(m.uw_list, (u, w))
    return m
end

function expand_bonddim!(m, layer; sector=Trivial())
    # Note that this breaks the isometricity of the MERA. A round of
    # optimization will fix that.
    V = get_outputspace(m, layer)
    V_new = # TODO

    w = get_w(m, layer)
    w = pad_with_zeros_to(w, 1, V)
    set_w!(m, w, layer)

    u_next, w_next = get_uw(m, layer+1)
    u_next = pad_with_zeros_to(u_next, 1, V)
    u_next = pad_with_zeros_to(u_next, 2, V)
    u_next = pad_with_zeros_to(u_next, 3, V)
    u_next = pad_with_zeros_to(u_next, 4, V)
    w_next = pad_with_zeros_to(w_next, 2, V)
    w_next = pad_with_zeros_to(w_next, 3, V)
    w_next = pad_with_zeros_to(w_next, 4, V)
    set_uw!(m, u, w, layer+1)
end

function get_inputspace(m, layer)
    u = get_u(m, layer)
    V = space(u, 3)'
    return V
end

function get_outputspace(m, layer)
    return get_inputspace(m, layer+1)
end

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Invariants

function space_invar(m)
    uw_list = m.uw_list
    u, w = uw_list[1]
    # We go to num_translayers(m)+2, to go a bit into the scale invariant part.
    for i in 2:(num_translayers(m)+1)
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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Functions for creating and replacing tensors

function randomisometry(Vout, Vin)
    temp = TensorMap(rand, ComplexF64, Vout ← Vin)
    U, S, Vt = svd(temp)
    u = U * Vt
    return u
end

function randomlayer(Vin, Vout)
    u = randomisometry(Vin ⊗ Vin, Vin ⊗ Vin)
    w = randomisometry(Vout, Vin ⊗ Vin ⊗ Vin)
    return u, w
end

function randomizelayer!(m, layer)
    Vin = get_inputspace(m, layer)
    Vout = get_outputspace(m, layer)
    u, w = randomlayer(Vin, Vout)
    set_uw!(m, u, w, layer)
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
    m = MERA(uw_list)
    return m
end

function pad_with_zeros(T)
    # TODO
end


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Ascending and descending superoperators

function ternary_ascend_twosite(op, u, w; pos=:avg)
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
        l = ternary_ascend_twosite(op, u, w; pos=:l)
        r = ternary_ascend_twosite(op, u, w; pos=:r)
        m = ternary_ascend_twosite(op, u, w; pos=:m)
        scaled_op = (l+r+m)/3.
    else
        throw(ArgumentError("Unknown position (should be :m, :l, :r, or :avg)."))
    end
    return scaled_op
end


function ternary_descend_twosite(rho, u, w; pos=:avg)
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
        l = ternary_descend_twosite(rho, u, w; pos=:l)
        r = ternary_descend_twosite(rho, u, w; pos=:r)
        m = ternary_descend_twosite(rho, u, w; pos=:m)
        scaled_rho = (l+r+m)/3.
    else
        throw(ArgumentError("Unknown position (should be :m, :l, :r, or :avg)."))
    end
    return scaled_rho
end


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Scaling functions

function asc_twosite(op, mera; endscale=num_translayers(mera)+1, startscale=1)
    if endscale < startscale
        throw(ArgumentError("endscale < startscale"))
    elseif endscale > startscale
        op = asc_twosite(op, mera; endscale=endscale-1, startscale=startscale)
        u, w = get_uw(mera, endscale-1)
        op = ternary_ascend_twosite(op, u, w; pos=:avg)
    end
    return op
end

function desc_twosite(op, mera; endscale=1, startscale=num_translayers(mera)+1)
    if endscale > startscale
        throw(ArgumentError("endscale > startscale"))
    elseif endscale < startscale
        op = desc_twosite(op, mera; endscale=endscale+1, startscale=startscale)
        u, w = get_uw(mera, endscale)
        op = ternary_descend_twosite(op, u, w; pos=:avg)
    end
    return op
end

function build_fixedpoint_rho(mera)
    f(x) = desc_twosite(x, mera; endscale=num_translayers(mera)+1,
                        startscale=num_translayers(mera)+2)
    V = get_outputspace(mera, Inf)
    typ = eltype(get_u(mera, Inf))
    eye = TensorMap(I, typ, V ← V)
    @tensor x0[-1,-2,-11,-12] := eye[-1,-11] * eye[-2,-12]
    vals, vecs, info = eigsolve(f, x0)
    rho = vecs[1]
    # rho is Hermitian only up to a phase. Divide out that phase.
    # TODO Remove this workaround for taking a trace.
    eye = TensorMap(I, typ, V ← V)
    @tensor tr[] := rho[1,2,3,4] * eye[3,1] * eye[4,2]
    rho /= scalar(tr)
    return rho
end

function build_rho(mera, layer)
    rho = build_fixedpoint_rho(mera)
    if layer < num_translayers(mera)+1
        rho = desc_twosite(rho, mera; endscale=layer,
                           startscale=num_translayers(mera)+1)
    end
    return rho
end

function build_rhos(m, lowest_to_generate=1)
    rho = build_fixedpoint_rho(m)
    rhos = [rho]
    for l in num_translayers(m):-1:lowest_to_generate
        rho = desc_twosite(rho, m, endscale=l, startscale=l+1)
        push!(rhos, rho)
    end
    rhos = reverse(rhos)
    return rhos
end


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Evaluation

function expect(op, mera; opscale=1, evalscale=num_translayers(mera)+1)
    rho = build_rho(mera, evalscale)
    op = asc_twosite(op, mera; startscale=opscale, endscale=evalscale)
    @tensor value_tens[] := rho[1,2,11,12] * op[11,12,1,2]
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
    println("Optimizing a MERA with $(num_translayers(m)+1) layers,"*
            " keeping the lowest $(lowest_to_optimize-1) fixed.")
    nt = num_translayers(m)
    horig = asc_twosite(h, m; endscale=lowest_to_optimize)
    energy = Inf
    energy_change = Inf
    counter = 0
    while (abs(energy_change) > pars[:energy_delta]
           && counter < pars[:energy_maxiter])
        h = horig
        rhos = build_rhos(m, lowest_to_optimize+1)

        counter += 1
        oldenergy = energy
        for l in lowest_to_optimize:nt
            rho = rhos[l-lowest_to_optimize+1]
            u, w = get_uw(m, l)
            u, w = minimize_expectation_uw(h, u, w, rho, pars)
            set_uw!(m, u, w, l)
            h = asc_twosite(h, m; startscale=l, endscale=l+1)
        end

        # Special case of the translation invariant layer.
        havg = h
        hi = h
        for i in 1:pars[:havg_depth]
            hi = asc_twosite(hi, m; startscale=nt+i, endscale=nt+i+1)
            hi = hi/3
            havg = havg + hi
        end
        u, w = get_uw(m, Inf)
        u, w = minimize_expectation_uw(havg, u, w, rhos[end], pars)
        set_uw!(m, u, w, Inf)

        energy = expect(h, m, opscale=nt+1, evalscale=nt+1)
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
    return u
end

function minimize_expectation_w(h, u, w, rho)
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
    return w
end

end  # module
