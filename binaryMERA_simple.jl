using TensorFactorizations
using TensorOperations
using NCon
using Printf
using LinearAlgebra
using Logging


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# The data type

mutable struct MERA
    uw_list::Vector{Tuple{Array, Array}}
    top::Array
    depth::Int

    function MERA(uw_list, top)
        m = new(uw_list, top, length(uw_list))
        bonddimension_invar(m)
        return m
    end
end


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Utility functions

function build_u_dg(u)
    u_dg = permutedims(conj(u), (3,4,1,2))
    return u_dg
end

function build_w_dg(w)
    w_dg = permutedims(conj(w), (2,3,1))
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
    bonddimension_invar(m)
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
    bonddimension_invar(m)
    return m
end

function addlayer!(m, u, w, layer)
    insert!(m.uw_list, layer, (u, w))
    m.depth += 1
    bonddimension_invar(m)
    depth_invar(m)
    return m
end

function removelayer!(m, layer)
    deleat!(m.uw_list, layer)
    m.depth -= 1
    bonddimension_invar(m)
    depth_invar(m)
    return m
end

function getbonddimension(m, layer)
    if layer == m.depth+1
        chi = size(m.top)[1]
    elseif layer <= m.depth
        u = m.uw_list[layer][1]
        chi = size(u)[3]
    else
        errmsg = "layer > depth of MERA"
        throw(ArgumentError(errmsg))
    end
    return chi
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

function bonddimension_invar(m)
    uw_list = m.uw_list
    u, w = uw_list[1]
    for i in 2:m.depth
        unext, wnext = uw_list[i]
        if !bonddimension_invar_intralayer(u, w)
            errmsg = "Mismatching bonds in MERA within layer $(i-1)."
            throw(ArgumentError(errmsg))
        end
        if !bonddimension_invar_interlayer(w, unext)
            errmsg = "Mismatching bonds in MERA between layers $(i-1) and $i."
            throw(ArgumentError(errmsg))
        end
        u, w = unext, wnext
    end

    if !bonddimension_invar_intralayer(u, w)
        errmsg = "Mismatching bonds in MERA within layer $(m.depth)."
        throw(ArgumentError(errmsg))
    end

    if !bonddimension_invar_top(w, m.top)
        errmsg = "Mismatching bonds in MERA for the top tensor."
        throw(ArgumentError(errmsg))
    end
    return true
end

function bonddimension_invar_intralayer(u, w)
    matching_bonds = [(size(u)[1], size(w)[3]),
                      (size(u)[2], size(w)[2])]
    allmatch = all([==(pair...) for pair in matching_bonds])
    return allmatch
end

function bonddimension_invar_interlayer(w, unext)
    matching_bonds = [(size(w)[1], size(unext)[3]),
                      (size(w)[1], size(unext)[4])]
    allmatch = all([==(pair...) for pair in matching_bonds])
    return allmatch
end

function bonddimension_invar_top(w, top)
    allmatch = (size(w)[1] == size(top)[1] == size(top)[2] == size(top)[3])
    return allmatch
end


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Functions for creating and replacing tensors

function random_complex_tensor(chi::Integer, rank::Integer)
    chis = fill(chi, (rank,))
    return random_complex_tensor(chis)
end

function random_complex_tensor(chis::Vector{T}) where {T<:Integer}
    real = randn(chis...)
    imag = randn(chis...)
    res = complex.(real, imag)
    return res
end

function randomisometry(chisin::Vector{Int}, chisout::Vector{Int})
    chisall = vcat(chisin, chisout)
    temp = random_complex_tensor(chisall)
    ininds = collect(1:length(chisin))
    outinds = collect(length(chisin)+1:length(chisin)+length(chisout))
    U, S, V = tensorsvd(temp, ininds, outinds)
    ininds *= -1
    outinds *= -1
    push!(ininds, 1)
    pushfirst!(outinds, 1)
    u = ncon((U, V), (ininds, outinds))
    return u
end

function randomtop(chi)
    top = random_complex_tensor(chi, 3)
    top /= norm(top)
    return top
end

function randomlayer(chiin, chiout)
    u = randomisometry([chiin, chiin], [chiin, chiin])
    w = randomisometry([chiout], [chiin, chiin])
    return u, w
end

function addrandomlayer!(m, layer, chiin=getbonddimension(m, layer))
    chiout = getbonddimension(m, layer)
    u, w = randomlayer(chiin, chiout)
    addlayer!(m, u, w, layer)
    return m
end

function randomizelayer!(m, layer)
    chiin = getbonddimension(m, layer)
    chiout = getbonddimension(m, layer+1)
    u, w = randomlayer(chiin, chiout)
    set_uw!(m, u, w, layer)
    return m
end

function randomizetop!(m)
    chi = getbonddimension(m, m.depth+1)
    top = randomtop(chi)
    set_top!(m, top)
    return m
end

function build_random_MERA(chi::Int, layers)
    chis = fill(chi, (layers+1,))
    return build_random_MERA(chis)
end

function build_random_MERA(chis::Vector{Int})
    layers = length(chis)-1
    uw_list = []
    for i in 1:layers
        chi = chis[i]
        chinext = chis[i+1]
        u, w = randomlayer(chi, chinext)
        push!(uw_list, (u, w))
    end
    top = randomtop(chis[length(chis)])
    m = MERA(uw_list, top)
    return m
end


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Ascending and descending superoperators

function binary_ascend_threesite(op, u, w; pos=:avg)
    u_dg = build_u_dg(u)
    w_dg = build_w_dg(w)
    if in(pos, (:left, :l, :L))
        scaled_op = ncon((w, w, w,
                          u, u,
                          op,
                          u_dg, u_dg,
                          w_dg, w_dg, w_dg),
                         ([-100,5,6], [-200,9,8], [-300,16,15],
                          [6,9,1,2], [8,16,10,12],
                          [1,2,10,3,4,14],
                          [3,4,7,13], [14,12,11,17],
                          [5,7,-400], [13,11,-500], [17,15,-600]))
    elseif in(pos, (:right, :r, :R))
        scaled_op = ncon((w, w, w,
                          u, u,
                          op,
                          u_dg, u_dg,
                          w_dg, w_dg, w_dg),
                         ([-100,15,16], [-200,8,9], [-300,6,5],
                          [16,8,12,10], [9,6,1,2],
                          [10,1,2,14,3,4],
                          [12,14,17,11], [3,4,13,7],
                          [15,17,-400], [11,13,-500], [7,5,-600]))
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
        scaled_rho = ncon((u_dg, u_dg,
                           w_dg, w_dg, w_dg,
                           rho,
                           w, w, w,
                           u, u),
                          ([-100,-200,11,12], [-300,103,13,14],
                           [101,11,1], [12,13,2], [14,102,3],
                           [1,2,3,4,5,6],
                           [4,101,21], [5,22,23], [6,24,102],
                           [21,22,-400,-500], [23,24,-600,103]),
                          order=[101,13,23,102,3,6,5,24,2,14,103,1,4,21,22,11,
                                 12])
    elseif in(pos, (:right, :r, :R))
        scaled_rho = ncon((u_dg, u_dg,
                           w_dg, w_dg, w_dg,
                           rho,
                           w, w, w,
                           u, u),
                          ([103,-100,14,13], [-200,-300,12,11],
                           [102,14,3], [13,12,2], [11,101,1],
                           [3,2,1,6,5,4],
                           [6,102,24], [5,23,22], [4,21,101],
                           [24,23,103,-400], [22,21,-500,-600]),
                          order=[101,13,23,102,3,6,5,24,2,14,103,1,4,21,22,11,
                                 12])
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
    rho = ncon((conj(top), top), ([-1,-2,-3], [-11,-12,-13]))
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
            + permutedims(op, [2,3,1,5,6,4])
            + permutedims(op, [3,1,2,6,4,5])
           )/3
    value = scalar(ncon((rho, opavg), ([1,2,3,11,12,13], [11,12,13,1,2,3])))
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
    env1 = ncon((rho,
                 w, w, w,
                 u,
                 h,
                 u_dg, u_dg,
                 w_dg, w_dg, w_dg),
                ([17,18,10,15,14,9],
                 [15,5,6], [14,16,-1], [9,-2,8],
                 [6,16,1,2],
                 [1,2,-3,3,4,13],
                 [3,4,7,12], [13,-4,11,19],
                 [5,7,17], [12,11,18], [19,8,10]))

    env2 = ncon((rho,
                 w, w, w,
                 u,
                 h,
                 u_dg, u_dg,
                 w_dg, w_dg, w_dg),
                ([4,15,6,3,10,5],
                 [3,1,11], [10,9,-1], [5,-2,2],
                 [11,9,12,19],
                 [19,-3,-4,18,7,8],
                 [12,18,13,14], [7,8,16,17],
                 [1,13,4], [14,16,15], [17,2,6]))

    env3 = ncon((rho,
                 w, w, w,
                 u,
                 h,
                 u_dg, u_dg,
                 w_dg, w_dg, w_dg),
                ([6,15,4,5,10,3],
                 [5,2,-1], [10,-2,9], [3,11,1],
                 [9,11,19,12],
                 [-3,-4,19,8,7,18],
                 [8,7,17,16], [18,12,14,13],
                 [2,17,6], [16,14,15], [13,1,4]))

    env4 = ncon((rho,
                 w, w, w,
                 u,
                 h,
                 u_dg, u_dg,
                 w_dg, w_dg, w_dg),
                ([10,18,17,9,14,15],
                 [9,8,-1], [14,-2,16], [15,6,5],
                 [16,6,2,1],
                 [-4,2,1,13,4,3],
                 [-3,13,19,11], [4,3,12,7],
                 [8,19,10], [11,12,18], [7,5,17]))
    env = env1 + env2 + env3 + env4
    U, S, V = tensorsvd(env, [1,2], [3,4])
    u = ncon((conj(U), conj(V)), ([-1,-2,1], [1,-3,-4]))
    return u
end

function minimize_expectation_w(h, u, w, rho)
    w_dg = build_w_dg(w)
    u_dg = build_u_dg(u)
    env1 = ncon((rho,
                 w, w,
                 u, u,
                 h,
                 u_dg, u_dg,
                 w_dg, w_dg, w_dg),
                ([16,15,19,18,17,-1],
                 [18,5,6], [17,9,8],
                 [6,9,2,1], [8,-2,10,11],
                 [2,1,10,4,3,12],
                 [4,3,7,14], [12,11,13,20],
                 [5,7,16], [14,13,15], [20,-3,19]))

    env2 = ncon((rho,
                 w, w,
                 u, u,
                 h,
                 u_dg, u_dg,
                 w_dg, w_dg, w_dg),
                ([18,17,19,16,15,-1],
                 [16,12,13], [15,5,6],
                 [13,5,9,7], [6,-2,2,1],
                 [7,2,1,8,4,3],
                 [9,8,14,11], [4,3,10,20],
                 [12,14,18], [11,10,17], [20,-3,19]))

    env3 = ncon((rho,
                 w, w,
                 u, u,
                 h,
                 u_dg, u_dg,
                 w_dg, w_dg, w_dg),
                ([19,20,15,18,-1,14],
                 [18,5,6], [14,17,13],
                 [6,-2,2,1], [-3,17,12,11],
                 [2,1,12,4,3,9],
                 [4,3,7,10], [9,11,8,16],
                 [5,7,19], [10,8,20], [16,13,15]))

    env4 = ncon((rho,
                 w, w,
                 u, u,
                 h,
                 u_dg, u_dg,
                 w_dg, w_dg, w_dg),
                ([15,20,19,14,-1,18],
                 [14,13,17], [18,6,5],
                 [17,-2,11,12], [-3,6,1,2],
                 [12,1,2,9,3,4],
                 [11,9,16,8], [3,4,10,7],
                 [13,16,15], [8,10,20], [7,5,19]))

    env5 = ncon((rho,
                 w, w,
                 u, u,
                 h,
                 u_dg, u_dg,
                 w_dg, w_dg, w_dg),
                ([19,17,18,-1,15,16],
                 [15,6,5], [16,13,12],
                 [-3,6,1,2], [5,13,7,9],
                 [1,2,7,3,4,8],
                 [3,4,20,10], [8,9,11,14],
                 [-2,20,19], [10,11,17], [14,12,18]))

    env6 = ncon((rho,
                 w, w,
                 u, u,
                 h,
                 u_dg, u_dg,
                 w_dg, w_dg, w_dg),
                ([19,15,16,-1,17,18],
                 [17,8,9], [18,6,5],
                 [-3,8,11,10], [9,6,1,2],
                 [10,1,2,12,3,4],
                 [11,12,20,13], [3,4,14,7],
                 [-2,20,19], [13,14,15], [7,5,16]))

    env = env1 + env2 + env3 + env4 + env5 + env6
    U, S, V = tensorsvd(env, [1], [2,3])
    w = ncon((conj(U), conj(V)), ([-1,1], [1,-2,-3]))
    return w
end

function minimize_expectation_top(h, pars)
    havg = (h
            + permutedims(h, [2,3,1,5,6,4])
            + permutedims(h, [3,1,2,6,4,5])
           )/3
    top = tensoreig(havg, [1,2,3], [4,5,6], hermitian=true)[2][:,:,:,1]
    return top
end

