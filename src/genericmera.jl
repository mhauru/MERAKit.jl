# The most important types of the package, GenericMERA and Layer, on which all specific MERA
# implementations (binary, ternary, ...) are built. Methods and functions that can be
# implemented on this level of abstraction, without having to know the details of the
# specific MERA type.
# To be included in MERA.jl.

"""
Layer is an abstract supertype of concrete types such as BinaryLayer. A typical Layer is a
collection of tensors, the orders and shapes of which depend on the type.
"""
abstract type Layer end

"""
A GenericMERA is a collection of Layers. The type of these layers then determines whether
the MERA is binary, ternary, etc.

A few notes on conventions and terminology:
- The physical indices of the MERA are at the "bottom", the scale invariant part at the
"top".
- The counting of layers starts from the bottom, so the layer with physical indices is layer
#1. The last layer is the scale invariant one, that then repeats upwards to infinity.
- Each layer is thought of as a linear map from its top, or input space to its bottom, or
output space.
"""
struct GenericMERA{T}
    layers::Vector{T}
    # Prevent the creation of GenericMERA{T} objects when T is not a subtype of Layer.
    GenericMERA{T}(layers) where T <: Layer = new(layers)
end

# # # Basic utility functions

"""
Return the type of the layers of `m`.
"""
layertype(m::T) where {T <: GenericMERA} = layertype(T)
layertype(::Type{GenericMERA{T}}) where {T <: Layer} = T

Base.eltype(m::GenericMERA) = reduce(promote_type, map(eltype, m.layers))

"""
The ratio by which the number of sites changes when go down by a layer.
"""
scalefactor(::Type{GenericMERA{T}}) where T = scalefactor(T)

"""
Each MERA has a stable width causal cone, that depends on the type of layers the MERA has.
Return that width.
"""
causal_cone_width(::Type{GenericMERA{T}}) where T <: Layer = causal_cone_width(T)

"""
Return the number of transition layers, i.e. layers below the scale invariant one, in the
MERA.
"""
num_translayers(m::GenericMERA) = length(m.layers)-1

"""
Return the layer at the given depth. 1 is the lowest layer, i.e. the one with physical
indices.
"""
get_layer(m::GenericMERA, depth) = (depth > num_translayers(m) ?
                                    m.layers[end] : m.layers[depth])

"""
Set, in-place, one of the layers of the MERA to a new, given value. If check_invar=true,
check that the indices match afterwards.
"""
function set_layer!(m::GenericMERA, layer, depth; check_invar=true)
    depth > num_translayers(m) ? (m.layers[end] = layer) : (m.layers[depth] = layer)
    check_invar && space_invar(m)
    return m
end

"""
Add one more transition layer at the top of the MERA, by taking the lowest of the scale
invariant ones and releasing it to vary independently.
"""
function release_transitionlayer!(m::GenericMERA)
    layer = get_layer(m, Inf)
    layer = copy(layer)
    push!(m.layers, layer)
    return m
end

"""
Given a MERA and a depth, return the vector space of the downwards-pointing (towards the
physical level) indices of the layer at that depth.
"""
outputspace(m::GenericMERA, depth) = outputspace(get_layer(m, depth))

"""
Given a MERA and a depth, return the vector space of the upwards-pointing (towards scale
invariance) indices of the layer at that depth.
"""
inputspace(m::GenericMERA, depth) = inputspace(get_layer(m, depth))

"""
Compute the von Neumann entropy of a density matrix `rho`.
"""
function densitymatrix_entropy(rho)
    eigs = eigh(rho)[1]
    eigs = real.(diag(convert(Array, eigs)))
    if sum(abs.(eigs[eigs .<= 0.])) > 1e-13
        warn("Significant negative eigenvalues for a density matrix: $eigs")
    end
    eigs = eigs[eigs .> 0.]
    S = -dot(eigs, log.(eigs))
    return S
end

densitymatrix_entropies(m::GenericMERA) = map(densitymatrix_entropy, densitymatrices(m))

# # # Generating random MERAs

"""
Replace the layer at the given depth with a random one. Additional keyword arguments are
passed to the function that generates the random layer, the details of which depend on the
specific layer type (binary, ternary, ...).
"""
function randomizelayer!(m::GenericMERA{T}, depth; kwargs...) where T
    Vin = inputspace(m, depth)
    Vout = outputspace(m, depth)
    layer = randomlayer(T, Vin, Vout; kwargs...)
    set_layer!(m, layer, depth)
    return m
end

"""
Generate a random MERA of type `T`, with `V` as the vector space of all layers, and with
`num_layers` as the number of different layers (including the scale invariant one).
Additional keyword arguments are passed on to the function that generates a single random
layer.
"""
function random_MERA(::Type{T}, V, num_layers; kwargs...) where T <: GenericMERA
    Vs = repeat([V], num_layers+1)
    return random_MERA(T, Vs; kwargs...)
end

"""
Generate a random MERA of type `T`, with `Vs` as the vector spaces of the various layers.
Number of layers will be the length of `Vs`, the `Vs[1]` will be the physical index space,
and `Vs[end]` will be the one at the scale invariant layer.  Additional keyword arguments
are passed on to the function that generates a single random layer.
"""
function random_MERA(::Type{T}, Vs; kwargs...) where T <: GenericMERA
    num_layers = length(Vs)
    layers = []
    for i in 1:num_layers
        V = Vs[i]
        Vnext = (i < num_layers ? Vs[i+1] : V)
        layer = randomlayer(layertype(T), Vnext, V; kwargs...)
        push!(layers, layer)
    end
    m = T(layers)
    return m
end

"""
Return a MERA layer with random tensors. The first argument is the Layer type, and the
second and third are the input and output spaces. May take further keyword arguments
depending on the Layer type. Each subtype of Layer should have its own method for this
function.
"""
function randomlayer end

"""
Expand the bond dimension of the MERA at the given depth. `depth=1` is the first virtual
level, just above the first layer of the MERA, and the numbering grows from there. The new
bond dimension is given by `newdims`, which for a non-symmetric MERA is just a number, and
for a symmetric MERA is a dictionary of {irrep => block dimension}. Not all irreps for a
bond need to be listed, the once left out are left untouched.

The expansion is done by padding tensors with zeros. Note that this breaks isometricity of
the individual tensors. This is however of no consequence, since the MERA as a state remains
exactly the same. A round of optimization on the MERA will restore isometricity of each
tensor.
"""
function expand_bonddim!(m::GenericMERA, depth, newdims)
    V = inputspace(m, depth)
    V = expand_vectorspace(V, newdims)

    layer = get_layer(m, depth)
    layer = expand_inputspace(layer, V)
    set_layer!(m, layer, depth; check_invar=false)

    next_layer = get_layer(m, depth+1)
    next_layer = expand_outputspace(next_layer, V)
    if depth == num_translayers(m)
        # next_layer is the scale invariant part, so we need to change its top
        # index too since we changed the bottom.
        next_layer = expand_inputspace(next_layer, V)
    end
    set_layer!(m, next_layer, depth+1; check_invar=true)
end

"""
Return a new layer where the tensors have been padded with zeros as necessary to change the
input space. The first argument is the layer, the second one is the new input space. Each
subtype of Layer should have its own method for this function.
"""
function expand_inputspace end

"""
Return a new layer where the tensors have been padded with zeros as necessary to change the
output space. The first argument is the layer, the second one is the new output space. Each
subtype of Layer should have its own method for this function.
"""
function expand_outputspace end

"""
Given a MERA which may possibly be built of symmetry preserving TensorMaps, return another
MERA that has the symmetry structure stripped from it, and all tensors are dense.
"""
remove_symmetry(m::T) where T <: GenericMERA = T(map(remove_symmetry, m.layers))

# # # Pseudo(de)serialization
# "Pseudo(de)serialization" refers to breaking the MERA down into types in Julia Base, and
# constructing it back. This can be used for storing MERAs on disk.
# TODO Once JLD or JLD2 works properly we should be able to get rid of this.
"""
Return a tuple of objects that can be used to reconstruct a given MERA, and that are all of
Julia base types.
"""
pseudoserialize(m::T) where T <: GenericMERA = (repr(T), map(pseudoserialize, m.layers))

"""
Reconstruct a MERA given the output of `pseudoserialize`.
"""
depseudoserialize(::Type{T}, args) where T <: GenericMERA = T([depseudoserialize(d...)
                                                               for d in args])

# Implementations for pseudoserialize(l::T) and depseudoserialize(::Type{T}, args...) should
# be written for each T <: Layer. They can make use of the (de)pseudoserialize methods for
# TensorMaps from `tensortools.jl`.

# # # Invariants

"""
Check that the indices of the various tensors in a given MERA are compatible with each
other. If not, throw an ArgumentError. If yes, return true. This relies on two checks,
`space_invar_intralayer` for checking indices within a layer and `space_invar_interlayer`
for checking indices between layers. These should be implemented for each subtype of Layer.
"""
function space_invar(m::GenericMERA{T}) where T
    layer = get_layer(m, 1)
    # We go to num_translayers(m)+2, to check that the scale invariant layer is consistent
    # with itself.
    for i in 2:(num_translayers(m)+2)
        next_layer = get_layer(m, i)
        if applicable(space_invar_intralayer, layer)
            if !space_invar_intralayer(layer)
                errmsg = "Mismatching bonds in MERA within layer $(i-1)."
                throw(ArgumentError(errmsg))
            end
        else
            msg = "space_invar_intralayer has no method for type $(T). We recommend writing one, to enable checking for space mismatches when assigning tensors."
            @warn(msg)
        end

        if applicable(space_invar_interlayer, layer, next_layer)
            if !space_invar_interlayer(layer, next_layer)
                errmsg = "Mismatching bonds in MERA between layers $(i-1) and $i."
                throw(ArgumentError(errmsg))
            end
        else
            msg = "space_invar_interlayer has no method for type $(T). We recommend writing one, to enable checking for space mismatches when assigning tensors."
            @warn(msg)
        end
        layer = next_layer
    end
    return true
end

"""
Given a Layer, return true if the indices within the layer are compatible with each other,
false otherwise.

This method should be implemented separately for each subtype of Layer.
"""
function space_invar_intralayer end

"""
Given two Layers, return true if the indices between them are compatible with each other,
false otherwise. The first argument is the layer below the second one.

This method should be implemented separately for each subtype of Layer.
"""
function space_invar_interlayer end

# # # Scaling functions

"""
Ascend the operator `op`, that lives on the layer `startscale`, through the MERA to the
layer `endscale`. Living "on" a layer means living on the indices right below it, so
`startscale=1` (the default) refers to the physical indices, and
`endscale=num_translayers(m)+1` (the default) to the indices just below the first scale
invariant layer.
"""
function ascend(op, m::GenericMERA; endscale=num_translayers(m)+1, startscale=1)
    if endscale < startscale
        throw(ArgumentError("endscale < startscale"))
    elseif endscale > startscale
        op = ascend(op, m; endscale=endscale-1, startscale=startscale)
        layer = get_layer(m, endscale-1)
        op = ascend(op, layer)
    end
    return op
end

"""
Descend the operator `op`, that lives on the layer `startscale`, through the MERA to the
layer `endscale`. Living "on" a layer means living on the indices right below it, so
`endscale=1` (the default) refers to the physical indices, and
`startscale=num_translayers(m)+1` (the default) to the indices just below the first scale
invariant layer.
"""
function descend(op, m::GenericMERA; endscale=1, startscale=num_translayers(m)+1)
    if endscale > startscale
        throw(ArgumentError("endscale > startscale"))
    elseif endscale < startscale
        op = descend(op, m; endscale=endscale+1, startscale=startscale)
        layer = get_layer(m, endscale)
        op = descend(op, layer)
    end
    return op
end

"""
Find the fixed point density matrix of the scale invariant part of the MERA.
"""
function fixedpoint_densitymatrix(m::T) where T <: GenericMERA
    f(x) = descend(x, m; endscale=num_translayers(m)+1, startscale=num_translayers(m)+2)
    V = inputspace(m, Inf)
    width = causal_cone_width(T)
    eye = id(V)
    x0 = ⊗(repeat([eye], width)...)
    vals, vecs, info = eigsolve(f, x0)
    rho = vecs[1]
    # rho is Hermitian only up to a phase. Divide out that phase.
    rho /= tr(rho)
    return rho
end

"""
Return the density matrix right below the layer at `depth`.
"""
function densitymatrix(m::GenericMERA, depth)
    rho = fixedpoint_densitymatrix(m)
    if depth < num_translayers(m)+1
        rho = descend(rho, m; endscale=depth, startscale=num_translayers(m)+1)
    end
    return rho
end

"""
Return the density matrices starting for the layers from `lowest_depth` upwards up to and
including the scale invariant one.
"""
function densitymatrices(m::GenericMERA, lowest_depth=1)
    rho = fixedpoint_densitymatrix(m)
    rhos = [rho]
    for l in num_translayers(m):-1:lowest_depth
        rho = descend(rho, m, endscale=l, startscale=l+1)
        push!(rhos, rho)
    end
    rhos = reverse(rhos)
    return rhos
end


# # # Extracting CFT data

"""
Diagonalize the scale invariant ascending superoperator to compute the scaling dimensions of
the underlying CFT. The return value is a dictionary, the keys of which are symmetry sectors
for a possible internal symmetry of the MERA (Trivial() if there is no internal symmetry),
and values are scaling dimensions in this symmetry sector.
"""
function scalingdimensions(m::GenericMERA; howmany=20)
    V = inputspace(m, Inf)
    typ = eltype(m)
    chi = dim(V)
    width = causal_cone_width(typeof(m))
    # Don't even try to get more than half of the eigenvalues. Its too expensive, and they
    # are garbage anyway.
    maxmany = Int(ceil(chi^width/2))
    howmany = min(maxmany, howmany)
    # Define a function that takes an operator and ascends it once through the scale
    # invariant layer.
    nm = num_translayers(m)
    f(x) = ascend(x, m; endscale=nm+2, startscale=nm+1)
    # Find out which symmetry sectors we should do the diagonalization in.
    interlayer_space = reduce(⊗, repeat([V], width))
    sects = sectors(fuse(interlayer_space))
    scaldim_dict = Dict()
    for irrep in sects
        # Diagonalize in each irrep sector.
        inspace = interlayer_space
        outspace = interlayer_space
        # If this is a non-trivial irrep sector, expand the input space with a dummy leg.
        irrep !== Trivial() && (inspace = inspace ⊗ typeof(V)(irrep => 1))
        # The initial guess for the eigenvalue search. Also defines the type for
        # eigenvectors.
        x0 = TensorMap(randn, typ, outspace ← inspace)
        S, U, info = eigsolve(f, x0, howmany, :LM)
        # sfact is the ratio by which the number of sites changes at each coarse-graining.
        sfact = scalefactor(typeof(m))
        scaldims = sort(-log.(sfact, abs.(real(S))))
        scaldim_dict[irrep] = scaldims
    end
    return scaldim_dict
end

# # # Evaluation

"""
Return the expecation value of operator `op` for this MERA. The layer on which `op` lives is
set by `opscale`, which by default is the physical one (`opscale=1`). `evalscale` can be
used to set whether the operator is ascended through the network or the density matrix is
descended.
"""
function expect(op, m::GenericMERA; opscale=1, evalscale=num_translayers(m)+1)
    rho = densitymatrix(m, evalscale)
    op = ascend(op, m; startscale=opscale, endscale=evalscale)
    # If the operator is defined on a smaller support (number of sites) than rho, expand it.
    op = expand_support(op, support(rho))
    value = tr(rho * op)
    if abs(imag(value)/norm(op)) > 1e-13
        @warn("Non-real expectation value: $value")
    end
    value = real(value)
    return value
end


# # # Optimization

"""
Optimize the MERA `m` to minimize the expectation value of `h`.

The optimization proceeds by looping over layers and optimizing each in turn, starting from
the bottom, and repeating this until convergence is reached.

The keyword argument `lowest_depth` sets the lowest layer in the MERA that the optimization
is allowed to change, by default `lowest=1` so all layers are optimized. They keyword
argument `normalization` is a function that takes in the expectation value of `h` and
returns another number that is the actual quantity of interest (because `h` may for instance
be the Hamiltonian but with a changed normalization).  It is only used for printing log
messages, and doesn't affect the optimization.

The argument `pars` is a dictionary with symbols as keys, that lists values for different
parameters for how to do the optimization. They all need to be specified by the user, and
have no default values. The different parameters are:
    :miniter, a minimum number of iterations to do over the layers before returning. The
     first half of the minimum number of iterations will be spent optimizing only isometric
     tensors, leaving the disentanglers untouched.
    :maxiter, the maximum number of iterations to do over the layers before returning.
    :densitymatrix_delta, the convergence threshold. Convergence is measured as changes in
     the reduced density matrices of the MERA. Once the maximum by which any of them changed
     over consecutive iterations falls below pars[:densitymatrix_delta], the optimization
     terminates. Change is measured as Frobenius norm of the difference. Note that this is
     much more demanding than convergence in just the expectation value.
    :havg_depth, how deep into the scale invariant layer to go when computing the ascended
     version of `h` to use for optimizing the scale invariant layer. Should in principle be
     infinite, but the error goes down exponentially, so in practice 10 is often plenty.
    Other parameters may be required depending on the type of MERA. See documentation for
    the different Layer types. Typical parameters are for instance how many times to iterate
    optimizing individual tensors.
"""
function minimize_expectation!(m::GenericMERA, h, pars; lowest_depth=1,
                               normalization=identity)
    msg = "Optimizing a MERA with $(num_translayers(m)+1) layers"
    msg *= lowest_depth > 1 ? ", keeping the lowest $(lowest_depth-1) fixed." : "."
    @info(msg)
          
    nt = num_translayers(m)
    horig = ascend(h, m; endscale=lowest_depth)
    expectation = Inf
    expectation_change = Inf
    rhos = nothing
    rhos_maxchange = Inf
    counter = 0
    last_status_print = -Inf
    while (
           counter <= pars[:miniter]
           || (abs(rhos_maxchange) > pars[:densitymatrix_delta]
               && counter < pars[:maxiter])
          )
        counter += 1
        old_rhos = rhos
        old_expectation = expectation
        rhos = densitymatrices(m, lowest_depth)

        # We only optimize disentanglers starting after half of the compulsory iterations
        # have been done, to not have a screwed up isometry mislead us.
        do_disentanglers = (counter >= pars[:miniter]/2)

        h = horig
        for l in lowest_depth:nt
            rho = rhos[l-lowest_depth+2]
            layer = get_layer(m, l)
            layer = minimize_expectation_layer(h, layer, rho, pars;
                                               do_disentanglers=do_disentanglers)
            set_layer!(m, layer, l)
            h = ascend(h, m; startscale=l, endscale=l+1)
        end

        # Special case of the translation invariant layer.
        # The Hamiltonian to use for optimizing the scale invariant layer is
        # havg = h + A(h)/f + A^2(h)/f^2 + A^3(h)/f^3 + ...
        # where A is the ascending superoperator and f is the scaling factor, e.g. 2 for a
        # binary MERA and 3 for ternary.  We truncate this series at a few
        # (pars[:havg_depth]) terms because they become negligible exponentially quickly.
        havg = h
        hi = h
        sf = scalefactor(typeof(m))
        for i in 1:pars[:havg_depth]
            hi = ascend(hi, m; startscale=nt+i, endscale=nt+i+1)
            hi = hi/sf
            havg = havg + hi
        end
        layer = get_layer(m, Inf)
        layer = minimize_expectation_layer(havg, layer, rhos[end], pars;
                                           do_disentanglers=do_disentanglers)
        set_layer!(m, layer, Inf)

        expectation = expect(h, m, opscale=nt+1, evalscale=nt+1)
        expectation = normalization(expectation)
        expectation_change = (expectation - old_expectation)/abs(expectation)

        if old_rhos !== nothing
            rho_diffs = [norm(r - ro) for (r, ro) in zip(rhos, old_rhos)]
            rhos_maxchange = maximum(rho_diffs)
        end

        # As the optimization gets further, don't print status updates at every
        # iteration any more.
        if (counter - last_status_print)/counter > 0.02
            @info(@sprintf("Expectation = %.9e,  change = %.3e,  max rho change = %.3e,  counter = %d.",
                           expectation, expectation_change, rhos_maxchange, counter))
            last_status_print = counter
        end
    end
    return m
end

