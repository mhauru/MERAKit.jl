# The most important types of the package, GenericMERA and Layer, on which all specific MERA
# implementations (binary, ternary, ...) will be built. Methods and functions that can be
# implemented on this level of abstraction, without having to know the details of the
# specific MERA type.
# To be included in MERA.jl.

"""
Layer is an abstract supertype of concrete types such as BinaryLayer. A typical Layer is a
collection of tensors, the orders and shapes of which depend on the type.
"""
abstract type Layer end

"""
A GenericMERA is a collection of Layers. The type of these layers then determines whether the
MERA is binary, ternary, etc. The counting starts from the bottom, i.e. from the physical
layer. The last layer is the scale invariant one, that then repeats upwards to infinity.
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
get_layer(m::GenericMERA, depth) = depth > num_translayers(m) ?  m.layers[end] : m.layers[depth]

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
    layer = map(copy, layer)
    push!(m.layers, layer)
    return m
end

"""
Given a MERA and a depth, return the vectorspace of the downwards-pointing (towards the
physical level) indices of the layer at that depth.
"""
inputspace(m::GenericMERA, depth) = inputspace(get_layer(m, depth))

"""
Given a MERA and a depth, return the vectorspace of the upwards-pointing (towards scale
invariance) indices of the layer at that depth.
"""
outputspace(m::GenericMERA, depth) = outputspace(get_layer(m, depth))

# # # Generating random MERAs

"""
Replace the layer at the given depth with a random tensors. Additional keyword arguments are
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

function random_MERA(::Type{T}, V, num_layers; kwargs...) where T <: GenericMERA
    Vs = repeat([V], num_layers+1)
    return random_MERA(T, Vs; kwargs...)
end

function random_MERA(::Type{T}, Vs; kwargs...) where T <: GenericMERA
    num_layers = length(Vs)
    layers = []
    for i in 1:num_layers
        V = Vs[i]
        Vnext = (i < num_layers ? Vs[i+1] : V)
        layer = randomlayer(layertype(T), V, Vnext; kwargs...)
        push!(layers, layer)
    end
    m = T(layers)
    return m
end

"""
A function for which the first argument is the Layer type, and the second and third are the
input and output spaces, that returns a MERA layer with random tensors. May take further
keyword arguments depending on the Layer type. Each subtype of Layer should have its own
method for this function.
"""
function randomlayer end

# TODO Fix the preservation of isometricity.
"""
Expand the bond dimension of the MERA at the given depth. `depth=1` is the first virtual
level, just above the first layer of the MERA, and the numbering grows from there. The new
bond dimension is given by `newdims`, which for a non-symmetric MERA is just a number, and
for a symmetric MERA is a dictionary of {irrep => block dimension}. Not all irreps for a
bond need to be listed, the once left out are left untouched.

The expansion is done by padding tensors with zeros. Note that this breaks isometricity.
"""
function expand_bonddim!(m::GenericMERA, depth, newdims)
    # Note that this breaks the isometricity of the MERA. A round of
    # optimization will fix that. TODO This shouldn't be allowed to be the case, fix it.
    V = outputspace(m, depth)
    V = expand_vectorspace(V, newdims)

    layer = get_layer(m, depth)
    layer = expand_outputspace(layer, V)
    set_layer!(m, layer, depth; check_invar=false)

    next_layer = get_layer(m, depth+1)
    next_layer = expand_inputspace(next_layer, V)
    if depth == num_translayers(m)
        # next_layer is the scale invariant part, so we need to change its top
        # index too since we changed the bottom..
        next_layer = expand_outputspace(next_layer, V)
    end
    set_layer!(m, next_layer, depth+1; check_invar=true)
end

# # # Pseudo(de)serialization
# "Pseudo(de)serialization" refers to breaking the MERA down into types in Julia base, and
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
depseudoserialize(::Type{T}, args) where T <: GenericMERA = T([depseudoserialize(d...) for d in args])
depseudoserialize(str::String, args) = depseudoserialize(eval(Meta.parse(str)), args)

"""
Return a tuple of objects that can be used to reconstruct a given TensorMap, and that are
all of Julia base types.
"""
function pseudoserialize(t::T) where T <: TensorMap
    # We make use of the nice fact that many TensorKit objects return on repr
    # strings that are valid syntax to reconstruct these objects.
    domstr = repr(t.dom)
    codomstr = repr(t.codom)
    eltyp = eltype(t)
    if isa(t.data, AbstractArray)
        data = t.data
    else
        data = Dict(repr(s) => deepcopy(d) for (s, d) in t.data)
    end
    return repr(T), (domstr, codomstr, eltyp, data)
end

"""
Reconstruct a TensorMap given the output of `pseudoserialize`.
"""
function depseudoserialize(::Type{T}, args) where T <: TensorMap
    # We make use of the nice fact that many TensorKit objects return on repr
    # strings that are valid syntax to reconstruct these objects.
    domstr, codomstr, eltyp, data = args
    dom = eval(Meta.parse(domstr))
    codom = eval(Meta.parse(codomstr))
    t = TensorMap(zeros, eltyp, codom ← dom)
    if isa(t.data, AbstractArray)
        t.data[:] = data
    else
        for (irrepstr, irrepdata) in data
            irrep = eval(Meta.parse(irrepstr))
            t.data[irrep][:] = irrepdata
        end
    end
    return t
end

# Implementations for pseudoserialize(l::T) and depseudoserialize(::Type{T}, args) should be
# written for each T <: Layer. They can make use of the (de)pseudoserialize methods for
# TensorMaps above.

# # # Invariants

"""
Check that the indices of the various tensors in a given MERA are compatible with each
other. If not, throw an ArgumentError. If yes, return true. This relies on two checks,
`space_invar_intralayer` for checking indices within a layer and `space_invar_interlayer`
for checking indices between layers. These should be implemented for each subtype of Layer.
"""
function space_invar(m::GenericMERA{T}) where T
    layer = get_layer(m, 1)
    # We go to num_translayers(m)+2, to go a bit into the scale invariant part.
    for i in 2:(num_translayers(m)+2)
        next_layer = get_layer(m, i)
        if applicable(space_invar_intralayer, layer)
            if !space_invar_intralayer(layer)
                errmsg = "Mismatching bonds in MERA within layer $(i-1)."
                throw(ArgumentError(errmsg))
            end
        else
            msg = "space_invar_intralayer has no method for type $(T). We recommend writing one, to enable checking for space mismatches when assigning tensors."
            println(msg)
        end

        if applicable(space_invar_interlayer, layer, next_layer)
            if !space_invar_interlayer(layer, next_layer)
                errmsg = "Mismatching bonds in MERA between layers $(i-1) and $i."
                throw(ArgumentError(errmsg))
            end
        else
            msg = "space_invar_interlayer has no method for type $(T). We recommend writing one, to enable checking for space mismatches when assigning tensors."
            println(msg)
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
Ascend the operator op, that lives on the layer startscale, through the MERA to the layer
endscale. Living "on" a layer means living on the indices right below it, so startscale=1
(the default) refers to the physical indices, and endscale=num_translayers(m)=1 (the
default) to the indices just below the first scale invariant layer.
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
Descend the operator op, that lives on the layer startscale, through the MERA to the layer
endscale. Living "on" a layer means living on the indices right below it, so endscale=1 (the
default) refers to the physical indices, and startscale=num_translayers(m)=1 (the default)
to the indices just below the first scale invariant layer.
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
    V = outputspace(m, Inf)
    width = causal_cone_width(T)
    eye = TensorMap(I, Float64, V ← V)
    x0 = ⊗(repeat([eye], width)...)
    vals, vecs, info = eigsolve(f, x0)
    rho = vecs[1]
    # rho is Hermitian only up to a phase. Divide out that phase.
    @tensor tr[] := rho[1,2,1,2]
    rho /= scalar(tr)
    return rho
end

"""
Return the density matrix right below the layer at `depth`.
"""
function densitymatrix(m::GenericMERA, depth)
    rho = fixedpoint_densitymatrix(m)
    if depth < num_translayers(m)+1
        rho = descend(rho, m; endscale=depth,
                      startscale=num_translayers(m)+1)
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
    value = tr(rho * op)
    if abs(imag(value)/value) > 1e-13
        @warn("Non-real expectation value: $value")
    end
    value = real(value)
    return value
end


# # # Optimization

"""
TODO Write this docstring.
"""
function minimize_expectation!(m::GenericMERA, h, pars; lowest_depth=1,
                               normalization=identity)
    println("Optimizing a MERA with $(num_translayers(m)+1) layers,"*
            " keeping the lowest $(lowest_depth-1) fixed.")
    nt = num_translayers(m)
    horig = ascend(h, m; endscale=lowest_depth)
    energy = Inf
    energy_change = Inf
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
        old_energy = energy
        rhos = densitymatrices(m, lowest_depth)

        h = horig
        for l in lowest_depth:nt
            rho = rhos[l-lowest_depth+2]
            layer = get_layer(m, l)
            # We only optimize u starting from the last of the compulsory
            # iterations, to not have a screwed up w mislead us.
            do_disentanglers = (counter >= pars[:miniter])
            layer = minimize_expectation_layer(h, layer, rho, pars;
                                               do_disentanglers=do_disentanglers)
            set_layer!(m, layer, l)
            h = ascend(h, m; startscale=l, endscale=l+1)
        end

        # Special case of the translation invariant layer.
        havg = h
        hi = h
        for i in 1:pars[:havg_depth]
            hi = ascend(hi, m; startscale=nt+i, endscale=nt+i+1)
            hi = hi/3
            havg = havg + hi
        end
        layer = get_layer(m, Inf)
        do_disentanglers = (counter >= pars[:miniter])
        layer = minimize_expectation_layer(havg, layer, rhos[end], pars;
                                           do_disentanglers=do_disentanglers)
        set_layer!(m, layer, Inf)

        energy = expect(h, m, opscale=nt+1, evalscale=nt+1)
        energy = normalization(energy)
        energy_change = (energy - old_energy)/energy

        if old_rhos !== nothing
            rho_diffs = [norm(r - ro) for (r, ro) in zip(rhos, old_rhos)]
            rhos_maxchange = maximum(rho_diffs)
        end

        # As the optimization gets further, don't print status updates at every
        # iteration any more.
        if (counter - last_status_print)/counter > 0.02
            @printf("Energy = %.9e,  energy change = %.3e,  max rho change = %.3e,  counter = %d.\n",
                    energy, energy_change, rhos_maxchange, counter)
            last_status_print = counter
        end
    end
    return m
end

