# Utilities for demo.jl and and demo_refine.jl.

module DemoTools

using Logging
using Dates
using LinearAlgebra
using TensorKit
using JLD2
using MERA

export setlogger
export expect, minimize_expectation, densitymatrix_entropies, scalingdimensions, remove_symmetry
export load_mera, store_mera
export build_H_Ising, build_H_XXZ, build_magop
export get_optimized_mera

# # # Log formatting

function metafmt(level, _module, group, id, file, line)
    color = Logging.default_logcolor(level)
    levelstr = (level == Logging.Warn ? "Warning" : string(level))
    timestr = round(unix2datetime(time()), Dates.Second)
    prefix = "$(levelstr) $(timestr):"
    suffix = ""
    Logging.Info <= level < Logging.Warn && return color, prefix, suffix
    _module !== nothing && (suffix *= "$(_module)")
    if file !== nothing
        _module !== nothing && (suffix *= " ")
        suffix *= Base.contractuser(file)
        if line !== nothing
            suffix *= ":$(isa(line, UnitRange) ? "$(first(line))-$(last(line))" : line)"
        end
    end
    !isempty(suffix) && (suffix = "@ " * suffix)
    return color, prefix, suffix
end

setlogger() = global_logger(ConsoleLogger(stdout, Logging.Info; meta_formatter=metafmt))

# # # Functions for creating Hamiltonians.

"""
Take a two-site operator `op` that defines the local term of a global operator, and block
sites together to form a new two-site operator for which each site corresponds to
`num_sites` sites of the original operator, and which sums up to the same global operator.
`num_sites` should be a power of 2.
"""
function block_op(op::SquareTensorMap{2}, num_sites)
    while num_sites > 1
        VL = space(op, 1)
        VR = space(op, 2)
        eyeL = id(VL)
        eyeR = id(VR)
        opcross = eyeR ⊗ op ⊗ eyeL
        opleft = op ⊗ eyeL ⊗ eyeR
        opright = eyeL ⊗ eyeR ⊗ op
        op_new_unfused = opcross + 0.5*(opleft + opright)
        fusionspace = domain(op)
        fusetop = isomorphism(fuse(fusionspace), fusionspace)
        fusebottom = isomorphism(fusionspace, fuse(fusionspace))
        op = (fusetop ⊗ fusetop) * op_new_unfused * (fusebottom ⊗ fusebottom)
        num_sites /= 2
    end
    if num_sites != 1
        msg = "`num_sites` needs to be a power of 2"
        throw(ArgumentError(msg))
    end
    return op
end

"""
Take a one-site operator `op` that defines the local term of a global operator, and block
sites together to form a new one-site operator for which each site corresponds to
`num_sites` sites of the original operator, and which sums up to the same global operator.
`num_sites` should be a power of 2.
"""
function block_op(op::SquareTensorMap{1}, num_sites)
    while num_sites > 1
        V = space(op, 1)
        eye = id(V)
        opleft = op ⊗ eye
        opright = eye ⊗ op
        op_new_unfused = opleft + opright
        fusionspace = domain(op_new_unfused)
        fusetop = isomorphism(fuse(fusionspace), fusionspace)
        fusebottom = isomorphism(fusionspace, fuse(fusionspace))
        op = fusetop * op_new_unfused * fusebottom
        num_sites /= 2
    end
    if num_sites != 1
        msg = "`num_sites` needs to be a power of 2"
        throw(ArgumentError(msg))
    end
    return op
end

"""
Return the local Hamiltonian term for the XXZ model: J_xy*XX + J_xy*YY + J_z*ZZ.
`symmetry` should be "none" or "group", and determmines whether the Hamiltonian should be an
explicitly U(1) symmetric TensorMap or a dense one. `block_size` determines how many sites
to block together, and should be a power of 2. If sites are blocked, then the Hamiltonian is
divided by a constant to make the energy per site be the same as for the original.
"""
function build_H_XXZ(J_xy, J_z; symmetry="none", block_size=1)
    if symmetry == "U1" || symmetry == "group"
        V = ℂ[U₁](-1=>1, 1=>1)
        Z = TensorMap(zeros, Float64, V ← V)
        Z.data[U₁(1)] .= 1.0
        Z.data[U₁(-1)] .= -1.0
        ZZ = Z ⊗ Z
        XXplusYY = TensorMap(zeros, Float64, V ⊗ V ← V ⊗ V)
        XXplusYY.data[U₁(0)] .= [0.0 2.0; 2.0 0.0]
    elseif symmetry == "none"
        V = ℂ^2
        Z = TensorMap(zeros, Float64, V ← V)
        Z.data .= [1.0 0.0; 0.0 -1.0]
        ZZ = Z ⊗ Z
        XXplusYY = TensorMap(zeros, Float64, V ⊗ V ← V ⊗ V)
        XXplusYY.data[2,3] = 2.0
        XXplusYY.data[3,2] = 2.0
    else
        error("Unknown symmetry $symmetry")
    end
    H = J_xy*XXplusYY + J_z*ZZ
    H = block_op(H, block_size)
    H = H / block_size
    return H
end

"""
Return the local Hamiltonian term for the Ising model: -XX - h*Z
`symmetry` should be "none", "group", or "anyon" and determmines whether the Hamiltonian
should be an explicitly Z2 symmetric or anyonic TensorMap, or a dense one. `block_size`
determines how many sites to block together, and should be a power of 2. If sites are
blocked, then the Hamiltonian is divided by a constant to make the energy per site be the
same as for the original.
"""
function build_H_Ising(h=1.0; symmetry="none", block_size=1)
    if symmetry == "Z2"
        V = ℂ[ℤ₂](0=>1, 1=>1)
        # Pauli Z
        Z = TensorMap(zeros, Float64, V ← V)
        Z.data[ℤ₂(0)] .= 1.0
        Z.data[ℤ₂(1)] .= -1.0
        eye = id(V)
        ZI = Z ⊗ eye
        IZ = eye ⊗ Z
        # Pauli XX
        XX = TensorMap(zeros, Float64, V ⊗ V ← V ⊗ V)
        XX.data[ℤ₂(0)] .= [0.0 1.0; 1.0 0.0]
        XX.data[ℤ₂(1)] .= [0.0 1.0; 1.0 0.0]
        H = -(XX + h/2 * (ZI+IZ))
    elseif symmetry == "anyons"
        V = RepresentationSpace{IsingAnyon}(:I => 0, :ψ => 0, :σ => 1)
        H = TensorMap(zeros, Float64, V ⊗ V ← V ⊗ V)
        H.data[IsingAnyon(:I)] .= 1.0
        H.data[IsingAnyon(:ψ)] .= -1.0
    elseif symmetry == "none"
        V = ℂ^2
        # Pauli matrices
        X = TensorMap(zeros, Float64, V ← V)
        Z = TensorMap(zeros, Float64, V ← V)
        eye = id(V)
        X.data .= [0.0 1.0; 1.0 0.0]
        Z.data .= [1.0 0.0; 0.0 -1.0]
        XX = X ⊗ X
        ZI = Z ⊗ eye
        IZ = eye ⊗ Z
        H = -(XX + h/2 * (ZI+IZ))
    else
        error("Unknown symmetry $symmetry")
    end
    H = block_op(H, block_size)
    H = H / block_size
    return H
end

"""
Return the magnetization operator for the Ising model, blocked over `block_size` sites.
"""
function build_magop(;block_size=1)
    V = ℂ^2
    X = TensorMap(zeros, Float64, V ← V)
    eye = id(V)
    X.data .= [0.0 1.0; 1.0 0.0]
    magop = block_op(X, block_size)
    return magop
end

# # # Functions for reading and writing to disk.

function store_mera(path, m)
    # TODO JLD2 still fails to store MERAs properly. Once that's fixed, switch away from
    # using pseudoserialization.
    #@save path m
    deser = pseudoserialize(m)
    @save path deser
end

function load_mera(path)
    # TODO JLD2 still fails to store MERAs properly. Once that's fixed, switch away from
    # using pseudoserialization.
    #@load path m
    @load path deser
    m = depseudoserialize(deser...)
    return m
end

# # # Functions for optimizing a MERA.

"""
Given a VectorSpace `V`, return a list of sectors in which it might make sense to increase
the dimension of `V`, if we want to increase the total dimension. This means all sectors of
`V`, but possible some other sectors as well.
"""
function sectors_to_expand(V)
    result = Set(sectors(V))
    if typeof(V) == U₁Space
        # The `collect` makes a copy, so that we don't iterate over the ones just added.
        for s in collect(result)
            # TODO The following is specific to XXZ: We make jumps by twos, as because of
            # the Hamiltonian the odd sectors are useless.
            splus = U₁(s.charge+2)
            sminus = U₁(s.charge-2)
            push!(result, splus)
            push!(result, sminus)
        end
    end
    return result
end

"""
For a MERA `m` and its layer number `i`, we want to increase its bond dimension (the input
dimension) to `chi`. To do that, we may have to decide on which symmetry sector to increase
the dimension in. For that purpose, go through the possible symmetry sectors, try increasing
each one, and optimize a bit too see how much it helps bring down the energy. Choose the
symmetry sector that benefits the energy the most. Return the expanded and slightly
optimized MERA.
"""
function expand_best_sector(m, i, chi, h, opt_pars)
    V = inputspace(m, i)
    d = dim(V)
    expanded_meras = Dict()
    # Parameters for an optimisation the only point of which is to restore isometricity.
    isometrisation_pars = merge(opt_pars, (method = :ev,
                                           maxiter = 1,
                                           isometries_only_iters = 0,))
    for s in sectors_to_expand(V)
        # Expand the bond dimension of symmetry sector s and try optimizing a bit to see
        # how useful this expansion is.
        ms = deepcopy(m)
        ds = dim(V, s)
        chi_s = ds + (chi - d)
        istop = (i == num_translayers(ms))
        ms = expand_bonddim(ms, i, Dict(s => chi_s); check_invar=!istop)
        ms = expand_internal_bonddim(ms, i+1, Dict(s => chi_s))
        # Just to make the MERA isometric again, after expanding bond dimensions.
        ms = minimize_expectation(ms, h, isometrisation_pars)
        msg = "Expanded layer $i to bond dimenson $chi_s in sector $s."
        @info(msg)
        ms = minimize_expectation(ms, h, opt_pars)
        expanded_meras[s] = ms
    end
    expanded_meras_array = collect(expanded_meras)
    # Of the MERAs that got different symmetry sectors expanded, pick the one that has
    # the smallest energy.
    minindex = argmin((pair -> expect(h, pair[2]) for pair in expanded_meras_array))
    s, m = expanded_meras_array[minindex]
    msg = "Expanding sector $s yielded the lowest energy, keeping that.\nOutput spaces now:"
    for i in 1:num_translayers(m)+1
        msg *= "\nLayer $(i): $(outputspace(m, i))"
    end
    @info(msg)
    return m
end

"""
Return the smallest vector space with the given symmetry (either "none", "U1", or "Z2", at
the moment) that makes sense as a index space of a MERA for the given model (either "Ising"
or "XXZ"). For instance, the Ising model should have a non-zero bond dimension in both
symmetry sectors, so the minimal space for Ising and Z2 is ComplexSpace(0 => 1, 1 => 1).
"""
function minimal_space(model, symmetry)
    if symmetry == "none"
        V = ℂ^1
    elseif model == "XXZ" && symmetry == "U1"
        V = ℂ[U₁](-2=>1, 0=>1, 2=>1)
    elseif model == "Ising" && symmetry == "Z2"
        V = ℂ[ℤ₂](0=>1, 1=>1)
    else
        error("Unknown symmetry $symmetry")
    end
    return V
end

"""
Get a MERA for the given model, optimized with the parameters `pars`. If the requested MERA
is already stored on disk, load and return it. If not, optimize for it, store the result on
disk for future use, and return it. This function often recursively calls itself, because
MERAs with lower bond dimensions are used as starting points for the optimisation of the
higher-bond-dimension ones.
"""
function get_optimized_mera(datafolder, model, pars)
    chi = pars[:chi]
    layers = pars[:layers]
    symmetry = pars[:symmetry]
    block_size = pars[:block_size]
    threads = pars[:threads]
    BLAS.set_num_threads(threads)
    meratypestr = pars[:meratype]
    if meratypestr == "binary"
        meratype = BinaryMERA
    elseif meratypestr == "ternary"
        meratype = TernaryMERA
    elseif meratypestr == "modifiedbinary"
        meratype = ModifiedBinaryMERA
    else
        msg = "Unknown MERA type: $(meratypestr)"
        throw(ArgumentError(msg))
    end

    # The path to where this MERA should be stored.
    mkpath(datafolder)
    filename = "MERA_$(model)_$(meratypestr)_$(chi)_$(block_size)_$(symmetry)_$(layers)"
    path = "$datafolder/$filename.jlm"

    if isfile(path)
        @info("Found $filename on disk, loading it.")
        m = load_mera(path)
        return m
    end

    @info("Did not find $filename on disk, generating it.")
    # Build the Hamiltonian.
    if model == "XXZ"
        h = build_H_XXZ(pars[:J_xy], pars[:J_z]; symmetry=symmetry,
                        block_size=block_size)
    elseif model == "Ising"
        h = build_H_Ising(pars[:h]; symmetry=symmetry, block_size=block_size)
    else
        msg = "Unknown model: $(model)"
        throw(ArgumentError(msg))
    end

    # Figure out whether we should start the optimization from a random MERA or a previous
    # MERA.
    V_minimal = minimal_space(model, symmetry)
    if chi < dim(V_minimal)
        msg = "chi = $chi is too small for a $(symmetry)-symmetric MERA for $(model) model."
        throw(ArgumentError(msg))
    elseif chi == dim(V_minimal)
        # The requested bond dimension is the smallest that makes sense to do with this
        # symmetry. So just create a random MERA and optimize that.
        V_virt = V_minimal
        V_phys = space(h, 1)
        Vs = tuple(V_phys, repeat([V_virt], layers-1)...)
        T = eltype(h)
        m = random_MERA(meratype, T, Vs; random_disentangler=false)
        # We run all three stages of the optimization for the initial bond dimensions as
        # well. This doesn't really make sense by itself, but it ensures that all bond
        # dimensions go through all the different stages, which is necessary for instance
        # because pars[:initial_opt_pars] typically involves a starting part where
        # disentanglers are not optimized, and we want to apply that here too.
        m = minimize_expectation(m, h, pars[:initial_opt_pars])
        m = minimize_expectation(m, h, pars[:mid_opt_pars])
        m = minimize_expectation(m, h, pars[:final_opt_pars])
    else
        # The bond dimensions requested is larger than the smallest that makes sense to do.
        # Get the MERA with a bond dimension one smaller to use as a starting point, expand
        # its bond dimension, and optimize that one.
        prevpars = deepcopy(pars)
        prevpars[:chi] -= 1
        m = get_optimized_mera(datafolder, model, prevpars)
        # Expand the bond dimension of each layer in turn.
        for i in 1:num_translayers(m)
            m = expand_best_sector(m, i, chi, h, pars[:initial_opt_pars])
            opt_pars = i == num_translayers(m) ? pars[:final_opt_pars] : pars[:mid_opt_pars]
            m = minimize_expectation(m, h, opt_pars)
        end
    end

    store_mera(path, m)
    return m
end

end  # module
