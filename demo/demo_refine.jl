# Script that loads a previously optimized MERA from disk, and optimizes it further
# ("refines" it).

using ArgParse
using LinearAlgebra
using TensorKit
include("demo_tools.jl")
using .DemoTools
using Logging

function parse_pars()
    settings = ArgParseSettings(autofix_names=true)
    @add_arg_table!(settings
                    , "--model", arg_type=String, default="Ising"
                    , "--meratype", arg_type=String, default="ternary"
                    , "--threads", arg_type=Int, default=1  # For BLAS parallelization
                    , "--chi", arg_type=Int, default=8  # Bond dimension
                    , "--layers", arg_type=Int, default=3
                    , "--checkpoint_frequency", arg_type=Int, default=1000
                    , "--symmetry", arg_type=String, default="none"  # "none" or "group"
                    , "--block_size", arg_type=Int, default=2  # Block two sites to start
                    , "--h", arg_type=Float64, default=1.0  # External field of Ising
                    , "--J_z", arg_type=Float64, default=0.5  # ZZ coupling for XXZ
                    , "--J_xy", arg_type=Float64, default=-1.0  # XX + YY coupling for XXZ
                    , "--datafolder", arg_type=String, default="JLMdata"
                    , "--datasuffix", arg_type=String, default=""
                    , "--method", arg_type=Symbol, default=:lbfgs
                    , "--isometrymanifold", arg_type=Symbol, default=:grassmann
                    , "--retraction", arg_type=Symbol, default=:exp
                    , "--transport", arg_type=Symbol, default=:exp
                    , "--metric", arg_type=Symbol, default=:euclidean
                    , "--precondition", arg_type=Bool, default=true
                    , "--lbfgs-m", arg_type=Int, default=8
                    , "--cg-flavor", arg_type=Symbol, default=:HagerZhang
                    , "--scale_invariant_eps", arg_type=Float64, default=1e-6
                    , "--verbosity", arg_type=Int, default=2
                   )
    pars = parse_args(ARGS, settings; as_symbols=true)
    return pars
end

function main()
    DemoTools.setlogger()
    pars = parse_pars()
    model = pars[:model]
    chi = pars[:chi]
    layers = pars[:layers]
    symmetry = pars[:symmetry]
    threads = pars[:threads]
    meratypestr = pars[:meratype]
    datafolder = pars[:datafolder]
    BLAS.set_num_threads(threads)
    if symmetry == "group"
        model == "Ising" && (symmetry = "Z2")
        model == "XXZ" && (symmetry = "U1")
        pars[:symmetry] = symmetry
    end
    block_size = pars[:block_size]
    checkpoint_frequency = pars[:checkpoint_frequency]
    scale_invariant_eps = pars[:scale_invariant_eps]
    opt_pars = (
                maxiter = typemax(Int),
                method = pars[:method],
                isometrymanifold = pars[:isometrymanifold],
                retraction = pars[:retraction],
                transport = pars[:transport],
                metric = pars[:metric],
                precondition = pars[:precondition],
                lbfgs_m = pars[:lbfgs_m],
                cg_flavor = pars[:cg_flavor],
                verbosity = pars[:verbosity],
                scaleinvariant_krylovoptions = (
                                                tol = scale_invariant_eps,
                                                krylovdim = 4,
                                                verbosity = 0,
                                                maxiter = 20,
                                               ),
               )

    logstr = "Running demo_refine.jl with"
    for (k, v) in pars
        logstr = logstr * "\n$k = $v"
    end
    @info(logstr)

    # Build the Hamiltonian.
    if model == "Ising"
        h = DemoTools.build_H_Ising(pars[:h]; symmetry=symmetry,
                                    block_size=block_size)
        symmetry == "none" && (magop = DemoTools.build_magop(block_size=block_size))
    elseif model == "XXZ"
        h = DemoTools.build_H_XXZ(pars[:J_xy], pars[:J_z]; symmetry=symmetry,
                                  block_size=block_size)
    else
        msg = "Unknown model $(model)."
        throw(ArgumentError(msg))
    end

    # The path where this MERA is located. There are two options, a file for a MERA that has
    # already been previously refined, and one for a MERA that hasn't. We favor the former.
    mkpath(datafolder)
    suffix = pars[:datasuffix]
    filename = "MERA_$(model)_$(meratypestr)_$(chi)_$(block_size)_$(symmetry)_$(layers)"
    path = "$datafolder/$filename.jlm"
    path_ref = "$datafolder/$(filename)_refined$(suffix).jlm"

    if isfile(path_ref)
        msg = "Found $(path_ref), refining it."
        @info(msg)
        m = DemoTools.load_mera(path_ref)
    elseif isfile(path)
        msg = "Found $(path), refining it."
        @info(msg)
        m = DemoTools.load_mera(path)
    else
        msg = "File not found: $(filename)"
        throw(ArgumentError(msg))
    end

    # Computing the magnetisation of the Ising model only makes sense if Z2 symmetry isn't
    # explicitly enforced.
    do_magnetisation = model == "Ising" && symmetry == "none"
    # Keep optimizing this MERA, storing the state and printing some extra information every
    # `checkpoint_frequency` iterations.
    checkpointtime = time()

    function finalize!(m, expectation, g, repnum)
        repnum % checkpoint_frequency != 0 && (return m, expectation, g)
        DemoTools.store_mera(path_ref, m)

        energy = expect(h, m)
        rhoees = densitymatrix_entropies(m)
        do_magnetisation && (magnetization = expect(magop, remove_symmetry(m)))
        scaldims = scalingdimensions(m)
        old_checkpointtime = checkpointtime
        checkpointtime = time()
        timetaken = (checkpointtime - old_checkpointtime) / 60.0

        do_magnetisation && @info("Magnetization: $(magnetization)")
        @info("rho ees: $rhoees")
        @info("Scaling dimensions: $scaldims")
        @info("Time passed: $(timetaken) mins")
        return m, expectation, g
    end

    m = minimize_expectation(m, h, opt_pars; finalize! = finalize!)
end

main()

