# Script that loads a previously optimized MERA from disk, and optimizes it further
# ("refines" it).

using ArgParse
using LinearAlgebra
using TensorKit
include("demo_tools.jl")
using .DemoTools
using Logging

# TODO Hopefully this can be dropped soon. See
# https://github.com/carlobaldassi/ArgParse.jl/issues/95.
ArgParse.parse_item(::Type{Symbol}, s::AbstractString) = Symbol(s)

function parse_pars()
    settings = ArgParseSettings(autofix_names=true)
    @add_arg_table(settings
                   , "--model", arg_type=String, default="Ising"
                   , "--meratype", arg_type=String, default="ternary"
                   , "--threads", arg_type=Int, default=1  # For BLAS parallelization
                   , "--chi", arg_type=Int, default=8  # Bond dimension
                   , "--layers", arg_type=Int, default=3
                   , "--reps", arg_type=Int, default=1000
                   , "--symmetry", arg_type=String, default="none"  # "none" or "group"
                   , "--block_size", arg_type=Int, default=2  # Block two sites to start
                   , "--h", arg_type=Float64, default=1.0  # External field of Ising
                   , "--J_z", arg_type=Float64, default=0.5  # ZZ coupling for XXZ
                   , "--J_xy", arg_type=Float64, default=-1.0  # XX + YY coupling for XXZ
                   , "--datafolder", arg_type=String, default="JLMdata"
                   , "--datasuffix", arg_type=String, default=""
                   , "--method", arg_type=Symbol, default=:lbfgs
                   , "--retraction", arg_type=Symbol, default=:cayley
                   , "--transport", arg_type=Symbol, default=:cayley
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
    reps = pars[:reps]
    opt_pars = Dict(:method => pars[:method],
                    :retraction => pars[:retraction],
                    :transport => pars[:transport],
                    :gradient_delta => 1e-15,
                    :densitymatrix_delta => 1e-15,
                    :maxiter => 500,
                    :miniter => 1,
                    :havg_depth => 10,
                    :layer_iters => 1,
                    :disentangler_iters => 1,
                    :isometry_iters => 1)

    # Build the Hamiltonian.
    if model == "Ising"
        h, normalization = DemoTools.build_H_Ising(pars[:h]; symmetry=symmetry,
                                                   block_size=block_size)
        symmetry == "none" && (magop = DemoTools.build_magop(block_size=block_size))
    elseif model == "XXZ"
        h, normalization = DemoTools.build_H_XXZ(pars[:J_xy], pars[:J_z]; symmetry=symmetry,
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
    # Keep repeatedly optimizing this MERA by doing maxiter iterations, storing
    # the state, and then starting again, until `reps*maxiter` iterations have
    # been done in total.
    starttime = time()
    for rep in 1:reps
        @info("Starting rep #$(rep).")
        m = minimize_expectation!(m, h, opt_pars; normalization=normalization)
        DemoTools.store_mera(path_ref, m)

        energy = normalization(expect(h, m))
        rhoees = densitymatrix_entropies(m)
        do_magnetisation && (magnetization = expect(magop, remove_symmetry(m)))
        scaldims = scalingdimensions(m)

        @info("Done with bond dimension $(chi).")
        @info("Energy numerical: $energy")
        model == "Ising" && @info("Energy exact:     $(-4/pi)")
        do_magnetisation && @info("Magnetization: $(magnetization)")
        @info("rho ees:")
        @info(rhoees)
        @info("Scaling dimensions:")
        @info(scaldims)
        endtime = time()
        timetaken = (endtime - starttime) / 60.0
        @info("Time passed: $(timetaken) mins")
    end
end

main()

