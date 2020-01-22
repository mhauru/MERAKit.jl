using ArgParse
using LinearAlgebra
using TensorKit
include("demo_tools.jl")
using .DemoTools
using .DemoTools.MERA

function parse_pars()
    settings = ArgParseSettings(autofix_names=true)
    @add_arg_table(settings
                   , "--model", arg_type=String, default="Ising"
                   , "--meratype", arg_type=String, default="binary"
                   , "--threads", arg_type=Int, default=1
                   , "--chi", arg_type=Int, default=3
                   , "--layers", arg_type=Int, default=3
                   , "--symmetry", arg_type=String, default="none"
                   , "--block_size", arg_type=Int, default=2
                   , "--reps", arg_type=Int, default=1000
                   , "--h", arg_type=Float64, default=1.0
                   , "--Delta", arg_type=Float64, default=-0.5
                   , "--datafolder", arg_type=String, default="JLMdata"
    )
    pars = parse_args(ARGS, settings; as_symbols=true)
    return pars
end

function main()
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
    opt_pars = Dict(:densitymatrix_delta => 1e-15,
                    :maxiter => 1000,
                    :miniter => 10,
                    :havg_depth => 10,
                    :layer_iters => 1,
                    :disentangler_iters => 1,
                    :isometry_iters => 1)

    # Build the Hamiltonian.
    if model == "Ising"
        h, dmax = build_H_Ising(pars[:h]; symmetry=symmetry, block_size=block_size)
        symmetry == "none" && (magop = build_magop(block_size=block_size))
    elseif model == "XXZ"
        h, dmax = build_H_XXZ(pars[:Delta]; symmetry=symmetry, block_size=block_size)
    else
        msg = "Unknown model $(model)."
        throw(ArgumentError(msg))
    end
    normalization(x) = normalize_energy(x, dmax, block_size)

    # The path where this MERA is located. There are two options, a file for a MERA that has
    # already been previously refined, and one for a MERA that hasn't. We favor the former.
    mkpath(datafolder)
    filename = "MERA_$(model)_$(meratypestr)_$(chi)_$(block_size)_$(symmetry)_$(layers)"
    path = "$datafolder/$filename.jlm"
    path_ref = "$datafolder/$(filename)_refined.jlm"

    if isfile(path_ref)
        msg = "Found $(path_ref), refining it."
        @info(msg)
        m = load_mera(path_ref)
    elseif isfile(path)
        msg = "Found $(path), refining it."
        @info(msg)
        m = load_mera(path)
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
    for rep in 1:reps
        @info("Starting rep #$(rep).")
        minimize_expectation!(m, h, opt_pars; normalization=normalization)
        store_mera(path_ref, m)

        energy = expect(h, m)
        energy = normalize_energy(energy, dmax, block_size)
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
    end
end

main()

