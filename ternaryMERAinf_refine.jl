module TenaryMERAInfRefine

using ArgParse
using LinearAlgebra
using TensorKit
include("ternaryMERAinf.jl")
include("ternaryMERAinf_modeltools.jl")
using .TernaryMERAInfModelTools
using .TernaryMERAInf

function parse_pars()
    settings = ArgParseSettings(autofix_names=true)
    @add_arg_table(settings
                   , "--model", arg_type=String, default=""
                   , "--threads", arg_type=Int, default=1
                   , "--chi", arg_type=Int, default=5
                   , "--layers", arg_type=Int, default=3
                   , "--symmetry", arg_type=String, default="group"
                   , "--block", arg_type=Int, default=2
                   , "--reps", arg_type=Int, default=1000
                   , "--h", arg_type=Float64, default=1.0
                   , "--Delta", arg_type=Float64, default=-0.5
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
    BLAS.set_num_threads(threads)
    if symmetry == "group"
        if model == "Ising"
            symmetry = "Z2"
        elseif model == "XXZ"
            symmetry = "U1"
        end
        pars[:symmetry] = symmetry
    end
    block = pars[:block]
    reps = pars[:reps]
    # Used when determining which sector to give bond dimension to.
    opt_pars = Dict(:rho_delta => 1e-15,
                    :maxiter => 1000,
                    :miniter => 10,
                    :havg_depth => 10,
                    :uw_iters => 1,
                    :u_iters => 1,
                    :w_iters => 1)

    if model == "Ising"
        h, dmax = build_H_Ising(pars[:h]; symmetry=symmetry, block=block)
        symmetry == "none" && (magop = build_magop(block=block))
    elseif model == "XXZ"
        h, dmax = build_H_XXZ(pars[:Delta]; symmetry=symmetry, block=block)
    else
        msg = "Unknown model $(model)."
        throw(ArgumentError(msg))
    end
    normalization(x) = normalize_energy(x, dmax, block)

    datafolder = "./JLMdata"
    mkpath(datafolder)
    filename = "MERA_$(model)_$(chi)_$(block)_$(symmetry)_$(layers)_$(version)"
    path = "$datafolder/$filename.jlm"
    path_ref = "$datafolder/$(filename)_refined.jlm"
    matlab_path_ref = "./matlabdata/$(filename)_refined.mat"

    if isfile(path_ref)
        msg = "Found $(path_ref), refining it."
        println(msg)
        m = load_mera(path_ref)
    elseif isfile(path)
        msg = "Found $(path), refining it."
        println(msg)
        m = load_mera(path)
    else
        msg = "File not found: $(filename)"
        throw(ArgumentError(msg))
    end

    # Keep repeatedly optimizing this MERA by doing maxiter iterations, storing
    # the state, and then starting again, until `reps*maxiter` iterations have
    # been done in total.
    for rep in 1:reps
        println("Starting rep #$(rep).")
        optimize_layerbylayer!(m, h, 0, normalization, opt_pars)

        store_mera(path_ref, m)
        store_mera_matlab(matlab_path_ref, m)

        energy = expect(h, m)
        energy = normalize_energy(energy, dmax, block)
        rhoees = getrhoees(m)
        model == "Ising" && symmetry == "none" && (magnetization = expect(magop, remove_symmetry(m)))

        println("Done.")
        println("Energy numerical: $energy")
        model == "Ising" && println("Energy exact:     $(-4/pi)")
        model == "Ising" && symmetry == "none" && println("Magnetization: $(magnetization)")
        println("rho ees:")
        println(rhoees)

        scaldims = get_scaldims(m)
        println("Scaling dimensions:")
        println(scaldims)
    end
end

main()

end  # module
