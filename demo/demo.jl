# A script that optimizes a MERA for either the Ising or XXZ model, and computes some
# physical quantities from it. The MERAs built are stored on disk.

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
                   , "--meratype", arg_type=String, default="ternary"
                   , "--threads", arg_type=Int, default=1  # For BLAS parallelization
                   , "--chis", arg_type=Vector{Int}, default=collect(2:3)  # Bond dimensions
                   , "--layers", arg_type=Int, default=3
                   , "--symmetry", arg_type=String, default="none"  # "none" or "group"
                   , "--block_size", arg_type=Int, default=2  # Block two sites to start
                   , "--h", arg_type=Float64, default=1.0  # External field of Ising
                   , "--Delta", arg_type=Float64, default=-0.5  # Isotropicity in XXZ
                   , "--datafolder", arg_type=String, default="JLMdata"
    )
    pars = parse_args(ARGS, settings; as_symbols=true)
    return pars
end

function main()
    pars = parse_pars()
    model = pars[:model]
    chis = pars[:chis]
    layers = pars[:layers]
    symmetry = pars[:symmetry]
    datafolder = pars[:datafolder]
    if symmetry == "group"
        model == "Ising" && (symmetry = "Z2")
        model == "XXZ" && (symmetry = "U1")
        pars[:symmetry] = symmetry
    end
    block_size = pars[:block_size]

    # Three sets of parameters are used when optimizing the MERA:
    # Used when determining which sector to give bond dimension to.
    pars[:initial_opt_pars] = Dict(:densitymatrix_delta => 1e-5,
                                   :maxiter => 10,
                                   :miniter => 10,
                                   :havg_depth => 10,
                                   :layer_iters => 1,
                                   :disentangler_iters => 1,
                                   :isometry_iters => 1)
    # Used when optimizing a MERA that has some layers expanded to desired bond
    # dimension, but not all.
    pars[:mid_opt_pars] = Dict(:densitymatrix_delta => 1e-5,
                               :maxiter => 30,
                               :miniter => 10,
                               :havg_depth => 10,
                               :layer_iters => 1,
                               :disentangler_iters => 1,
                               :isometry_iters => 1)
    # Used when optimizing a MERA that has all bond dimensions at the full,
    # desired value.
    pars[:final_opt_pars] = Dict(:densitymatrix_delta => 1e-7,
                                 :maxiter => 1000,
                                 :miniter => 10,
                                 :havg_depth => 10,
                                 :layer_iters => 1,
                                 :disentangler_iters => 1,
                                 :isometry_iters => 1)

    # Get the Hamiltonian.
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

    # Computing the magnetisation of the Ising model only makes sense if Z2 symmetry isn't
    # explicitly enforced.
    do_magnetisation = model == "Ising" && symmetry == "none"
    for chi in chis
        # For each bond dimension, get the optimized MERA and print out some interesting
        # numbers for it.
        temppars = deepcopy(pars)
        temppars[:chi] = chi
        m = get_optimized_mera(datafolder, model, temppars)

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
