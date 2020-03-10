# A script that optimizes a MERA for either the Ising or XXZ model, and computes some
# physical quantities from it. The MERAs built are stored on disk.

using ArgParse
using LinearAlgebra
using TensorKit
include("demo_tools.jl")
using .DemoTools

# TODO Hopefully this can be dropped soon. See
# https://github.com/carlobaldassi/ArgParse.jl/issues/95.
ArgParse.parse_item(::Type{Symbol}, s::AbstractString) = Symbol(s)

function parse_pars()
    settings = ArgParseSettings(autofix_names=true)
    @add_arg_table(settings
                   , "--model", arg_type=String, default="Ising"
                   , "--meratype", arg_type=String, default="ternary"
                   , "--threads", arg_type=Int, default=1  # For BLAS parallelization
                   , "--chi", arg_type=Int, default=8  # Max bond dimension
                   , "--layers", arg_type=Int, default=3
                   , "--symmetry", arg_type=String, default="none"  # "none" or "group"
                   , "--block_size", arg_type=Int, default=2  # Block two sites to start
                   , "--h", arg_type=Float64, default=1.0  # External field of Ising
                   , "--J_z", arg_type=Float64, default=0.5  # ZZ coupling for XXZ
                   , "--J_xy", arg_type=Float64, default=-1.0  # XX + YY coupling for XXZ
                   , "--datafolder", arg_type=String, default="JLMdata"
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
    datafolder = pars[:datafolder]
    if symmetry == "group"
        model == "Ising" && (symmetry = "Z2")
        model == "XXZ" && (symmetry = "U1")
        pars[:symmetry] = symmetry
    end
    block_size = pars[:block_size]
    method = pars[:method]
    retraction = pars[:retraction]
    transport = pars[:transport]

    logstr = "Running demo.jl with"
    for (k, v) in pars
        logstr = logstr * "\n$k = $v"
    end
    @info(logstr)

    # Three sets of parameters are used when optimizing the MERA:
    # Used when determining which sector to give bond dimension to.
    havg_depth = 10
    pars[:initial_opt_pars] = Dict(:method => method,
                                   :retraction => retraction,
                                   :transport => transport,
                                   :gradient_delta => 1e-5,
                                   :densitymatrix_delta => 1e-5,
                                   :maxiter => 30,
                                   :miniter => 10,
                                   :havg_depth => havg_depth,
                                   :layer_iters => 1,
                                   :disentangler_iters => 1,
                                   :isometry_iters => 1)
    # Used when optimizing a MERA that has some layers expanded to desired bond dimension,
    # but not all.
    pars[:mid_opt_pars] = Dict(:method => method,
                               :retraction => retraction,
                               :transport => transport,
                               :gradient_delta => 1e-5,
                               :densitymatrix_delta => 1e-5,
                               :maxiter => 50,
                               :miniter => 10,
                               :havg_depth => havg_depth,
                               :layer_iters => 1,
                               :disentangler_iters => 1,
                               :isometry_iters => 1)
    # Used when optimizing a MERA that has all bond dimensions at the full, desired value.
    pars[:final_opt_pars] = Dict(:method => method,
                                 :retraction => retraction,
                                 :transport => transport,
                                 :gradient_delta => 1e-7,
                                 :densitymatrix_delta => 1e-7,
                                 :maxiter => 300,
                                 :miniter => 10,
                                 :havg_depth => havg_depth,
                                 :layer_iters => 1,
                                 :disentangler_iters => 1,
                                 :isometry_iters => 1)

    # Get the Hamiltonian.
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

    if symmetry == "none"
        chis = 1:chi
    else
        model == "Ising" && (chis = 2:chi)
        model == "XXZ" && (chis = 3:chi)
    end

    # Computing the magnetisation of the Ising model only makes sense if Z2 symmetry isn't
    # explicitly enforced.
    do_magnetisation = model == "Ising" && symmetry == "none"
    for chi in chis
        # For each bond dimension, get the optimized MERA and print out some interesting
        # numbers for it.
        temppars = deepcopy(pars)
        temppars[:chi] = chi
        m = DemoTools.get_optimized_mera(datafolder, model, temppars)

        energy = normalization(expect(h, m))
        rhoees = densitymatrix_entropies(m)
        do_magnetisation && (magnetization = expect(magop, remove_symmetry(m)))
        scaldims = scalingdimensions(m)

        @info("Done with bond dimension $(chi).")
        @info("Energy numerical: $energy")
        model == "Ising" && @info("Energy exact:     $(-4/pi)")
        do_magnetisation && @info("Magnetization: $(magnetization)")
        @info("rho ees: $rhoees")
        @info("Scaling dimensions: $scaldims")
    end
end

main()
