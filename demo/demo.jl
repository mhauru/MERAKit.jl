# A script that optimizes a MERA for either the Ising or XXZ model, and computes some
# physical quantities from it. The MERAs built are stored on disk.

using ArgParse
using LinearAlgebra
using TensorKit
include("demo_tools.jl")
using .DemoTools

function parse_pars()
    settings = ArgParseSettings(autofix_names=true)
    @add_arg_table!(settings
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
    datafolder = pars[:datafolder]
    if symmetry == "group"
        model == "Ising" && (symmetry = "Z2")
        model == "XXZ" && (symmetry = "U1")
        pars[:symmetry] = symmetry
    end
    block_size = pars[:block_size]

    logstr = "Running demo.jl with"
    for (k, v) in pars
        logstr = logstr * "\n$k = $v"
    end
    @info(logstr)

    # Three sets of parameters are used when optimizing the MERA:
    # Used when determining which sector to give bond dimension to.
    scale_invariant_eps = pars[:scale_invariant_eps]
    pars[:initial_opt_pars] = (
                               maxiter = 30,
                               isometries_only_iters = 10,
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
    # Used when optimizing a MERA that has some layers expanded to desired bond dimension,
    # but not all.
    pars[:mid_opt_pars] = merge(pars[:initial_opt_pars],
                                (maxiter = 100, isometry_iters = 0))
    # Used when optimizing a MERA that has all bond dimensions at the full, desired value.
    pars[:final_opt_pars] = merge(pars[:initial_opt_pars],
                                  (gradient_delta = 1e-7,
                                   maxiter = 1000,
                                   isometry_iters = 0))

    # Get the Hamiltonian.
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

        energy = expect(h, m)
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
