module Demo

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
                   , "--chis", arg_type=Vector{Int}, default=collect(2:4)
                   , "--layers", arg_type=Int, default=3
                   , "--symmetry", arg_type=String, default="none"
                   , "--block_size", arg_type=Int, default=2
                   , "--h", arg_type=Float64, default=1.0
                   , "--Delta", arg_type=Float64, default=-0.5
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
    if symmetry == "group"
        if model == "Ising"
            symmetry = "Z2"
        elseif model == "XXZ"
            symmetry = "U1"
        end
        pars[:symmetry] = symmetry
    end
    block_size = pars[:block_size]

    # Used when determining which sector to give bond dimension to.
    pars[:initial_opt_pars] = Dict(:densitymatrix_delta => 1e-5,
                                   :maxiter => 100,
                                   :miniter => 10,
                                   :havg_depth => 10,
                                   :layer_iters => 1,
                                   :disentangler_iters => 1,
                                   :isometry_iters => 1)
    # Used when optimizing a MERA that has some layers expanded to desired bond
    # dimension, but not all.
    pars[:mid_opt_pars] = Dict(:densitymatrix_delta => 1e-5,
                               :maxiter => 1000,
                               :miniter => 10,
                               :havg_depth => 10,
                               :layer_iters => 1,
                               :disentangler_iters => 1,
                               :isometry_iters => 1)
    # Used when optimizing a MERA that has all bond dimensions at the full,
    # desired value.
    pars[:final_opt_pars] = Dict(:densitymatrix_delta => 1e-7,
                                 :maxiter => 10000,
                                 :miniter => 10,
                                 :havg_depth => 10,
                                 :layer_iters => 1,
                                 :disentangler_iters => 1,
                                 :isometry_iters => 1)

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

    energies = Vector{Float64}()
    rhoeevects = Vector{Vector{Float64}}()
    for chi in chis
        temppars = deepcopy(pars)
        temppars[:chi] = chi
        m = get_optimized_mera("JLMdata", model, temppars)

        energy = expect(h, m)
        energy = normalize_energy(energy, dmax, block_size)
        push!(energies, energy)
        rhoees = densitymatrix_entropies(m)
        push!(rhoeevects, rhoees)
        model == "Ising" && symmetry == "none" && (magnetization = expect(magop, remove_symmetry(m)))

        @info("Done with bond dimension $(chi).")
        @info("Energy numerical: $energy")
        model == "Ising" && @info("Energy exact:     $(-4/pi)")
        model == "Ising" && symmetry == "none" && @info("Magnetization: $(magnetization)")
        @info("rho ees:")
        @info(rhoees)

        scaldims = scalingdimensions(m)
        @info("Scaling dimensions:")
        @info(scaldims)
    end

    @show rhoeevects
    @show energies
    if model == "Ising"
        energyerrs = energies .+ 4/pi
        energyerrs = abs.(energyerrs ./ energies)
        energyerrs = log.(10, energyerrs)
        @show energyerrs
    end
end

main()

end  # module
