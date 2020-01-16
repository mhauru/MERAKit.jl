module TernaryMERAInfModel

using ArgParse
using LinearAlgebra
using TensorKit
include("ternaryMERAinf_modeltools.jl")
using .TernaryMERAInfModelTools
using .TernaryMERAInfModelTools.MERA

function parse_pars()
    settings = ArgParseSettings(autofix_names=true)
    @add_arg_table(settings
                   , "--model", arg_type=String, default="Ising"
                   , "--threads", arg_type=Int, default=1
                   , "--chis", arg_type=Vector{Int}, default=collect(4:4)
                   , "--layers", arg_type=Int, default=3
                   , "--symmetry", arg_type=String, default="group"
                   , "--block", arg_type=Int, default=2
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
    block = pars[:block]
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
        h, dmax = build_H_Ising(pars[:h]; symmetry=symmetry, block=block)
        symmetry == "none" && (magop = build_magop(block=block))
    elseif model == "XXZ"
        h, dmax = build_H_XXZ(pars[:Delta]; symmetry=symmetry, block=block)
    else
        msg = "Unknown model $(model)."
        throw(ArgumentError(msg))
    end
    normalization(x) = normalize_energy(x, dmax, block)

    energies = Vector{Float64}()
    rhoeevects = Vector{Vector{Float64}}()
    for chi in chis
        temppars = deepcopy(pars)
        temppars[:chi] = chi
        m = get_optimized_mera("JLMdata", model, temppars)

        energy = expect(h, m)
        energy = normalize_energy(energy, dmax, block)
        push!(energies, energy)
        rhoees = densitymatrix_entropies(m)
        push!(rhoeevects, rhoees)
        model == "Ising" && symmetry == "none" && (magnetization = expect(magop, remove_symmetry(m)))

        println("Done with bond dimension $(chi).")
        println("Energy numerical: $energy")
        model == "Ising" && println("Energy exact:     $(-4/pi)")
        model == "Ising" && symmetry == "none" && println("Magnetization: $(magnetization)")
        println("rho ees:")
        println(rhoees)

        scaldims = get_scaldims(m)
        println("Scaling dimensions:")
        println(scaldims)
    end

    println("------------------------------")
    @show rhoeevects
    println("------------------------------")
    @show energies
    if model == "Ising"
        println("------------------------------")
        energyerrs = energies .+ 4/pi
        energyerrs = abs.(energyerrs ./ energies)
        energyerrs = log.(10, energyerrs)
        @show energyerrs
    end
end

main()

end  # module
