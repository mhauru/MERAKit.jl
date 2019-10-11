module TenaryMERAInfIsing

using Test
using Gadfly
using ArgParse
using LinearAlgebra
using TensorKit
using Serialization
using MAT
using KrylovKit
import Cairo, Fontconfig  # For Gadfly
include("ternaryMERAinf.jl")
include("IsingAnyons.jl")
using .IsingAnyons
using .TernaryMERAInf

version = 1.0

function parse_pars()
    settings = ArgParseSettings(autofix_names=true)
    @add_arg_table(settings
                   , "--chis", arg_type=Vector{Int}, default=collect(1:16)
                   , "--layers", arg_type=Int, default=3
                   , "--symmetry", arg_type=String, default="Z2"
                   , "--group", arg_type=Int, default=2
    )
    pars = parse_args(ARGS, settings; as_symbols=true)
    return pars
end

# Ising model with transverse magnetic field h (critical h=1 by default)
function build_H_Ising(h=1.0; symmetry="none", group=1)
    if symmetry == "Z2"
        V = ℂ[ℤ₂](0=>1, 1=>1)
        Z = TensorMap(zeros, Float64, V ← V)
        Z.data[ℤ₂(0)] .= 1.0
        Z.data[ℤ₂(1)] .= -1.0
        eye = TensorMap(I, Float64, V ← V)
        @tensor ZI[-1,-2,-11,-12] := Z[-1,-11] * eye[-2,-12]
        @tensor IZ[-1,-2,-11,-12] := eye[-1,-11] * Z[-2,-12]
        XX = Tensor(zeros, Float64, V ⊗ V ⊗ V' ⊗ V')
        XX.data[ℤ₂(0)] .= [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
        H = -(XX + h/2 * (ZI+IZ))
    elseif symmetry == "anyons"
        V = RepresentationSpace{IsingAnyon}(:I => 0, :ψ => 0, :σ => 1)
        H = TensorMap(zeros, Float64, V ⊗ V ← V ⊗ V)
        H.data[IsingAnyon(:I)] .= 1.0
        H.data[IsingAnyon(:ψ)] .= -1.0
        eye = TensorMap(I, Float64, V ← V)
    elseif symmetry == "none"
        V = ℂ^2
        X = TensorMap(zeros, Float64, V ← V)
        Z = TensorMap(zeros, Float64, V ← V)
        eye = TensorMap(I, Float64, V ← V)
        X.data .= [0.0 1.0; 1.0 0.0]
        Z.data .= [1.0 0.0; 0.0 -1.0]
        @tensor XX[-1,-2,-11,-12] := X[-1,-11] * X[-2,-12]
        @tensor ZI[-1,-2,-11,-12] := Z[-1,-11] * eye[-2,-12]
        @tensor IZ[-1,-2,-11,-12] := eye[-1,-11] * Z[-2,-12]
        H = -(XX + h/2 * (ZI+IZ))
    else
        error("Unknown symmetry $symmetry")
    end
    while group > 1
        VL = space(H, 1)
        VR = space(H, 2)
        eyeL = TensorMap(I, Float64, VL ← VL)
        eyeR = TensorMap(I, Float64, VR ← VR)
        @tensor(Hcross[-1,-2,-3,-4,-11,-12,-13,-14] := 
                eyeR[-1,-11] * H[-2,-3,-12,-13] * eyeL[-4,-14])
        @tensor(Hleft[-1,-2,-3,-4,-11,-12,-13,-14] :=
                H[-1,-2,-11,-12] * eyeL[-3,-13] * eyeR[-4,-14])
        @tensor(Hright[-1,-2,-3,-4,-11,-12,-13,-14] :=
                eyeL[-1,-11] * eyeR[-2,-12] * H[-3,-4,-13,-14])
        H_new_unfused = Hcross + 0.5*(Hleft + Hright)
        fusionspace = space(H, (1,2))
        fusetop = TensorMap(I, fuse(fusionspace) ← fusionspace)
        fusebottom = TensorMap(I, fusionspace ← fuse(fusionspace))
        @tensor(
                H_new[-1,-2,-11,-12] :=
                fusetop[-1,1,2] * fusetop[-2,3,4]
                * H_new_unfused[1,2,3,4,11,12,13,14]
                * fusebottom[11,12,-11] * fusebottom[13,14,-12]
               )
        H = H_new
        group /= 2
    end
    if group != 1
        msg = "`group` needs to be a power of 2"
        throw(ArgumentError(msg))
    end
    H = permuteind(H, (1,2), (3,4))
    # Subtract a constant, so that the spectrum is negative
    # TODO Switch to using an eigendecomposition?
    D_max = norm(H)
    bigeye = TensorMap(I, codomain(H) ← domain(H))
    H = H - bigeye*D_max
    return H, D_max
end

function getrhoee(rho)
    eigs = eigen(rho, (1,2), (3,4))[1]
    eigs = real.(diag(convert(Array, eigs)))
    if sum(abs.(eigs[eigs .<= 0.])) > 1e-13
        warn("Significant negative eigenvalues: $eigs")
    end
    eigs = eigs[eigs .> 0.]
    S = - dot(eigs, log.(eigs))
    return S
end

function getrhoees(m)
    rhos = build_rhos(m)
    ees = Vector{Float64}()
    for rho in rhos
        ee = getrhoee(rho)
        push!(ees, ee)
    end
    return ees
end

function normalize_energy(energy, dmax, group)
    energy = (energy + dmax)/group
    return energy
end

function build_superop_onesite(m)
    w = get_w(m, Inf)
    w_dg = w'
    @tensor(superop[-1,-2,-11,-12] := w[-1,1,-11,2] * w_dg[1,-12,2,-2])
    return superop
end

function get_scaldims(m)
    superoperator = :onesite  # TODO Hard constant.
    if superoperator == :onesite
        superop = build_superop_onesite(m)
        S, U = eig(superop, (1,2), (3,4))
    elseif superoperator == :twosite
        V = get_outputspace(m, Inf)
        typ = eltype(get_u(m, Inf))
        chi = dim(V)
        maxmany = Int(ceil(chi^2/2))
        howmany = min(maxmany, 20)   # TODO Hard constant.
        f(x) = asc_twosite(x, m; endscale=num_translayers(m)+2,
                           startscale=num_translayers(m)+1)
        x0 = Tensor(randn, typ, V ⊗ V ⊗ V' ⊗ V')
        S, U, info = eigsolve(f, x0, howmany, :LM)
    else
        raise(ArgumentError("Unknown superoperator type $(superoperator)."))
    end
    if isa(S, Tensor) || isa(S, TensorMap)
        b = blocks(S)
        scaldims = Dict()
        for (k, v) in b
            scaldims[k] = sort(-log.(3, abs.(real(diag(v)))))  # TODO Why the abs?
        end
    else
        scaldims = sort(-log.(3, abs.(real(S))))  # TODO Why the abs?
    end
    return scaldims
end

function optimize_layerbylayer!(m, h, fixedlayers, normalization, opt_pars)
    while fixedlayers >= 0
        minimize_expectation!(m, h, opt_pars;
                              lowest_to_optimize=fixedlayers+1,
                              normalization=normalization)
        fixedlayers -= 1
    end
end

function store_mera(path, m)
    # TODO JLD, JLD2 and plain HDF5 all suck, and fail with my MERA type.
    # Switch them to if they stop sucking.
    serialize(path, m)
end

function load_mera(path)
    # TODO JLD, JLD2 and plain HDF5 all suck, and fail with my MERA type.
    # Switch them to if they stop sucking.
    m = deserialize(path)
    return m
end

function store_mera_matlab(path, m)
    d = Dict{String, Array}()
    for i in 1:(num_translayers(m)+1)
        u, w = map(x -> convert(Array, x), get_uw(m, i))
        # Permute indices to match Mathias's convention.
        d["u$i"] = permutedims(u, (3,4,1,2))
        d["w$i"] = permutedims(w, (2,3,4,1))
    end
    matwrite(path, d)
end

function load_mera_matlab(path)
    d = matread(path)
    chi = size(d["u"], 1)
    V = ℂ^chi
    u = TensorMap(zeros, Complex{Float64}, V ⊗ V ← V ⊗ V)
    w = TensorMap(zeros, Complex{Float64}, V ← V ⊗ V ⊗ V)
    u.data[:] = permutedims(d["u"], (3,4,1,2))
    w.data[:] = permutedims(d["w"], (4,1,2,3))
    m = MERA([(u, w)])
end

function get_optimized_mera(h, normalization, pars; loadfromdisk=true)
    chi = pars[:chi]
    layers = pars[:layers]
    symmetry = pars[:symmetry]
    group = pars[:group]
    global version

    datafolder = "./JLDdata"
    mkpath(datafolder)
    filename = "MERA_$(chi)_$(group)_$(symmetry)_$(layers)_$(version)"
    path = "$datafolder/$filename.jls"

    if loadfromdisk && isfile(path)
        m = load_mera(path)
        return m
    else
        if chi == 1
            V_phys = space(h, 1)
            if symmetry == "none"
                V_virt = ℂ^1
            elseif symmetry == "Z2"
                V_virt = ℂ[ℤ₂](0=>1, 1=>0)
            elseif symmetry == "anyons"
                V_virt = RepresentationSpace{IsingAnyon}(:I => 1,
                                                         :ψ => 0,
                                                         :σ => 0)
            else
                error("Unknown symmetry $symmetry")
            end

            Vs = tuple(V_phys, repeat([V_virt], layers-1)...)
            m = build_random_MERA(Vs)

            optimize_layerbylayer!(m, h, 0, normalization,
                                   pars[:final_opt_pars])
            store_mera(path, m)
            return m
        else
            prevpars = deepcopy(pars)
            prevpars[:chi] -= 1
            m = get_optimized_mera(h, normalization, prevpars; loadfromdisk=loadfromdisk)

            if symmetry == "none"
                allsectors = (Trivial(),)
            elseif symmetry == "Z2"
                allsectors = (ℤ₂(0), ℤ₂(1))
            else
                allsectors = (FibonacciAnyon(:I),
                              FibonacciAnyon(:ψ),
                              FibonacciAnyon(:σ))
            end

            for i in 1:num_translayers(m)
                V = get_outputspace(m, i)
                d = dim(V)
                if d < chi  # This should always be true.
                    expanded_meras = Dict()
                    for s in allsectors
                        ms = deepcopy(m)
                        ds = dim(V, s)
                        chi_s = ds + (chi - d)
                        expand_bonddim!(ms, i, Dict(s => chi_s))
                        msg = "Expanded layer $i to bond dimenson $chi_s in sector $s."
                        @info(msg)
                        optimize_layerbylayer!(ms, h, i-1, normalization,
                                               pars[:initial_opt_pars])
                        expanded_meras[s] = ms
                    end
                    expanded_meras_array = collect(expanded_meras)
                    minindex = argmin(map(pair -> expect(h, pair[2]),
                                          expanded_meras_array))
                    s, m = expanded_meras_array[minindex]
                    msg = "Expanding sector $s yielded the lowest energy, keeping that. Optimizing it further."
                    @info(msg)
                    opt_pars = (i == num_translayers(m) ?
                                pars[:final_opt_pars]
                                : pars[:mid_opt_pars]
                               )
                    optimize_layerbylayer!(m, h, 0, normalization, opt_pars)
                end
            end

            store_mera(path, m)
            return m
        end
    end
end

function main()
    pars = parse_pars()
    chis = pars[:chis]
    layers = pars[:layers]
    symmetry = pars[:symmetry]
    group = pars[:group]
    # Used when determining which sector to give bond dimension to.
    pars[:initial_opt_pars] = Dict(:rho_delta => 1e-5,
                                   :maxiter => 100,
                                   :miniter => 10,
                                   :havg_depth => 10,
                                   :uw_iters => 1,
                                   :u_iters => 1,
                                   :w_iters => 1)
    # Used when optimizing a MERA that has some layers expanded to desired bond
    # dimension, but not all.
    pars[:mid_opt_pars] = Dict(:rho_delta => 1e-5,
                                   :maxiter => 1000,
                                   :miniter => 10,
                                   :havg_depth => 10,
                                   :uw_iters => 1,
                                   :u_iters => 1,
                                   :w_iters => 1)
    # Used when optimizing a MERA that has all bond dimensions at the full,
    # desired value.
    pars[:final_opt_pars] = Dict(:rho_delta => 1e-7,
                                 :maxiter => 10000,
                                 :miniter => 10,
                                 :havg_depth => 10,
                                 :uw_iters => 1,
                                 :u_iters => 1,
                                 :w_iters => 1)

    h, dmax = build_H_Ising(symmetry=symmetry, group=group)
    normalization(x) = normalize_energy(x, dmax, group)

    energies = Vector{Float64}()
    rhoeevects = Vector{Vector{Float64}}()
    for chi in chis
        temppars = deepcopy(pars)
        temppars[:chi] = chi
        m = get_optimized_mera(h, normalization, temppars)

        energy = expect(h, m)
        energy = normalize_energy(energy, dmax, group)
        push!(energies, energy)
        rhoees = getrhoees(m)
        push!(rhoeevects, rhoees)

        println("Done with bond dimension $(chi).")
        println("Energy numerical: $energy")
        println("Energy exact:     $(-4/pi)")
        println("rho ees:")
        println(rhoees)

        scaldims = get_scaldims(m)
        println("Scaling dimensions:")
        println(scaldims)
    end

    energyerrs = energies .+ 4/pi
    energyerrs = abs.(energyerrs ./ energies)
    energyerrs = log.(10, energyerrs)

    println("------------------------------")
    @show rhoeevects
    println("------------------------------")
    @show energies
    println("------------------------------")
    @show energyerrs

    eeplot = plot(y=rhoeevects[length(rhoeevects)])
    energyplot = plot(y=energies)
    energyerrsplot = plot(y=energyerrs)

    draw(PDF("eeplot.pdf", 4inch, 3inch), eeplot)
    draw(PDF("energyplot.pdf", 4inch, 3inch), energyplot)
    draw(PDF("energyerrsplot.pdf", 4inch, 3inch), energyerrsplot)
end

main()

end  # module
