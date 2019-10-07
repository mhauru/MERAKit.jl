module TenaryMERAInfIsing

using Test
using Gadfly
using ArgParse
using LinearAlgebra
using TensorKit
import Cairo, Fontconfig  # For Gadfly
include("ternaryMERAinf.jl")
include("IsingAnyons.jl")
using .IsingAnyons
using .TernaryMERAInf

function parse_pars()
    settings = ArgParseSettings(autofix_names=true)
    @add_arg_table(settings
        , "--chi", arg_type=Int, default=4
        , "--layers", arg_type=Int, default=3
        , "--symmetry", arg_type=String, default="none"
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
    superop = build_superop_onesite(m)
    S, U = eig(superop, (1,2), (3,4))
    b = blocks(S)
    scaldims = Dict()
    for (k, v) in b
        scaldims[k] = -log.(3, abs.(diag(v)))./2  # TODO Why the abs?
    end
    return scaldims
end


cmdlinepars = parse_pars()
chi = cmdlinepars[:chi]
layers = cmdlinepars[:layers]
symmetry = cmdlinepars[:symmetry]
group = cmdlinepars[:group]
parsrandom = Dict(:energy_delta => 1e-5,
                  :energy_maxiter => 500,
                  :havg_depth => 5,
                  :uw_iters => 10,
                  :u_iters => 3,
                  :w_iters => 3)
parsinitialized = Dict(:energy_delta => 1e-6,
                       :energy_maxiter => 500,
                       :havg_depth => 5,
                       :uw_iters => 3,
                       :u_iters => 3,
                       :w_iters => 3)

h, dmax = build_H_Ising(symmetry=symmetry, group=group)
V_phys = space(h, 1)
if symmetry == "none"
    V_virt = ℂ^chi
elseif symmetry == "Z2"
    V_virt = ℂ[ℤ₂](0=>chi, 1=>chi)
elseif symmetry == "anyons"
    V_virt = RepresentationSpace{IsingAnyon}(:I => chi, :ψ => chi, :σ => chi)
else
    error("Unknown symmetry $symmetry")
end
m = build_random_MERA((V_phys, V_virt, V_virt))
fixedlayers = 0
energies = Vector{Float64}()
rhoeevects = Vector{Vector{Float64}}()
normalization(x) = normalize_energy(x, dmax, group)

minimize_expectation!(m, h, parsinitialized, normalization=normalization)

while num_translayers(m)+1 < layers
    release_transitionlayer!(m)
    fixedlayers = num_translayers(m)
    parsrandom[:energy_delta] /= 2
    parsinitialized[:energy_delta] /= 2

    println()
    minimize_expectation!(m, h, parsrandom; lowest_to_optimize=fixedlayers+1,
                          normalization=normalization)
    fixedlayers -= 1
    minimize_expectation!(m, h, parsinitialized;
                          lowest_to_optimize=fixedlayers+1,
                          normalization=normalization)
    minimize_expectation!(m, h, parsinitialized, normalization=normalization)

    energy = expect(h, m)
    energy = normalize_energy(energy, dmax, group)
    push!(energies, energy)
    rhoees = getrhoees(m)
    push!(rhoeevects, rhoees)

    println("Done with $(num_translayers(m)+1) layers.")
    println("Energy numerical: $energy")
    println("Energy exact:     $(-4/pi)")
    println("rho ees:")
    println(rhoees)
end

energyerrs = energies .+ 4/pi
energyerrs = abs.(energyerrs ./ energies)
energyerrs = log.(10, energyerrs)

scaldims = get_scaldims(m)

println("------------------------------")
@show rhoeevects
println("------------------------------")
@show energies
println("------------------------------")
@show energyerrs
println("------------------------------")
@show scaldims

eeplot = plot(y=rhoeevects[length(rhoeevects)])
energyplot = plot(y=energies)
energyerrsplot = plot(y=energyerrs)

draw(PDF("eeplot.pdf", 4inch, 3inch), eeplot)
draw(PDF("energyplot.pdf", 4inch, 3inch), energyplot)
draw(PDF("energyerrsplot.pdf", 4inch, 3inch), energyerrsplot)

end  # module
