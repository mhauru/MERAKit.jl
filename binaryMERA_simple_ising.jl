using Test
using Gadfly
using ArgParse
using LinearAlgebra
using TensorKit
import Cairo, Fontconfig  # For Gadfly
include("binaryMERA_simple.jl")
include("IsingAnyons.jl")
using .IsingAnyons

function parse_pars()
    settings = ArgParseSettings(autofix_names=true)
    @add_arg_table(settings
        , "--chi", arg_type=Int, default=2
        , "--layers", arg_type=Int, default=10
        , "--groupthree", arg_type=Bool, default=false
        , "--symmetry", arg_type=String, default="none"
    )
    pars = parse_args(ARGS, settings; as_symbols=true)
    return pars
end

# Ising model with transverse magnetic field h (critical h=1 by default)
function build_H_Ising(h=1.0; groupthree=false, symmetry="none")
    k = groupthree ? 3 : 1
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
        H2 = -(XX + h/2 * (ZI+IZ))
    elseif symmetry == "anyons"
        V = RepresentationSpace{IsingAnyon}(:I => 0, :ψ => 0, :σ => 1)
        H2 = TensorMap(zeros, Float64, V ⊗ V ← V ⊗ V)
        H2.data[IsingAnyon(:I)] .= 1.0
        H2.data[IsingAnyon(:ψ)] .= -1.0
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
        H2 = -(XX + h/2 * (ZI+IZ))
    else
        error("Unknown symmetry $symmetry")
    end
    if groupthree
        H = H2 / 3  # See below for an explanation of the 1/3.
        for n = 3:9
            prodspace = reduce(⊗, repeat([ℂ^2], n-2))
            eyen2 = TensorMap(I, Float64, prodspace ← prodspace)
            # Terms at the borders of the blocks of three that get grouped
            # together need to be normalized differently from the ones that are
            # within blocks.
            factor = (n == 4 || n == 7) ? 1/2 : 1/3
            H_inds = vcat(collect(1:n-1), collect(n+1:2*n-1))
            eye_inds = [n, 2*n]
            out_inds = collect(1:2*n)
            eval(:(@tensor H_term1[$(out_inds...)] := $H[$(H_inds...)] * $eye[$(eye_inds...)]))
            H2_inds = [n-1, n, 2*n-1, 2*n]
            eyen2_inds = vcat(collect(1:n-2), collect(n+1:2*n-2))
            eval(:(@tensor H_term2[$(out_inds...)] := $eyen2[$(eyen2_inds...)] * $H2[$(H2_inds...)]))
            H = H_term1 + H_term2*factor
        end
        topspaces = space(H, (1,2,3))
        fusetop = TensorMap(I, fuse(topspaces) ← topspaces)
        bottomspaces = space(H, (1,2,3))
        fusebottom = TensorMap(I, bottomspaces ← fuse(bottomspaces))
        @tensor(
                H_new[-1,-2,-3,-11,-12,-13] :=
                fusetop[-1,1,2,3] * fusetop[-2,4,5,6] * fusetop[-3,7,8,9]
                * H[1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19]
                * fusebottom[11,12,13,-11] * fusebottom[14,15,16,-12] * fusebottom[17,18,19,-13]
               )
        H = H_new
    else
        @tensor HL[-1,-2,-3,-11,-12,-13] := H2[-1,-2,-11,-12] * eye[-3,-13]
        @tensor HR[-1,-2,-3,-11,-12,-13] := eye[-1,-11] * H2[-2,-3,-12,-13 ]
        H = (HL + HR) / 2.0
    end
    # Subtract a constant, so that the spectrum is negative
    # TODO Switch to using an eigendecomposition?
    D_max = norm(H)
    bigeye = permuteind(TensorMap(I, space(H, (1,2,3)) ← space(H, (1,2,3))),
                        (1,2,3,4,5,6))
    H = H - bigeye*D_max
    return H, D_max
end

function minimize_expectation_layerbylayer!(m, h, pars)
    horig = asc_threesite(h, m; endscale=lowest_to_optimize)
    energy = Inf
    energy_change = Inf
    counter = 0
    while (abs(energy_change) > pars[:energy_delta]
           && counter < pars[:energy_maxiter])
        h = horig
        rhos = minimize_expectation_buildrhos(m, lowest_to_optimize)

        counter += 1
        oldenergy = energy
        for l in lowest_to_optimize:m.depth
            rho = rhos[l-lowest_to_optimize+1]
            u, w = get_uw(m, l)
            u, w = minimize_expectation_uw(h, u, w, rho, pars)
            set_uw!(m, u, w, l)
            h = asc_threesite(h, m; startscale=l, endscale=l+1)
        end
        top = minimize_expectation_top(h, pars)
        set_top!(m, top)
        energy = expect(h, m, opscale=m.depth+1, evalscale=m.depth+1)
        energy_change = (energy - oldenergy)/energy
        @printf("Energy = %.5e,    change = %.5e,    counter = %d\n",
                energy, energy_change, counter)
    end
    return m
end

function getrhoee(rho)
    eigs = eigen(rho, (1,2,3), (4,5,6))[1]
    eigs = real.(diag(convert(Array, eigs)))
    if sum(abs.(eigs[eigs .<= 0.])) > 1e-13
        warn("Significant negative eigenvalues: $eigs")
    end
    eigs = eigs[eigs .> 0.]
    S = - dot(eigs, log.(eigs))
    return S
end

function getrhoees(m)
    rhos = buildrhos(m)
    ees = Vector{Float64}()
    for rho in rhos
        ee = getrhoee(rho)
        push!(ees, ee)
    end
    return ees
end

function normalize_energy(energy, dmax, k)
    energy = (energy + dmax)/k
    return energy
end

cmdlinepars = parse_pars()
chi = cmdlinepars[:chi]
layers = cmdlinepars[:layers]
groupthree = cmdlinepars[:groupthree]
symmetry = cmdlinepars[:symmetry]
parsrandom = Dict(:energy_delta => 1e-5,
                  :energy_maxiter => 500,
                  :uw_iters => 10,
                  :u_iters => 3,
                  :w_iters => 3)
parsinitialized = Dict(:energy_delta => 1e-6,
                       :energy_maxiter => 500,
                       :uw_iters => 3,
                       :u_iters => 3,
                       :w_iters => 3)

h, dmax = build_H_Ising(groupthree=groupthree, symmetry=symmetry)
k = groupthree ? 3 : 1
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
normalization(x) = normalize_energy(x, dmax, k)

minimize_expectation!(m, h, parsinitialized, normalization=normalization)

while m.depth < layers
    addrandomlayer!(m, m.depth+1)
    fixedlayers = m.depth - 1
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
    energy = normalize_energy(energy, dmax, k)
    push!(energies, energy)
    rhoees = getrhoees(m)
    push!(rhoeevects, rhoees)

    println("Done with $(m.depth) layers.")
    println("Energy numerical: $energy")
    println("Energy exact:     $(-4/pi)")
    println("rho ees:")
    println(rhoees)
end

energyerrs = energies .+ 4/pi
energyerrs = abs.(energyerrs ./ energies)
energyerrs = log.(energyerrs, 10)

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

