using Test
using Gadfly
using ArgParse
using LinearAlgebra
import Cairo, Fontconfig  # For Gadfly
include("binaryMERA_simple.jl")


function parse_pars()
    settings = ArgParseSettings(autofix_names=true)
    @add_arg_table(settings
        , "--chi", arg_type=Int, default=3
        , "--layers", arg_type=Int, default=10
        , "--groupthree", arg_type=Bool, default=false
    )
    pars = parse_args(ARGS, settings; as_symbols=true)
    return pars
end

# Ising model with transverse magnetic field h (critical h=1 by default)
function build_H_Ising(h=1.0; groupthree=false)
    X = [0 1; 1 0]
    Z = [1 0; 0 -1]
    eye = Diagonal(ones(2))
    XX = kron(X, X)
    ZI = kron(Z, eye)
    IZ = kron(eye, Z)
    H2 = -(XX + h/2*(ZI+IZ))
    if groupthree
        H = H2 / 3  # See below for an explanation of the 1/3.
        for n = 3:9
            eyen2 = Diagonal(ones(2^(n-2)))
            # Terms at the borders of the blocks of three that get grouped
            # together need to be normalized differently from the ones that are
            # within blocks.
            factor = (n == 4 || n == 7) ? 1/2 : 1/3
            H = kron(H, eye) + kron(eyen2, H2)*factor
        end
    else
        H = (kron(H2, eye) + kron(eye, H2)) ./ 2.0
    end
    E = eigen(Hermitian(H))
    D = E.values
    D_max = D[end]
    # subtract largest eigenvalue, so that the spectrum is negative
    k = groupthree ? 3 : 1
    H = H - Diagonal(ones(2^(3*k)))*D_max
    d = 2^k
    H = reshape(H, repeat([2^k], 6)...)
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
    eigs = tensoreig(rho, [1,2,3], [4,5,6], hermitian=true)[1]
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

h, dmax = build_H_Ising(groupthree=groupthree)
k = groupthree ? 3 : 1
m = build_random_MERA([2^k, chi, chi])
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

