# This script demonstrates how to use MERA.jl to find an approximation to the ground state
# of the critical Ising model.
using Random
using LinearAlgebra
using TensorKit
using MERA
# DemoTools holds some utility functions needed by this script, for creating Hamiltonians
# and storing and reading MERAs to/from disk. It exports nothing, so all calls need to be
# qualified as DemoTools.whatever.
include("demo_tools.jl")
using .DemoTools

function main()
    # Fix the random seed for the sake of reproducibility. MERA algorithms sometimes get
    # stuck in local minima, so you should usually try a few different seeds for the same
    # optimisation. This one is going to do well though.
    Random.seed!(1)

    # Create the local Hamiltonian term `h` for the critical Ising model. Use Z2 symmetric
    # tensors, and block two sites together, so that the local state space dimension is 4.
    # The blocking simply obviates the need for a trivial, unitary layer at the bottom of
    # the MERA.
    h = DemoTools.ising_hamiltonian(; symmetry = :Z2, block_size = 2)
    exact_energy = -4/pi
    V_physical = space(h, 1)

    # Choose the type of MERA to use.
    meratype = TernaryMERA
    layers = 3
    # It's typically beneficial to gradually increase the bond dimension during the
    # optimisation. We do this here using the following steps: Each element of
    # `V_virtual_steps` is a vector space for the virtual indices of the MERA, and we run
    # the optimisation for each of them, using as the starting point the result of the
    # previous step.
    # Note that for Z2 symmetric tensors we have to specify how the bond dimension is
    # divided between the symmetry sectors. It is a priori not obvious what the optimal
    # choice is, and although here it happens to be such that both sectors get the same
    # dimension, this isn't always the case. In general, finding the optimal proportions
    # requires manually trying different options.
    # Note also that we could use different vector spaces for the different indices in the
    # MERA, both the ones between layers and the ones within layers (between isometries and
    # disentanglers, making the disentanglers isometric instead of unitary). This would
    # probably be better for accuracy/speed, but using the same `V_virtual` everywhere works
    # well too, so for simplicity we stick to that.
    V_virtual_steps = (
        Z2Space(0 => 2, 1 => 2),
        Z2Space(0 => 3, 1 => 3),
        Z2Space(0 => 4, 1 => 4),
    )

    # Specify the optimisation method. We use here Riemannian LBFGS as described in
    # https://arxiv.org/abs/2007.03638
    method = :lbfgs
    verbosity = 2
    gradient_delta = 1e-7
    # In practice the above threshold for the gradient norm, which determines convergence,
    # will not be reached. Instead, we set the maximum number of iterations to be done. We
    # specify this separately for the different bond dimensions in `V_virtual_steps`.
    # Choosing these numbers is more art than science, but many different choices would work
    # roughly equally well.
    maxiter_steps = (1000, 500, 500)

    # Once the optimisation is done, the resulting MERA will be written to a file at this
    # path.
    path = "./demo_result.jld2"

    # To keep track of how the optimisation is progressing, we will every once in a while
    # print out the energy error, the entropies of the density matrices at different scales
    # in the MERA, and the scaling dimensions. For this we use the `finalize!` keyword
    # argument of `minimize_expectation`, which can be a function that is run at the end of
    # every iteration. `finalize!` could be used to do plenty of things, including modifying
    # the current MERA `m` during the optimisation, but we simply use it to print things out
    # every `checkpoint_frequency` iterations. See the documentation of MERA.jl for more
    # details on `finalize!`.
    checkpoint_frequency = 100
    function finalize!(m, energy, g, repnum)
        repnum % checkpoint_frequency != 0 && (return m, energy, g)
        rhoees = densitymatrix_entropies(m)
        scaldims = scalingdimensions(m)
        @info("Energy error: $(energy - exact_energy)")
        @info("Density matrix entropies: $rhoees")
        @info("Scaling dimensions: $scaldims")
        return m, energy, g
    end

    # The main loop, that for every vector space in `V_virtual_steps` optimises the MERA.
    # For the first one, we use a randomly initialised MERA as the starting point, for the
    # latter ones we expand the bond dimension by padding tensors with zeros.
    local m
    for i in 1:length(V_virtual_steps)
        V_virtual = V_virtual_steps[i]
        if i == 1
            @info("Creating a random MERA with bond dimension $(dim(V_virtual)).")
            # The lowest index is of the physical bond dimension, the others are virtual.
            Vs = (V_physical, ntuple(x -> V_virtual, layers-1)...)
            m = random_MERA(meratype, Float64, Vs)
        else
            @info("Expanding to bond dimension $(dim(V_virtual)).")
            for j in 2:layers
                m = expand_bonddim(m, j-1, V_virtual)
                m = expand_internal_bonddim(m, j, V_virtual)
            end
        end
        # Many more optimisation parameters can be specified, see the documentation on
        # `minimize_expectation`.
        pars = (
            gradient_delta = gradient_delta,
            maxiter = maxiter_steps[i],
            method = method,
            verbosity = verbosity
        )
        m = minimize_expectation(m, h, pars; finalize! = finalize!)
    end

    # Write the resulting MERA to disc. It can then be read using
    # `DemoTools.load_mera(path)`.
    DemoTools.store_mera(path, m)
    return m
end

# Run the actual script. The whole thing is in a function mainly to avoid polluting the
# global namespace.
main()
