using Test
using TensorKit
using LinearAlgebra
using MERA
using Logging
using Random

function particle_number_operator(::Type{Z2Space})
    V = Z2Space(ℤ₂(0) => 1, ℤ₂(1) => 1)
    z = TensorMap(zeros, Float64, V ← V)
    z.data[ℤ₂(0)] .= 0.0
    z.data[ℤ₂(0)] .= 1.0
    return z
end

function particle_number_operator(::Type{ComplexSpace})
    V = ℂ^2
    z = TensorMap(zeros, Float64, V ← V)
    z.data[1,1] = 0.0
    z.data[2,2] = 1.0
    return z
end

function random_space(::Type{Z2Space}, dlow=2, dhigh=6)
    dtotal = rand(dlow:dhigh)
    d0 = rand(1:dtotal-1)
    d1 = dtotal - d0
    V = Z2Space(ℤ₂(0) => d0, ℤ₂(1) => d1)
    return V
end

function random_space(::Type{ComplexSpace}, dlow=1, dhigh=8)
    d = rand(dlow:dhigh)
    V = ℂ^d
    return V
end

"""
Generate vector spaces for a MERA of `meratype`, with `n` layers, and `T` vector space type.
"""
function random_layerspaces(T, meratype, n, dlow=3, dhigh=6)
    width = causal_cone_width(meratype)
    V = random_space(T, dlow, dhigh)
    spaces = [V]
    for i in 1:(n-1)
        V_prev = V
        V_prev_fusion = fuse(reduce(⊗, repeat([V_prev], width)))
        V = random_space(T, dlow, dhigh)
        # If a layer from V_prev to V would be of such a dimension that it couldn't be
        # isometric, try generating another V until success. Since V = V_prev is a valid
        # choice, this will eventually terminate.
        while infinum(V_prev_fusion, V) != V
            V = random_space(T, dlow, dhigh)
        end
        push!(spaces, V)
    end
    return spaces
end

"""
Generate vector spaces for the layer-internal indices of a MERA of `meratype`, that has
`extspaces` as its interlayer spaces.
"""
function random_internalspaces(extspaces, meratype)
    width = (meratype == TernaryMERA ? 3 :
             meratype in (BinaryMERA, ModifiedBinaryMERA) ? 2 : nothing)
    intspaces = []
    for i in 1:length(extspaces)
        Vext = fuse(extspaces[i])  # The fuse just removes nested product state structure.
        Vnext = i == length(extspaces) ? Vext : extspaces[i+1]
        sects = Dict()
        for s in sectors(Vext)
            d = dim(Vext, s)
            rand(Bool) && d > 1 && (d = d - 1)
            sects[s] = d
        end
        Vint = spacetype(Vext)(sects...)
        # Check that we didn't accidentally make the isometric part non-isometric. If we
        # did, just make Vint = Vext.
        Vint_fusion = fuse(reduce(⊗, repeat([Vint], width)))
        infinum(Vint_fusion, Vnext) != Vnext && (Vint = Vext)
        push!(intspaces, Vint)
    end
    return intspaces
end

"""
For each layer, generate a random operator above and below it (not necessarily Hermitian),
and confirm that ascending the lower or descending the upper one both lead to the same
expectation value (trace of product).
"""
function test_ascend_and_descend(meratype, spacetype)
    layers = 4
    spaces = random_layerspaces(spacetype, meratype, layers)
    intspaces = random_internalspaces(spaces, meratype)
    m = random_MERA(meratype, spaces, intspaces; random_disentangler=true)
    width = causal_cone_width(meratype)

    for i in 1:layers
        Vin = inputspace(m, i)
        Vout = outputspace(m, i)
        upper_space = reduce(⊗, repeat([Vin], width))
        lower_space = reduce(⊗, repeat([Vout], width))
        randomop1 = TensorMap(randn, ComplexF64, upper_space ← upper_space)
        randomop2 = TensorMap(randn, ComplexF64, lower_space ← lower_space)
        down1 = descend(randomop1, m, startscale=i+1, endscale=i)
        up2 = ascend(randomop2, m, startscale=i, endscale=i+1)
        e1 = dot(down1, randomop2)
        e2 = dot(randomop1, up2)
        @test e1 ≈ e2
    end
end

"""
Test that the expectation value of the identity is 1.0, regardless of which layer we
evaluate it at.
"""
function test_expectation_of_identity(meratype, spacetype)
    layers = 4
    spaces = random_layerspaces(spacetype, meratype, layers)
    intspaces = random_internalspaces(spaces, meratype)
    m = random_MERA(meratype, spaces, intspaces; random_disentangler=true)
    V = outputspace(m, 1)
    eye = id(V)
    for i in 1:(layers+1)
        @test expect(eye, m; evalscale=i) ≈ 1.0
    end
end

"""
Test that the expectation value of a random Hermitian operator does not depend on the layer
that we evaluate it at.
"""
function test_expectation_evalscale(meratype, spacetype)
    layers = 4
    spaces = random_layerspaces(spacetype, meratype, layers)
    intspaces = random_internalspaces(spaces, meratype)
    m = random_MERA(meratype, spaces, intspaces; random_disentangler=true)
    V = outputspace(m, 1)
    randomop = TensorMap(randn, ComplexF64, V ← V)
    randomop = (randomop + randomop')/2
    expectation = expect(randomop, m)
    for i in 1:(layers+1)
        expectation_i = expect(randomop, m; evalscale=i)
        @test expectation_i ≈ expectation
    end
end

"""
Test that pseudoserializing and depseudoserializing back does not change the expectation
value of a random Hermitian operator.
"""
function test_pseudoserialization(meratype, spacetype)
    layers = 4
    spaces = random_layerspaces(spacetype, meratype, layers)
    intspaces = random_internalspaces(spaces, meratype)
    m = random_MERA(meratype, spaces, intspaces; random_disentangler=true)
    V = outputspace(m, 1)
    randomop = TensorMap(randn, ComplexF64, V ← V)
    randomop = (randomop + randomop')/2
    expectation = expect(randomop, m)
    ser = pseudoserialize(m)
    m_reco = depseudoserialize(ser...)
    new_expectation = expect(randomop, m_reco)
    @test new_expectation ≈ expectation
end

"""
Confirm that expanding bond dimensions does not change the expectation value of a random
Hermitian operator.
"""
function test_expand_bonddim(meratype, spacetype)
    layers = 4
    spaces = random_layerspaces(spacetype, meratype, layers, 4)
    intspaces = random_internalspaces(spaces, meratype)
    m = random_MERA(meratype, spaces, intspaces; random_disentangler=true)
    V = outputspace(m, 1)
    randomop = TensorMap(randn, ComplexF64, V ← V)
    randomop = (randomop + randomop')/2
    expectation = expect(randomop, m)
    sf = scalefactor(meratype)

    # Expand the interlayer spaces, check that expectation value is preserved.
    for i in 1:(layers-1)
        V = inputspace(m, i)
        expandable_sectors = sectors(V)
        Vint = fuse(⊗(repeat([internalspace(m, i)], sf)...))
        expandable_sectors = [s for s in expandable_sectors if dim(V, s) < dim(Vint, s)]
        if i == layers-1
            Vint = fuse(⊗(repeat([internalspace(m, i+1)], sf)...))
            expandable_sectors = [s for s in expandable_sectors if dim(V, s) < dim(Vint, s)]
        end
        newdims = Dict(s => dim(V, s) + 1 for s in expandable_sectors)
        m = expand_bonddim!(m, i, newdims)
    end
    new_expectation = expect(randomop, m)
    @test new_expectation ≈ expectation

    # Expand the intralayer spaces, check that expectation value is preserved.
    for i in 1:(layers-1)
        V = internalspace(m, i)
        Vout = outputspace(m, i)
        newdims = Dict(s => dim(V, s) < dim(Vout, s) ? dim(V, s) + 1 : dim(V, s)
                       for s in sectors(V)
                      )
        m = expand_internal_bonddim!(m, i, newdims)
    end
    new_expectation = expect(randomop, m)
    @test new_expectation ≈ expectation
end

"""
Confirm that releasing a does not change the expectation value of a random Hermitian
operator.
"""
function test_release_layer(meratype, spacetype)
    layers = 4
    spaces = random_layerspaces(spacetype, meratype, layers)
    intspaces = random_internalspaces(spaces, meratype)
    m = random_MERA(meratype, spaces, intspaces; random_disentangler=true)
    V = outputspace(m, 1)
    randomop = TensorMap(randn, ComplexF64, V ← V)
    randomop = (randomop + randomop')/2
    expectation = expect(randomop, m)
    m = release_transitionlayer!(m)
    new_expectation = expect(randomop, m)
    @test new_expectation ≈ expectation
end

"""
Create a random MERA and operator, evaluate the expectation values, strip both of their
symmetry structure, and confirm that the expectation value hasn't changed.
"""
function test_remove_symmetry(meratype, spacetype)
    layers = 4
    spaces = random_layerspaces(spacetype, meratype, layers)
    intspaces = random_internalspaces(spaces, meratype)
    m = random_MERA(meratype, spaces, intspaces; random_disentangler=true)
    V = outputspace(m, 1)
    randomop = TensorMap(randn, ComplexF64, V ← V)
    randomop = (randomop + randomop')/2
    expectation = expect(randomop, m)
    m_nosym = remove_symmetry(m)
    randomop_nosym = remove_symmetry(randomop)
    expectation_nosym = expect(randomop_nosym, m_nosym)
    @test expectation ≈ expectation_nosym
end

"""
Create a random MERA and operator, evaluate the expectation values, strip both of their
symmetry structure, and confirm that the expectation value hasn't changed.
"""
function test_reset_storage(meratype, spacetype)
    layers = 4
    spaces = random_layerspaces(spacetype, meratype, layers)
    intspaces = random_internalspaces(spaces, meratype)
    m = random_MERA(meratype, spaces, intspaces; random_disentangler=true)
    V = outputspace(m, 1)
    randomop = TensorMap(randn, ComplexF64, V ← V)
    randomop = (randomop + randomop')/2
    expectation_orig = expect(randomop, m)

    reset_storage!(m, 2)
    expectation_reset = expect(randomop, m)
    @test expectation_orig ≈ expectation_reset

    reset_storage!(m)
    expectation_reset = expect(randomop, m)
    @test expectation_orig ≈ expectation_reset

    reset_operator_storage!(m, randomop)
    expectation_reset = expect(randomop, m)
    @test expectation_orig ≈ expectation_reset

    reset_operator_storage!(m)
    expectation_reset = expect(randomop, m)
    @test expectation_orig ≈ expectation_reset

    # TODO Consider writing tests that check that a) storage is correctly reset after
    # assigning new tensors, b) storage isn't needlessly reset when assigning tensors, i.e.
    # no unnecessary recomputation is done.
end

"""
Create a MERA that breaks the isometricity condition, by summing up two random MERAs.
Restore isometricity with projectisometric and projectisometric!, and confirm that the
expectation value of the identity is indeed 1 after this.
"""
function test_projectisometric(meratype, spacetype)
    layers = 4
    spaces = random_layerspaces(spacetype, meratype, layers)
    intspaces = random_internalspaces(spaces, meratype)
    m1 = random_MERA(meratype, spaces, intspaces; random_disentangler=true)
    m2 = random_MERA(meratype, spaces, intspaces; random_disentangler=true)
    m = tensorwise_sum(m1, m2)
    V = outputspace(m, 1)
    eye = id(V)
    # Test that we've broken isometricity
    @test !(expect(eye, m; evalscale=1) ≈ 1.0)
    # Test that the new MERA is isometric, but that the old one hasn't changed.
    m_new = projectisometric(m)
    @test expect(eye, m_new; evalscale=1) ≈ 1.0
    @test !(expect(eye, m; evalscale=1) ≈ 1.0)
    # Test that projectisometric! works too. Note that there's no guarantee that it modifies
    # the old m to be isometric. The ! simply gives it permission to mess with old objects.
    m = projectisometric!(m)
    @test expect(eye, m; evalscale=1) ≈ 1.0
end

"""
Test optimization on a Hamiltonian that is just the particle number operator We know what it
should converge to, and it should converge fast.
"""
function test_optimization(meratype, spacetype, method, precondition=false)
    layers = 3
    # eps is the threshold for how close we need to be to the actual ground state energy
    # to pass the test.
    eps = 1e-2
    dlow = 2
    dhigh = 3
    pars = Dict(:method => method,
                :gradient_delta => 1e-4,
                :maxiter => 500,
                :isometries_only_iters => 30,
                :precondition => precondition,
                :verbosity => 0,
                :scaleinvariant_krylovoptions => Dict(
                                                      :tol => 1e-8,
                                                      :krylovdim => 4,
                                                      :verbosity => 0,
                                                      :maxiter => 20,
                                                     ),
               )

    op = particle_number_operator(spacetype)
    width = causal_cone_width(meratype)
    V = domain(op)
    ham = -reduce(⊗, repeat([op], width))
    spaces = random_layerspaces(spacetype, meratype, layers-1, dlow, dhigh)
    spaces = (V, spaces...)
    intspaces = random_internalspaces(spaces, meratype)
    m = random_MERA(meratype, spaces, intspaces)
    m = minimize_expectation!(m, ham, pars)
    expectation = expect(ham, m)
    @test abs(expectation + 1.0) < eps
end

"""
Test gradients and retraction.
"""
function test_gradient_and_retraction(meratype, spacetype, alg, metric)
    layers = 4
    spaces = random_layerspaces(spacetype, meratype, layers)
    intspaces = random_internalspaces(spaces, meratype)
    morig = random_MERA(meratype, spaces, intspaces; random_disentangler=true)

    width = causal_cone_width(meratype)
    V = outputspace(morig, 1)
    hamspace = reduce(⊗, repeat([V], width))
    ham = TensorMap(randn, ComplexF64, hamspace ← hamspace)
    ham = ham + ham'
    eye = id(V)

    pars = Dict(:metric => metric, :precondition => false)
    pars = merge(MERA.default_pars, pars)

    fg(x) = (expect(ham, x), gradient(ham, x, pars; metric=metric))
    scale!(vec, beta) = tensorwise_scale(vec, beta)
    add!(vec1, vec2, beta) = tensorwise_sum(vec1, scale!(vec2, beta))

    # Retract along the trajectory generated by the gradient of ham, by alpha and by
    # alpha+delta.
    alpha = 2.7
    delta = 0.0001
    tanorig = fg(morig)[2]
    m1, tan1 = retract(morig, tanorig, alpha; alg=alg)
    m2, tan2 = retract(morig, tanorig, alpha+delta; alg=alg)

    # Check that the points we retracted to are normalized MERAs.
    @test expect(eye, m1) ≈ 1.0
    @test expect(eye, m2) ≈ 1.0

    # TODO If we implement getting a tangent vector that goes from m1 to m2, we should put
    # in a test that checks that that vector is roughly tan1*delta.

    # Get the energies and gradients at both points. Check that the energy difference
    # between them is the inner product of the gradient and the tangent.
    f1, g1 = fg(m1)
    f2, g2 = fg(m2)
    reco1 = (f2 - f1)/delta
    reco2 = (inner(m1, tan1, g1; metric=metric) + inner(m2, tan2, g2; metric=metric)) / 2.0
    @test isapprox(reco1, reco2; rtol=10*delta)
end

"""
Test vector transport.
"""
function test_transport(meratype, spacetype, alg, metric)
    layers = 4
    spaces = random_layerspaces(spacetype, meratype, layers)
    intspaces = random_internalspaces(spaces, meratype)
    m = random_MERA(meratype, spaces, intspaces; random_disentangler=true)

    width = causal_cone_width(meratype)
    V = outputspace(m, 1)
    hamspace = reduce(⊗, repeat([V], width))
    # Make three different random Hamiltonians.
    hams = [TensorMap(randn, ComplexF64, hamspace ← hamspace) for i in 1:3]
    hams = [ham + ham' for ham in hams]

    pars = Dict(:metric => metric, :precondition => false)
    pars = merge(MERA.default_pars, pars)

    g1, g2, g3 = [gradient(ham, m, pars; metric=metric) for ham in hams]
    angle_pre = inner(m, g2, g3; metric=metric)
    # Transport g2 and g3 along the retraction by g1, by distance alpha.
    alpha = 2.7
    mt = retract(m, g1, alpha; alg=alg)[1]
    g2t = transport!(g2, m, g1, alpha, mt; alg=alg)
    g3t = transport!(g3, m, g1, alpha, mt; alg=alg)
    angle_post = inner(mt, g2t, g3t; metric=metric)

    # Check that the inner product has been preserved by the transport.
    @test angle_pre ≈ angle_post
end

function test_with_all_types(testfunc, meratypes, spacetypes, args...)
    for meratype in meratypes
        for spacetype in spacetypes
            testfunc(meratype, spacetype, args...)
        end
    end
end

Random.seed!(1)  # For reproducing the same tests again and again.
meratypes = (ModifiedBinaryMERA, BinaryMERA, TernaryMERA)
spacetypes = (ComplexSpace, Z2Space)

# Run the tests on different MERAs and vector spaces.
# Basics
@testset "Ascend and descend" begin
    test_with_all_types(test_ascend_and_descend, meratypes, spacetypes)
end
@testset "Expectation of identity" begin
    test_with_all_types(test_expectation_of_identity, meratypes, spacetypes)
end
@testset "Expectation at various evalscales" begin
    test_with_all_types(test_expectation_evalscale, meratypes, spacetypes)
end
@testset "Pseudoserialization" begin
    test_with_all_types(test_pseudoserialization, meratypes, spacetypes)
end
@testset "Expanding bond dimension" begin
    test_with_all_types(test_expand_bonddim, meratypes, spacetypes)
end
@testset "Releasing layers" begin
    test_with_all_types(test_release_layer, meratypes, spacetypes)
end
@testset "Removing symmetry" begin
    test_with_all_types(test_remove_symmetry, meratypes, spacetypes)
end
@testset "Reset storage" begin
    test_with_all_types(test_reset_storage, meratypes, spacetypes)
end
@testset "Reset projectisometric" begin
    test_with_all_types(test_projectisometric, meratypes, spacetypes)
end

# Manifold operations
@testset "Gradient and retraction, Cayley transform, canonical metric" begin
    test_with_all_types(test_gradient_and_retraction, meratypes, spacetypes,
                        :cayley, :canonical)
end
@testset "Gradient and retraction, Cayley transform, Euclidean metric" begin
    test_with_all_types(test_gradient_and_retraction, meratypes, spacetypes,
                        :cayley, :euclidean)
end
@testset "Gradient and retraction, exponential, canonical metric" begin
    test_with_all_types(test_gradient_and_retraction, meratypes, spacetypes,
                        :exp, :canonical)
end
@testset "Gradient and retraction, exponential, Euclidean metric" begin
    test_with_all_types(test_gradient_and_retraction, meratypes, spacetypes,
                        :exp, :euclidean)
end
@testset "Transport, cayley transform, canonical metric" begin
    test_with_all_types(test_transport, meratypes, spacetypes, :cayley, :canonical)
end
@testset "Transport, cayley transform, Euclidean metric" begin
    test_with_all_types(test_transport, meratypes, spacetypes, :cayley, :euclidean)
end
@testset "Transport, exponential, canonical metric" begin
    test_with_all_types(test_transport, meratypes, spacetypes, :exp, :canonical)
end
@testset "Transport, exponential, Euclidean metric" begin
    test_with_all_types(test_transport, meratypes, spacetypes, :exp, :euclidean)
end

# Optimization
@testset "Optimization E-V" begin
    test_with_all_types((mt, st) -> test_optimization(mt, st, :ev), meratypes, spacetypes)
end
@testset "Optimization LBFGS" begin
    test_with_all_types((mt, st) -> test_optimization(mt, st, :lbfgs, false),
                        meratypes, spacetypes)
end
@testset "Optimization LBFGS with preconditioning" begin
    test_with_all_types((mt, st) -> test_optimization(mt, st, :lbfgs, true),
                        meratypes, spacetypes)
end
