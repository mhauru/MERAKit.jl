using Test
using TensorKit
using LinearAlgebra
using MERA
using Logging

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
Generate layers for a MERA of `meratype`, with `n` layers, and `T` vector space type.
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
        while min(V_prev_fusion, V) != V
            V = random_space(T, dlow, dhigh)
        end
        push!(spaces, V)
    end
    return spaces
end

"""
For each layer, generate a random operator above and below it (not necessarily Hermitian),
and confirm that ascending the lower or descending the upper one both lead to the same
expectation value (trace of product).
"""
function test_ascend_and_descend(meratype, spacetype)
    layers = 4
    spaces = random_layerspaces(spacetype, meratype, layers)
    m = random_MERA(meratype, spaces; random_disentangler=true)
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
        e1 = tr(down1 * randomop2)
        e2 = tr(up2 * randomop1)
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
    m = random_MERA(meratype, spaces; random_disentangler=true)
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
    m = random_MERA(meratype, spaces; random_disentangler=true)
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
    m = random_MERA(meratype, spaces; random_disentangler=true)
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
    spaces = random_layerspaces(spacetype, meratype, layers)
    m = random_MERA(meratype, spaces; random_disentangler=true)
    V = outputspace(m, 1)
    randomop = TensorMap(randn, ComplexF64, V ← V)
    randomop = (randomop + randomop')/2
    expectation = expect(randomop, m)
    for i in 1:(layers-1)
        V = inputspace(m, i)
        newdims = Dict(s => dim(V, s)+1 for s in sectors(V))
        m = expand_bonddim!(m, i, newdims)
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
    m = random_MERA(meratype, spaces; random_disentangler=true)
    V = outputspace(m, 1)
    randomop = TensorMap(randn, ComplexF64, V ← V)
    randomop = (randomop + randomop')/2
    expectation = expect(randomop, m)
    m = release_transitionlayer!(m)
    new_expectation = expect(randomop, m)
    @test new_expectation ≈ expectation
end

"""
Test optimization on a Hamiltonian that is just the particle number operator We know what it
should converge to, and it should converge fast.
"""
function test_optimization(meratype, spacetype)
    layers = 3
    # eps is the threshold for how close we need to be to the actual ground state energy
    # to pass the test.
    eps = 5e-2 
    dlow = 2
    dhigh = 3
    pars = Dict(:method => :trad,
                :densitymatrix_delta => 1e-5,
                :maxiter => 100,
                :miniter => 30,
                :havg_depth => 10,
                :layer_iters => 1,
                :disentangler_iters => 1,
                :isometry_iters => 1)

    op = particle_number_operator(spacetype)
    width = causal_cone_width(meratype)
    V = domain(op)
    ham = -reduce(⊗, repeat([op], width))
    spaces = random_layerspaces(spacetype, meratype, layers-1, dlow, dhigh)
    m = random_MERA(meratype, (V, spaces...))
    minimize_expectation!(m, ham, pars)
    expectation = expect(ham, m)
    @test abs(expectation + 1.0) < eps
end

"""
Test gradients and retraction.
"""
function test_stiefel_gradient_and_retraction(meratype, spacetype, retract)
    layers = 4
    spaces = random_layerspaces(spacetype, meratype, layers)
    morig = random_MERA(meratype, spaces; random_disentangler=true)

    width = causal_cone_width(meratype)
    V = outputspace(morig, 1)
    hamspace = reduce(⊗, repeat([V], width))
    ham = TensorMap(randn, ComplexF64, hamspace ← hamspace)
    ham = ham + ham'
    eye = id(V)

    pars = Dict(:havg_depth => 10)

    fg(x) = (expect(ham, x), stiefel_gradient(ham, x, pars))
    inner(m, x, y) = 2*real(stiefel_inner(m, x, y))
    scale!(vec, beta) = tensorwise_scale(vec, beta)
    add!(vec1, vec2, beta) = tensorwise_sum(vec1, scale!(vec2, beta))

    # Retract along the trajectory generated by the gradient of ham, by alpha and by
    # alpha+delta.
    alpha = 2.7
    delta = 0.0001
    tanorig = fg(morig)[2]
    m1, tan1 = retract(morig, tanorig, alpha)
    m2, tan2 = retract(morig, tanorig, alpha+delta)

    # Check that the gradient is a Stiefel tangent vector.
    @test istangent(morig, tanorig)
    # Check that the tangents really are tangents
    @test istangent(m1, tan1)
    @test istangent(m2, tan2)
    # Check that the points we retracted to are normalized MERAs.
    @test expect(eye, m1) ≈ 1.0
    @test expect(eye, m2) ≈ 1.0

    # Get the energies and gradients at both points.
    f1, g1 = fg(m1)
    f2, g2 = fg(m2)

    # We have three ways of computing the different between m1 and m2: Just taking the
    # different, multiplying the tangent at m1 by delta, and multiplying the tangent at m2
    # by delta. These should all give roughly the same answer, to order delta or so.
    # Confirm this by taking the overlaps between different options, and make sure that the
    # overlap is roughly the same as the norm.
    diff1 = scale!(tan1, delta)
    diff2 = scale!(tan2, delta)
    diffreco = add!(m2, m1, -1)
    overlap1 = inner(m1, diff1, diffreco)
    overlap2 = inner(m2, diff2, diffreco)
    norm1 = inner(m1, diff1, diff1)
    norm2 = inner(m2, diff2, diff2)
    @test isapprox(overlap1, norm1; rtol=10*delta)
    @test isapprox(overlap2, norm2; rtol=10*delta)

    reco1 = (f2 - f1)/delta
    reco2 = inner(m1, tan1, g1)
    reco3 = inner(m2, tan2, g2)
    @test isapprox(reco1, reco2; rtol=10*delta)
    @test isapprox(reco1, reco3; rtol=10*delta)
end

"""
Test the isometric Cayley parallel transport from
http://www.optimization-online.org/DB_FILE/2016/09/5617.pdf
"""
function test_cayley_transport(meratype, spacetype)
    layers = 4
    spaces = random_layerspaces(spacetype, meratype, layers)
    m = random_MERA(meratype, spaces; random_disentangler=true)

    width = causal_cone_width(meratype)
    V = outputspace(m, 1)
    hamspace = reduce(⊗, repeat([V], width))
    # Make three different random Hamiltonians.
    hams = [TensorMap(randn, ComplexF64, hamspace ← hamspace) for i in 1:3]
    hams = [ham + ham' for ham in hams]

    pars = Dict(:havg_depth => 10)

    retract = cayley_retract
    inner(m, x, y) = 2*real(stiefel_inner(m, x, y))

    g1, g2, g3 = [stiefel_gradient(ham, m, pars) for ham in hams]
    # Transport g2 and g3 along the retraction by g1, by distance alpha.
    alpha = 2.7
    g2t = cayley_transport(m, g1, g2, alpha)
    g3t = cayley_transport(m, g1, g3, alpha)

    # Check that the transported vectors are proper tangents.
    mt = retract(m, g1, alpha)[1]
    @test istangent(mt, g2t)
    @test istangent(mt, g3t)
    # Check that the inner product has been preserved by the transport.
    @test inner(m, g2, g3) ≈ inner(mt, g2t, g3t)
end

function test_with_all_types(testfunc, meratypes, spacetypes, args...)
    for meratype in meratypes
        for spacetype in spacetypes
            testfunc(meratype, spacetype, args...)
        end
    end
end

meratypes = (BinaryMERA, TernaryMERA)
spacetypes = (ComplexSpace, Z2Space)

# Run the tests on different MERAs and vector spaces.
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
@testset "Stiefel gradient and Cayley retraction" begin
    test_with_all_types(test_stiefel_gradient_and_retraction, meratypes, spacetypes,
                        cayley_retract)
end
@testset "Stiefel gradient and Stiefel geodesic retraction" begin
    test_with_all_types(test_stiefel_gradient_and_retraction, meratypes, spacetypes,
                        stiefel_geodesic)
end
@testset "Cayley parallel transport" begin
    test_with_all_types(test_cayley_transport, meratypes, spacetypes)
end
@testset "Optimization" begin
    test_with_all_types(test_optimization, meratypes, spacetypes)
end
