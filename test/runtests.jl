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
    pars = Dict(:densitymatrix_delta => 1e-5,
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

# Run the tests on different MERAs and vector spaces.
for meratype in (BinaryMERA, TernaryMERA)
    for spacetype in (ComplexSpace, Z2Space)
        @info("Testing $(meratype) with $(spacetype).")
        test_ascend_and_descend(meratype, spacetype)
        test_expectation_of_identity(meratype, spacetype)
        test_expectation_evalscale(meratype, spacetype)
        test_optimization(meratype, spacetype)
    end
end
