using Test
using TensorKit
using TensorKitManifolds
using LinearAlgebra
using MERA
using Logging
using Random

function particle_number_operator(::Type{Vect[FibonacciAnyon]})
    V = Vect[FibonacciAnyon](FibonacciAnyon(:I) => 1, FibonacciAnyon(:τ) => 1)
    z = TensorMap(zeros, Float64, V ← V)
    z.data[FibonacciAnyon(:I)] .= 0.0
    z.data[FibonacciAnyon(:τ)] .= 1.0
    return z
end

function particle_number_operator(::Type{Vect[IsingAnyon]})
    V = Vect[IsingAnyon](
        IsingAnyon(:I) => 1,
        IsingAnyon(:σ) => 1,
        IsingAnyon(:ψ) => 1
    )
    z = TensorMap(zeros, Float64, V ← V)
    z.data[IsingAnyon(:I)] .= 0.0
    z.data[IsingAnyon(:σ)] .= 1.0
    z.data[IsingAnyon(:ψ)] .= 0.0
    return z
end

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

function random_space(::Type{Vect[FibonacciAnyon]}, dlow=2, dhigh=6)
    dtotal = rand(dlow:dhigh)
    d0 = rand(1:dtotal-1)
    d1 = dtotal - d0
    V = Vect[FibonacciAnyon](FibonacciAnyon(:I) => d0, FibonacciAnyon(:τ) => d1)
    return V
end

function random_space(::Type{Vect[IsingAnyon]}, dlow=3, dhigh=6)
    @assert dlow > 2
    dtotal = rand(dlow:dhigh)
    d0 = rand(1:dtotal-2)
    d1 = rand(1:(dtotal-d0-1))
    d2 = dtotal - d0 - d1
    V = Vect[IsingAnyon](
        IsingAnyon(:I) => d0,
        IsingAnyon(:σ) => d1,
        IsingAnyon(:ψ) => d2
    )
    return V
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
Generate vector spaces for a MERA of `M`, with `n` layers, and `S` vector space type.
"""
function random_layerspaces(::Type{S}, ::Type{M}, n, dlow=3, dhigh=6) where {S, M}
    width = causal_cone_width(M)
    V = random_space(S, dlow, dhigh)
    spaces = S[V]
    for i = 2:n
        V_prev = V
        V_prev_fusion = fuse(fill(V_prev, width)...)
        V = random_space(S, dlow, dhigh)
        # If a layer from V_prev to V would be of such a dimension that
        # it couldn't be isometric, try generating another V until
        # success.  Since V = V_prev is a valid choice, this will
        # eventually terminate.
        while infimum(V_prev_fusion, V) != V
            V = random_space(S, dlow, dhigh)
        end
        push!(spaces, V)
    end
    return tuple(spaces...)::NTuple{n,S}
end

"""
Generate vector spaces for the layer-internal indices of a MERA of `M`, that has
`extspaces` as its interlayer spaces.
"""
function random_internalspaces(extspaces, M)
    width = (M == TernaryMERA ? 3 :
    M in (BinaryMERA, ModifiedBinaryMERA) ? 2 : nothing)
    intspaces = ntuple(length(extspaces)) do i
        # The fuse just removes nested product state structure.
        Vext = fuse(extspaces[i])
        Vnext = i == length(extspaces) ? Vext : extspaces[i+1]
        sects = Dict()
        for s in sectors(Vext)
            d = dim(Vext, s)
            rand(Bool) && d > 1 && (d = d - 1)
            sects[s] = d
        end
        Vint = spacetype(Vext)(sects...)
        # Check that we didn't accidentally make the isometric part
        # non-isometric. If we did, just make Vint = Vext.
        Vint_fusion = fuse(fill(Vint, width)...)
        infimum(Vint_fusion, Vnext) != Vnext && (Vint = Vext)
        Vint
    end
    return intspaces
end

"""
Test ModifiedBinaryOp.
"""
function test_modifiedbinaryop(::Type{S}) where {S}
    V = random_space(S, 3)  # Use a minimum dimension of 3.
    # Unary operations
    a = 0.1
    unaryops = [
        copy,
        adjoint,
        one,
        (x) -> MERA.expand_support(x, 2),
        (x) -> a * x,
        (x) -> x * a,
        (x) -> x / a,
        (x) -> fill!(x, a),
        (x) -> rmul!(x, a),
        (x) -> lmul!(a, x),
    ]
    if isreal(sectortype(S))
        # These don't make sense complex sector types.
        push!(unaryops, imag)
        push!(unaryops, real)
    end
    for f in unaryops
        mid = TensorMap(randn, ComplexF64, V, V)
        gap = TensorMap(randn, ComplexF64, V, V)
        m = ModifiedBinaryOp(mid, gap)
        res_t = ModifiedBinaryOp(f(mid), f(gap))
        res_m = f(m)
        @test res_t == res_m
    end

    # Binary operations
    a = 0.1
    b = 2.7
    binaryops = (
        *,
        +,
        -,
        /,
        (x, y) -> axpby!(a, x, b, y),
        (x, y) -> axpy!(a, x, y),
        (x, y) -> mul!(x, a, y),
        (x, y) -> mul!(x, y, b),
    )
    for f in binaryops
        t1 = TensorMap(randn, ComplexF64, V, V)
        t2 = TensorMap(randn, ComplexF64, V, V)
        m1 = ModifiedBinaryOp(t1)
        m2 = ModifiedBinaryOp(t2)
        res_tt = ModifiedBinaryOp(f(t1, t2))
        res_mm = f(m1, m2)
        res_mt = f(m1, t2)
        res_tm = f(t1, m2)
        @test res_tt == res_mm
        @test res_tt == res_mt
        @test res_tt == res_tm

        mid1 = TensorMap(randn, ComplexF64, V, V)
        gap1 = TensorMap(randn, ComplexF64, V, V)
        mid2 = TensorMap(randn, ComplexF64, V, V)
        gap2 = TensorMap(randn, ComplexF64, V, V)
        m1 = ModifiedBinaryOp(mid1, gap1)
        m2 = ModifiedBinaryOp(mid2, gap2)
        res_t = ModifiedBinaryOp(f(mid1, mid2), f(gap1, gap2))
        res_m = f(m1, m2)
        @test res_t == res_m
    end

    # Ones that didn't fit the above patterns.
    mid1 = TensorMap(randn, ComplexF64, V, V)
    gap1 = TensorMap(randn, ComplexF64, V, V)
    mid2 = TensorMap(randn, ComplexF64, V, V)
    gap2 = TensorMap(randn, ComplexF64, V, V)
    m1 = ModifiedBinaryOp(mid1, gap1)
    m2 = ModifiedBinaryOp(mid2, gap2)
    res_t = (dot(mid1, mid2) + dot(gap1, gap2)) / 2
    res_m = dot(m1, m2)
    @test res_t == res_m
    res_t = (tr(mid1) + tr(gap1)) / 2
    res_m = tr(m1)
    @test res_t == res_m

    res_t = ModifiedBinaryOp(copyto!(mid1, mid2), copyto!(gap1, gap2))
    res_m = copyto!(m1, m2)
    @test res_t == res_m
end

"""
Test type stability, and type stability only, of various methods.
"""
function test_type_stability(::Type{M}, ::Type{S}) where {M, S}
    layers = 4
    width = @inferred causal_cone_width(M)
    L = layertype(M)
    spaces = random_layerspaces(S, M, layers)
    intspaces = random_internalspaces(spaces, M)
    m = random_MERA(M, ComplexF64, spaces, intspaces; random_disentangler=true)

    V1, V2 = spaces[1:2]
    Vend = spaces[end]
    randomop1 = TensorMap(randn, ComplexF64, V1 ← V1)
    randomop1 = randomop1 + randomop1'
    randomop1 = convert(
        MERA.operatortype(m), MERA.expand_support(randomop1, causal_cone_width(M))
    )
    randomop2 = TensorMap(randn, ComplexF64, V1 ← V1)
    randomop2 = randomop2 + randomop2'
    randomop2 = convert(
        MERA.operatortype(m), MERA.expand_support(randomop2, causal_cone_width(M))
    )
    randomrho1 = TensorMap(randn, ComplexF64, Vend ← Vend)
    randomrho1 = randomrho1 + randomrho1'
    randomrho1 = convert(
        MERA.operatortype(m), MERA.expand_support(randomrho1, causal_cone_width(M))
    )
    randomrho2 = TensorMap(randn, ComplexF64, V2 ← V2)
    randomrho2 = randomrho2 + randomrho2'
    randomrho2 = convert(
        MERA.operatortype(m), MERA.expand_support(randomrho2, causal_cone_width(M))
    )

    nt = @inferred num_translayers(m)
    l1 = @inferred randomlayer(
        L, ComplexF64, V2, V1, intspaces[1]; random_disentangler=false
    )

    @inferred replace_layer(m, l1, 1)
    @inferred release_transitionlayer(m)
    @inferred projectisometric(m)
    @inferred outputspace(m, Inf)
    @inferred inputspace(m, Inf)
    @inferred MERA.environment(randomop1, l1, randomrho2)

    @inferred MERA.ascend(randomop1, m)
    @inferred ascended_operator(randomop1, m, 1)
    @inferred MERA.descend(randomrho1, m)
    @inferred MERA.thermal_densitymatrix(m, Inf)
    @inferred MERA.fixedpoint_densitymatrix(m)
    @inferred MERA.scale_invariant_operator_sum(randomop1, m, (;))
    @inferred densitymatrix(m, 1)

    # TODO Finish this part, and add @inferred checks for more functions, optimally all
    # functions that can reasonably be expected to be type stable.
    pars = (metric = :euclidean, precondition = false)
    pars = merge(MERA.default_pars, pars)
    mtan1 = gradient(randomop1, m, pars)
    mtan2 = gradient(randomop2, m, pars)
    ltan1 = getlayer(mtan1, 1)
    ltan2 = getlayer(mtan2, 1)
    @inferred (x -> tuple(x...))(ltan1)
    @inferred MERA.tensorwise_scale(ltan1, 0.1)
    @inferred MERA.tensorwise_sum(ltan1, ltan2)
    @inferred inner(l1, ltan1, ltan2)
    @inferred MERA.tensorwise_scale(mtan1, 0.1)
    @inferred MERA.tensorwise_sum(mtan1, mtan2)
    @inferred inner(m, mtan1, mtan2)
end

"""
For each layer, generate a random operator above and below it (not necessarily Hermitian),
and confirm that ascending the lower or descending the upper one both lead to the same
expectation value (trace of product).
"""
function test_ascend_and_descend(::Type{M}, ::Type{S}) where {M, S}
    layers = 4
    spaces = random_layerspaces(S, M, layers)
    intspaces = random_internalspaces(spaces, M)
    m = random_MERA(M, ComplexF64, spaces, intspaces; random_disentangler=true)
    width = causal_cone_width(M)

    for i in 1:layers
        Vin = inputspace(m, i)
        Vout = outputspace(m, i)
        upper_space = ⊗(fill(Vin, width)...)
        lower_space = ⊗(fill(Vout, width)...)
        randomop1 = TensorMap(randn, ComplexF64, upper_space ← upper_space)
        randomop2 = TensorMap(randn, ComplexF64, lower_space ← lower_space)
        down1 = MERA.descend(randomop1, m, i, i+1)
        up2 = MERA.ascend(randomop2, m, i+1, i)
        e1 = dot(down1, randomop2)
        e2 = dot(randomop1, up2)
        @test e1 ≈ e2
    end
end

"""
Test that the expectation value of the identity is 1.0, regardless of which layer we
evaluate it at.
"""
function test_expectation_of_identity(::Type{M}, ::Type{S}) where {M, S}
    layers = 4
    spaces = random_layerspaces(S, M, layers)
    intspaces = random_internalspaces(spaces, M)
    m = random_MERA(M, ComplexF64, spaces, intspaces; random_disentangler=true)
    V = outputspace(m, 1)
    eye = id(V)
    for i in 1:(layers+1)
        @test expect(eye, m, (;), 1, i) ≈ 1.0
    end
end

"""
Test that the expectation value of a random Hermitian operator does not depend on the layer
that we evaluate it at.
"""
function test_expectation_evalscale(::Type{M}, ::Type{S}) where {M, S}
    layers = 4
    spaces = random_layerspaces(S, M, layers)
    intspaces = random_internalspaces(spaces, M)
    m = random_MERA(M, ComplexF64, spaces, intspaces; random_disentangler=true)
    V = outputspace(m, 1)
    randomop = TensorMap(randn, ComplexF64, V ← V)
    randomop = (randomop + randomop')/2
    expectation = expect(randomop, m)
    for i in 1:(layers+1)
        expectation_i = expect(randomop, m, (;), 1, i)
        @test expectation_i ≈ expectation
    end
end

"""
Test that pseudoserializing and depseudoserializing back does not change the expectation
value of a random Hermitian operator.
"""
function test_pseudoserialization(::Type{M}, ::Type{S}) where {M, S}
    layers = 4
    spaces = random_layerspaces(S, M, layers)
    intspaces = random_internalspaces(spaces, M)
    m = random_MERA(M, ComplexF64, spaces, intspaces; random_disentangler=true)
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
function test_expand_bonddim(::Type{M}, ::Type{S}) where {M, S}
    num_layers = 4
    spaces = random_layerspaces(S, M, num_layers, 4)
    intspaces = random_internalspaces(spaces, M)
    m = random_MERA(M, ComplexF64, spaces, intspaces; random_disentangler=true)
    V = outputspace(m, 1)
    randomop = TensorMap(randn, ComplexF64, V ← V)
    randomop = (randomop + randomop')/2
    expectation = expect(randomop, m)
    sf = scalefactor(M)

    # Expand the interlayer spaces, check that expectation value is preserved.
    for i in 1:(num_layers-1)
        V = inputspace(m, i)
        expandable_sectors = sectors(V)
        Vint = fuse(⊗(fill(internalspace(m, i), sf)...))
        expandable_sectors = [s for s in expandable_sectors if dim(V, s) < dim(Vint, s)]
        if i == num_layers-1
            Vint = fuse(⊗(fill(internalspace(m, i+1), sf)...))
            expandable_sectors = [s for s in expandable_sectors if dim(V, s) < dim(Vint, s)]
        end
        newdims = Dict(s => dim(V, s) + 1 for s in expandable_sectors)
        m = expand_bonddim(m, i, newdims)
    end
    new_expectation = expect(randomop, m)
    @test new_expectation ≈ expectation

    # Expand the intralayer spaces, check that expectation value is preserved.
    for i in 1:(num_layers)
        V = internalspace(m, i)
        Vout = outputspace(m, i)
        newdims = Dict(
            s => dim(V, s) < dim(Vout, s) ? dim(V, s) + 1 : dim(V, s)
            for s in sectors(V)
        )
        m = expand_internal_bonddim(m, i, newdims)
    end
    new_expectation = expect(randomop, m)
    @test new_expectation ≈ expectation
end

"""
Confirm that releasing a does not change the expectation value of a random Hermitian
operator.
"""
function test_release_layer(::Type{M}, ::Type{S}) where {M, S}
    layers = 4
    spaces = random_layerspaces(S, M, layers)
    intspaces = random_internalspaces(spaces, M)
    m = random_MERA(M, ComplexF64, spaces, intspaces; random_disentangler=true)
    V = outputspace(m, 1)
    randomop = TensorMap(randn, ComplexF64, V ← V)
    randomop = (randomop + randomop')/2
    expectation = expect(randomop, m)
    m = release_transitionlayer(m)
    new_expectation = expect(randomop, m)
    @test new_expectation ≈ expectation
end

"""
Create a random MERA and operator, evaluate the expectation values, strip both of their
symmetry structure, and confirm that the expectation value hasn't changed.
"""
function test_remove_symmetry(::Type{M}, ::Type{S}) where {M, S}
    layers = 4
    spaces = random_layerspaces(S, M, layers)
    intspaces = random_internalspaces(spaces, M)
    m = random_MERA(M, ComplexF64, spaces, intspaces; random_disentangler=true)
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
function test_reset_storage(::Type{M}, ::Type{S}) where {M, S}
    layers = 4
    spaces = random_layerspaces(S, M, layers)
    intspaces = random_internalspaces(spaces, M)
    m = random_MERA(M, ComplexF64, spaces, intspaces; random_disentangler=true)
    V = outputspace(m, 1)
    randomop = TensorMap(randn, ComplexF64, V ← V)
    randomop = (randomop + randomop')/2
    expectation_orig = expect(randomop, m)

    m = replace_layer(m, getlayer(m, 2), 2)
    expectation_reset = expect(randomop, m)
    @test expectation_orig ≈ expectation_reset

    m = reset_storage(m)
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
function test_projectisometric(::Type{M}, ::Type{S}) where {M, S}
    layers = 4
    spaces = random_layerspaces(S, M, layers)
    intspaces = random_internalspaces(spaces, M)
    m1 = random_MERA(M, ComplexF64, spaces, intspaces; random_disentangler=true)
    m2 = random_MERA(M, ComplexF64, spaces, intspaces; random_disentangler=true)
    m = tensorwise_sum(m1, m2)
    V = outputspace(m, 1)
    eye = id(V)
    # Test that we've broken isometricity
    @test !(expect(eye, m) ≈ 1.0)
    # Test that the new MERA is isometric, but that the old one hasn't changed.
    m_new = projectisometric(m)
    @test expect(eye, m_new) ≈ 1.0
    @test !(expect(eye, m) ≈ 1.0)
    # Test that projectisometric! works too. Note that there's no guarantee that it modifies
    # the old m to be isometric. The ! simply gives it permission to mess with old objects.
    m = projectisometric!(m)
    @test expect(eye, m) ≈ 1.0
end

"""
Create a random MERA and compute its scaling dimensions. The numbers will be nonsense, but
at least we know the function runs.
"""
function test_scalingdimensions(::Type{M}, ::Type{S}) where {M, S}
    layers = 4
    spaces = random_layerspaces(S, M, layers)
    intspaces = random_internalspaces(spaces, M)
    m = random_MERA(M, ComplexF64, spaces, intspaces; random_disentangler=true)
    scalingdimensions(m)
end

"""
Test optimization on a Hamiltonian that is just the particle number operator We know what it
should converge to, and it should converge fast.
"""
function test_optimization(::Type{M}, ::Type{S}, method, precondition = false) where {M, S}
    layers = 3
    # eps is the threshold for how close we need to be to the actual ground state energy
    # to pass the test.
    eps = 1e-2
    dlow = 2
    dhigh = 3
    pars = (
        method = method,
        gradient_delta = 1e-6,
        maxiter = 1000,
        isometries_only_iters = 30,
        precondition = precondition,
        verbosity = 0,
        scaleinvariant_krylovoptions = (
            tol = 1e-8,
            verbosity = 0,
            maxiter = 20,
        ),
    )

    op = particle_number_operator(S)
    width = causal_cone_width(M)
    V = domain(op)
    ham = -1*⊗(fill(op, width)...)
    spaces = random_layerspaces(S, M, layers-1, dlow, dhigh)
    spaces = (V, spaces...)
    intspaces = random_internalspaces(spaces, M)
    m = random_MERA(M, ComplexF64, spaces, intspaces)
    m = minimize_expectation(m, ham, pars)
    expectation = expect(ham, m)
    @test abs(expectation + 1.0) < eps
    # TODO We used to check here that these entropies are small, but this doesn't hold for
    # anyons. Figure out what's a good check.
    entropies = densitymatrix_entropies(m)
end

"""
Test gradients and retraction.
"""
function test_gradient_and_retraction(::Type{M}, ::Type{S}, alg, metric) where {M, S}
    layers = 4
    spaces = random_layerspaces(S, M, layers)
    intspaces = random_internalspaces(spaces, M)
    morig = random_MERA(M, ComplexF64, spaces, intspaces; random_disentangler=true)

    width = causal_cone_width(M)
    V = outputspace(morig, 1)
    hamspace = ⊗(fill(V, width)...)
    ham = TensorMap(randn, ComplexF64, hamspace ← hamspace)
    ham = ham + ham'
    eye = id(V)

    pars = (metric = metric, precondition = false)
    pars = merge(MERA.default_pars, pars)

    fg(x) = (expect(ham, x), gradient(ham, x, pars))
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
function test_transport(::Type{M}, ::Type{S}, alg, metric) where {M, S}
    layers = 4
    spaces = random_layerspaces(S, M, layers)
    intspaces = random_internalspaces(spaces, M)
    m = random_MERA(M, ComplexF64, spaces, intspaces; random_disentangler=true)

    width = causal_cone_width(M)
    V = outputspace(m, 1)
    hamspace = ⊗(fill(V, width)...)
    # Make three different random Hamiltonians.
    hams = [TensorMap(randn, ComplexF64, hamspace ← hamspace) for i in 1:3]
    hams = [ham + ham' for ham in hams]

    pars = (metric = metric, precondition = false)
    pars = merge(MERA.default_pars, pars)

    g1, g2, g3 = [gradient(ham, m, pars) for ham in hams]
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
    for M in meratypes
        for S in spacetypes
            testfunc(M, S, args...)
        end
    end
end

Random.seed!(2)  # For reproducing the same tests again and again.
meratypes = (ModifiedBinaryMERA, BinaryMERA, TernaryMERA)
spacetypes = (ComplexSpace, Z2Space, Vect[FibonacciAnyon])

# Run the tests on different MERAs and vector spaces.
# Basics
@testset "ModifiedBinaryOp" begin
    for S in spacetypes
        test_modifiedbinaryop(S)
    end
end
@testset "Type stability" begin
    test_with_all_types(test_type_stability, meratypes, spacetypes)
end
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
    # This doesn't make sense for anyons.
    nonanonic_spacetypes = (
        ST for ST in spacetypes if BraidingStyle(sectortype(ST)) != Anyonic()
    )
    test_with_all_types(test_remove_symmetry, meratypes, nonanonic_spacetypes)
end
@testset "Reset storage" begin
    test_with_all_types(test_reset_storage, meratypes, spacetypes)
end
@testset "Projectisometric" begin
    test_with_all_types(test_projectisometric, meratypes, spacetypes)
end
@testset "Scaling dimensions" begin
    # TODO Make this work with anyons too.
    test_with_all_types(test_scalingdimensions, meratypes, (ComplexSpace, Z2Space))
end

# Manifold operations
@testset "Gradient and retraction, Cayley transform, canonical metric" begin
    test_with_all_types(
        test_gradient_and_retraction, meratypes, spacetypes, :cayley, :canonical
    )
end
@testset "Gradient and retraction, Cayley transform, Euclidean metric" begin
    test_with_all_types(
        test_gradient_and_retraction, meratypes, spacetypes, :cayley, :euclidean
    )
end
@testset "Gradient and retraction, exponential, canonical metric" begin
    test_with_all_types(
        test_gradient_and_retraction, meratypes, spacetypes, :exp, :canonical
    )
end
@testset "Gradient and retraction, exponential, Euclidean metric" begin
    test_with_all_types(
        test_gradient_and_retraction, meratypes, spacetypes, :exp, :euclidean
    )
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
    test_with_all_types((M, S) -> test_optimization(M, S, :ev), meratypes, spacetypes)
end
@testset "Optimization LBFGS" begin
    test_with_all_types(
        (M, S) -> test_optimization(M, S, :lbfgs, false), meratypes, spacetypes
    )
end
@testset "Optimization LBFGS with preconditioning" begin
    test_with_all_types(
        (M, S) -> test_optimization(M, S, :lbfgs, true), meratypes, spacetypes
    )
end
