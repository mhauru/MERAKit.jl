module TernaryMERAInfModelTools

using LinearAlgebra
using TensorKit
using JLD2
using MAT
using KrylovKit
include("MERA.jl")
using .MERA

version = 1.0

export version
export load_mera, store_mera, load_mera_matlab, store_mera_matlab
export build_H_Ising, build_H_XXZ, build_magop
export normalize_energy
export getrhoee, getrhoees
export build_superop_onesite, get_scaldims, remove_symmetry
export get_optimized_mera, optimize_layerbylayer!

function block_H(H, block)
    while block > 1
        VL = space(H, 1)
        VR = space(H, 2)
        eyeL = TensorMap(I, Float64, VL ← VL)
        eyeR = TensorMap(I, Float64, VR ← VR)
        Hcross = eyeR ⊗ H ⊗ eyeL
        Hleft = H ⊗ eyeL ⊗ eyeR
        Hright = eyeL ⊗ eyeR ⊗ H
        H_new_unfused = Hcross + 0.5*(Hleft + Hright)
        fusionspace = domain(H)
        fusetop = TensorMap(I, fuse(fusionspace) ← fusionspace)
        fusebottom = TensorMap(I, fusionspace ← fuse(fusionspace))
        H = (fusetop ⊗ fusetop) * H_new_unfused * (fusebottom ⊗ fusebottom)
        block /= 2
    end
    if block != 1
        msg = "`block` needs to be a power of 2"
        throw(ArgumentError(msg))
    end
    return H
end

function normalize_H(H)
    # Subtract a constant, so that the spectrum is negative
    # TODO Switch to using an eigendecomposition?
    D_max = norm(H)
    bigeye = TensorMap(I, codomain(H) ← domain(H))
    H = H - bigeye*D_max
    return H, D_max
end

function build_H_XXZ(Delta=0.0; symmetry="none", block=1)
    if symmetry == "U1" || symmetry == "group"
        V = ℂ[U₁](-1=>1, 1=>1)
        Z = TensorMap(zeros, Float64, V ← V)
        Z.data[U₁(1)] .= 1.0
        Z.data[U₁(-1)] .= -1.0
        ZZ = Z ⊗ Z
        XXplusYY = TensorMap(zeros, Float64, V ⊗ V ← V ⊗ V)
        XXplusYY.data[U₁(0)] .= [0.0 2.0; 2.0 0.0]
        H = -(XXplusYY + Delta*ZZ)
    elseif symmetry == "none"
        V = ℂ^2
        Z = TensorMap(zeros, Float64, V ← V)
        Z.data .= [1.0 0.0; 0.0 -1.0]
        ZZ = Z ⊗ Z
        XXplusYY = TensorMap(zeros, Float64, V ⊗ V ← V ⊗ V)
        XXplusYY.data[2,3] = 2.0
        XXplusYY.data[3,2] = 2.0
        H = -(XXplusYY + Delta*ZZ)
    else
        error("Unknown symmetry $symmetry")
    end
    H = block_H(H, block)
    H, D_max = normalize_H(H)
    return H, D_max
end

# Ising model with transverse magnetic field h (critical h=1 by default)
function build_H_Ising(h=1.0; symmetry="none", block=1)
    if symmetry == "Z2"
        V = ℂ[ℤ₂](0=>1, 1=>1)
        # Pauli Z
        Z = TensorMap(zeros, Float64, V ← V)
        Z.data[ℤ₂(0)] .= 1.0
        Z.data[ℤ₂(1)] .= -1.0
        eye = TensorMap(I, Float64, V ← V)
        ZI = Z ⊗ eye
        IZ = eye ⊗ Z
        # Pauli XX
        XX = TensorMap(zeros, Float64, V ⊗ V ← V ⊗ V)
        XX.data[ℤ₂(0)] .= [0.0 1.0; 1.0 0.0]
        XX.data[ℤ₂(1)] .= [0.0 1.0; 1.0 0.0]
        H = -(XX + h/2 * (ZI+IZ))
    elseif symmetry == "anyons"
        V = RepresentationSpace{IsingAnyon}(:I => 0, :ψ => 0, :σ => 1)
        H = TensorMap(zeros, Float64, V ⊗ V ← V ⊗ V)
        H.data[IsingAnyon(:I)] .= 1.0
        H.data[IsingAnyon(:ψ)] .= -1.0
    elseif symmetry == "none"
        V = ℂ^2
        # Pauli matrices
        X = TensorMap(zeros, Float64, V ← V)
        Z = TensorMap(zeros, Float64, V ← V)
        eye = TensorMap(I, Float64, V ← V)
        X.data .= [0.0 1.0; 1.0 0.0]
        Z.data .= [1.0 0.0; 0.0 -1.0]
        XX = X ⊗ X
        ZI = Z ⊗ eye
        IZ = eye ⊗ Z
        H = -(XX + h/2 * (ZI+IZ))
    else
        error("Unknown symmetry $symmetry")
    end
    H = block_H(H, block)
    H, D_max = normalize_H(H)
    return H, D_max
end

function build_magop(;block=1)
    V = ℂ^2
    X = TensorMap(zeros, Float64, V ← V)
    eye = TensorMap(I, Float64, V ← V)
    X.data .= [0.0 1.0; 1.0 0.0]
    XI = X ⊗ eye
    IX = eye ⊗ X
    magop = (XI + IX)/2
    magop = block_H(mapop, block)
    return magop
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
    rhos = densitymatrices(m)
    ees = Vector{Float64}()
    for rho in rhos
        ee = getrhoee(rho)
        push!(ees, ee)
    end
    return ees
end

function normalize_energy(energy, dmax, block)
    energy = (energy + dmax)/block
    return energy
end

function build_superop_onesite(m)
    w = get_isometry(m, Inf)
    w_dg = w'
    @tensor(superop[-1,-2,-11,-12] := w[-1,1,-11,2] * w_dg[1,-12,2,-2])
    return superop
end

function remove_symmetry(m::T) where T <: GenericMERA
    uw_list_sym = m.layers
    uw_list_nosym = []
    for (u_sym, w_sym) in uw_list_sym
        V_in = ℂ^dim(space(u_sym, 3))
        V_out = ℂ^dim(space(w_sym, 1))
        u_nosym = TensorMap(zeros, eltype(u_sym), V_in ⊗ V_in ← V_in ⊗ V_in)
        w_nosym = TensorMap(zeros, eltype(u_sym), V_out ← V_in ⊗ V_in ⊗ V_in)
        u_nosym.data[:] = convert(Array, u_sym)
        w_nosym.data[:] = convert(Array, w_sym)
        push!(uw_list_nosym, (u_nosym, w_nosym))
    end
    m_nosym = T(uw_list_nosym)
    return m_nosym
end

function get_scaldims(m; mode=:onesite)
    # TODO we remove symmetries because we are lazy to code a symmetric
    # eigenvalue search.
    m = remove_symmetry(m)
    if mode == :onesite
        superop = build_superop_onesite(m)
        S, U = eig(superop, (1,2), (3,4))
    elseif mode == :twosite
        V = outputspace(m, Inf)
        typ = eltype(get_disentangler(m, Inf))
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
            scaldims[k] = sort(-log.(3, abs.(real(diag(v)))))
        end
    else
        scaldims = sort(-log.(3, abs.(real(S))))
    end
    return scaldims
end

function optimize_layerbylayer!(m, h, fixedlayers, normalization, opt_pars)
    while fixedlayers >= 0
        minimize_expectation!(m, h, opt_pars;
                              lowest_depth=fixedlayers+1,
                              normalization=normalization)
        fixedlayers -= 1
    end
end

function store_mera(path, m)
    # JLD2 still sucks on the workstation. Sigh.
    #@save path m
    deser = pseudoserialize(m)
    @save path deser
end

function load_mera(path)
    # JLD2 still sucks on the workstation. Sigh.
    #@load path m
    @load path deser
    m = depseudoserialize(deser...)
    return m
end

function store_mera_matlab(path, m)
    d = Dict{String, Array}()
    for i in 1:(num_translayers(m)+1)
        u, w = map(x -> convert(Array, x), get_layer(m, i))
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

function get_sectors_to_expand(V)
    result = Set(sectors(V))
    if typeof(V) == U₁Space
        # The `collect` makes a copy, so that we don't iterate over the ones
        # just added.
        for s in collect(result)
            # We make jumps by twos, for because of the Hamiltonian the odd
            # sectors are useless.
            splus = U₁(s.charge+2)
            sminus = U₁(s.charge-2)
            push!(result, splus)
            push!(result, sminus)
        end
    end
    return result
end

function get_optimized_mera(datafolder, model, pars; loadfromdisk=true)
    chi = pars[:chi]
    layers = pars[:layers]
    symmetry = pars[:symmetry]
    block = pars[:block]
    threads = pars[:threads]
    BLAS.set_num_threads(threads)
    global version

    mkpath(datafolder)
    filename = "MERA_$(model)_$(chi)_$(block)_$(symmetry)_$(layers)_$(version)"
    path = "$datafolder/$filename.jlm"
    matlab_folder = "./matlabdata"
    mkpath(matlab_folder)
    matlab_path = "$(matlab_folder)/$(filename).mat"

    if loadfromdisk && isfile(path)
        println("Found $filename on disk, loading it.")
        m = load_mera(path)
        return m
    else
        println("Did not find $filename on disk, generating it.")
        if model == "XXZ"
            h, dmax = build_H_XXZ(pars[:Delta]; symmetry=symmetry, block=block)
        elseif model == "Ising"
            h, dmax = build_H_Ising(pars[:h]; symmetry=symmetry, block=block)
        else
            msg = "Unknown model: $(model)"
            throw(ArgumentError(msg))
        end
        normalization(x) = normalize_energy(x, dmax, block)

        chitoosmall = ((symmetry == "U1" && chi < 3)
                       || (symmetry == "Z2" && chi < 2)
                       || (chi < 1)
                      )
        chiminimal = ((symmetry == "U1" && chi == 3)
                      || (symmetry == "Z2" && chi == 2)
                      || (chi == 1)
                     )
        if chitoosmall
            msg = "chi = $chi is too small for the symmetry group $symmetry."
            throw(ArgumentError(msg))
        elseif chiminimal
            V_phys = space(h, 1)
            if symmetry == "none"
                V_virt = ℂ^chi
            elseif symmetry == "U1" && chi == 3
                V_virt = ℂ[U₁](-2=>1, 0=>1, 2=>1)
            elseif symmetry == "Z2" && chi == 2
                V_virt = ℂ[ℤ₂](0=>1, 1=>1)
            else
                error("Unknown symmetry $symmetry")
            end

            Vs = tuple(V_phys, repeat([V_virt], layers-1)...)
            m = random_MERA(TernaryMERA, Vs)

            optimize_layerbylayer!(m, h, 0, normalization,
                                   pars[:final_opt_pars])
            store_mera(path, m)
            return m
        else
            prevpars = deepcopy(pars)
            prevpars[:chi] -= 1
            m = get_optimized_mera(datafolder, model, prevpars; loadfromdisk=loadfromdisk)

            for i in 1:num_translayers(m)
                V = outputspace(m, i)
                d = dim(V)
                if d < chi  # This should always be true.
                    expanded_meras = Dict()
                    for s in get_sectors_to_expand(V)
                        # Expand the bond dimension of sector s and try
                        # optimizing a bit to see how useful this expansion is.
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
            store_mera_matlab(matlab_path, m)
            return m
        end
    end
end

end  # module
