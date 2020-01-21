module DemoTools

using LinearAlgebra
using TensorKit
using JLD2
using MAT
using KrylovKit
using MERA

version = 1.0

export version
export load_mera, store_mera, store_mera_matlab
export build_H_Ising, build_H_XXZ, build_magop
export normalize_energy
export build_superop_onesite, remove_symmetry
export get_optimized_mera, optimize_layerbylayer!
export remove_symmetry

# # # Functions for creating Hamiltonians.

"""
Take a two-site operator `op` that defines the local term of a global operator, and block
sites together to form a new two-site operator for which each site corresponds to
`num_sites` sites of the original operator. `num_sites` should be a power of 2.
"""
function block_op(op::SquareTensorMap{2}, num_sites)
    while num_sites > 1
        VL = space(op, 1)
        VR = space(op, 2)
        eyeL = TensorMap(I, Float64, VL ← VL)
        eyeR = TensorMap(I, Float64, VR ← VR)
        opcross = eyeR ⊗ op ⊗ eyeL
        opleft = op ⊗ eyeL ⊗ eyeR
        opright = eyeL ⊗ eyeR ⊗ op
        op_new_unfused = opcross + 0.5*(opleft + opright)
        fusionspace = domain(op)
        fusetop = TensorMap(I, fuse(fusionspace) ← fusionspace)
        fusebottom = TensorMap(I, fusionspace ← fuse(fusionspace))
        op = (fusetop ⊗ fusetop) * op_new_unfused * (fusebottom ⊗ fusebottom)
        num_sites /= 2
    end
    if num_sites != 1
        msg = "`num_sites` needs to be a power of 2"
        throw(ArgumentError(msg))
    end
    return op
end

"""
Take a one-site operator `op` that defines the local term of a global operator, and block
sites together to form a new one-site operator for which each site corresponds to
`num_sites` sites of the original operator. `num_sites` should be a power of 2.
"""
function block_op(op::SquareTensorMap{1}, num_sites)
    while num_sites > 1
        V = space(op, 1)
        eye = TensorMap(I, Float64, V ← V)
        opleft = op ⊗ eye
        opright = eye ⊗ op
        op_new_unfused = opleft + opright
        fusionspace = domain(op_new_unfused)
        fusetop = TensorMap(I, fuse(fusionspace) ← fusionspace)
        fusebottom = TensorMap(I, fusionspace ← fuse(fusionspace))
        op = fusetop * op_new_unfused * fusebottom
        num_sites /= 2
    end
    if num_sites != 1
        msg = "`num_sites` needs to be a power of 2"
        throw(ArgumentError(msg))
    end
    return op
end


"""
Normalize an operator by subtracting a constant so that it's spectrum is negative
semidefinite. Return the normalized operator and the constant that was subtracted.
"""
function normalize_H(H)
    # TODO Switch to using an eigendecomposition?
    D_max = norm(H)
    bigeye = TensorMap(I, codomain(H) ← domain(H))
    H = H - bigeye*D_max
    return H, D_max
end

"""
Return the local Hamiltonian term for the XXZ model: -XX - YY - Delta*ZZ.
`symmetry` should be "none" or "group", and determmines whether the Hamiltonian should be an
explicitly U(1) symmetric TensorMap or a dense one. `block_size` determines how many sites
to block together, and should be a power of 2. The Hamiltonian is normalized with an
additive constant to be negative semidefinite, and the constant of normalization is also
returned.
"""
function build_H_XXZ(Delta=0.0; symmetry="none", block_size=1)
    if symmetry == "U1" || symmetry == "group"
        V = ℂ[U₁](-1=>1, 1=>1)
        Z = TensorMap(zeros, Float64, V ← V)
        Z.data[U₁(1)] .= 1.0
        Z.data[U₁(-1)] .= -1.0
        ZZ = Z ⊗ Z
        XXplusYY = TensorMap(zeros, Float64, V ⊗ V ← V ⊗ V)
        XXplusYY.data[U₁(0)] .= [0.0 2.0; 2.0 0.0]
    elseif symmetry == "none"
        V = ℂ^2
        Z = TensorMap(zeros, Float64, V ← V)
        Z.data .= [1.0 0.0; 0.0 -1.0]
        ZZ = Z ⊗ Z
        XXplusYY = TensorMap(zeros, Float64, V ⊗ V ← V ⊗ V)
        XXplusYY.data[2,3] = 2.0
        XXplusYY.data[3,2] = 2.0
    else
        error("Unknown symmetry $symmetry")
    end
    H = -(XXplusYY + Delta*ZZ)
    H = block_op(H, block_size)
    H, D_max = normalize_H(H)
    return H, D_max
end

"""
Return the local Hamiltonian term for the Ising model: -XX - h*Z
`symmetry` should be "none", "group", or "anyon" and determmines whether the Hamiltonian
should be an explicitly Z2 symmetric or anyonic TensorMap, or a dense one. `block_size`
determines how many sites to block together, and should be a power of 2. The Hamiltonian is
normalized with an additive constant to be negative semidefinite, and the constant of
normalization is also returned.
"""
function build_H_Ising(h=1.0; symmetry="none", block_size=1)
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
    H = block_op(H, block_size)
    H, D_max = normalize_H(H)
    return H, D_max
end

"""
Return the magnetization operator for the Ising model, possibly blocked over `block_size`
sites.
"""
function build_magop(;block_size=1)
    V = ℂ^2
    X = TensorMap(zeros, Float64, V ← V)
    eye = TensorMap(I, Float64, V ← V)
    X.data .= [0.0 1.0; 1.0 0.0]
    magop = block_op(X, block_size)
    return magop
end

"""
Given the normalization and block_size constants used in creating a Hamiltonian, and the
expectation value of the normalized and blocked Hamiltonian, return the actual energy.
"""
normalize_energy(energy, D_max, block_size) = (energy + D_max)/block_size

"""
Given a MERA which may possibly be built of symmetry preserving TensorMaps, and return
another MERA that has the symmetry structure stripped from it, and all tensors are dense.
"""
remove_symmetry(m::T) where T <: GenericMERA = T(map(remove_symmetry, m.layers))
"""Strip a TernaryLayer of its internal symmetries."""
remove_symmetry(layer::TernaryLayer) = TernaryLayer(map(remove_symmetry, layer)...)
"""Strip a BinaryLayer of its internal symmetries."""
remove_symmetry(layer::BinaryLayer) = BinaryLayer(map(remove_symmetry, layer)...)
"""Strip a real ElementarySpace of its symmetry structure."""
remove_symmetry(V::ElementarySpace{ℝ}) = CartesianSpace(dim(V))
"""Strip a complex ElementarySpace of its symmetry structure."""
remove_symmetry(V::ElementarySpace{ℂ}) = ComplexSpace(dim(V), isdual(V))

""" Strip a TensorMap of its internal symmetries."""
function remove_symmetry(t::TensorMap)
    domain_nosym = reduce(⊗, map(remove_symmetry, domain(t)))
    codomain_nosym = reduce(⊗, map(remove_symmetry, codomain(t)))
    t_nosym = TensorMap(zeros, eltype(t), codomain_nosym ← domain_nosym)
    t_nosym.data[:] = convert(Array, t)
    return t_nosym
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
        d["u$i"] = u
        d["w$i"] = w
    end
    matwrite(path, d)
end

function get_sectors_to_expand(V)
    result = Set(sectors(V))
    if typeof(V) == U₁Space
        # The `collect` makes a copy, so that we don't iterate over the ones just added.
        for s in collect(result)
            # We make jumps by twos, for because of the Hamiltonian the odd sectors are
            # useless.
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
    block_size = pars[:block_size]
    threads = pars[:threads]
    BLAS.set_num_threads(threads)
    meratypestr = pars[:meratype]
    if meratypestr == "binary"
        meratype = BinaryMERA
    elseif meratypestr == "ternary"
        meratype = TernaryMERA
    else
        msg = "Unknown MERA type: $(meratypestr)"
        throw(ArgumentError(msg))
    end
    global version

    mkpath(datafolder)
    filename = "MERA_$(model)_$(meratypestr)_$(chi)_$(block_size)_$(symmetry)_$(layers)_$(version)"
    path = "$datafolder/$filename.jlm"
    matlab_folder = "./matlabdata"
    mkpath(matlab_folder)
    matlab_path = "$(matlab_folder)/$(filename).mat"

    if loadfromdisk && isfile(path)
        @info("Found $filename on disk, loading it.")
        m = load_mera(path)
        return m
    else
        @info("Did not find $filename on disk, generating it.")
        if model == "XXZ"
            h, dmax = build_H_XXZ(pars[:Delta]; symmetry=symmetry, block_size=block_size)
        elseif model == "Ising"
            h, dmax = build_H_Ising(pars[:h]; symmetry=symmetry, block_size=block_size)
        else
            msg = "Unknown model: $(model)"
            throw(ArgumentError(msg))
        end
        normalization(x) = normalize_energy(x, dmax, block_size)

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
            m = random_MERA(meratype, Vs)

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
