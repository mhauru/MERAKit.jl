# Utilities for demo.jl
module DemoTools

using LinearAlgebra
using TensorKit
using JLD2
using MERA

"""
    block_op(op, num_sites)

Take an operator `op` that defines the local term of a global operator, and block sites
together to form a new operator for which each site corresponds to `num_sites` sites of the
original operator, and which sums up to the same global operator. `num_sites` should be a
power of 2.
"""
function block_op(op::MERA.SquareTensorMap{2}, num_sites)
    while num_sites > 1
        VL = space(op, 1)
        VR = space(op, 2)
        eyeL = id(VL)
        eyeR = id(VR)
        opcross = eyeR ⊗ op ⊗ eyeL
        opleft = op ⊗ eyeL ⊗ eyeR
        opright = eyeL ⊗ eyeR ⊗ op
        op_new_unfused = opcross + 0.5*(opleft + opright)
        fusionspace = domain(op)
        fusetop = isomorphism(fuse(fusionspace), fusionspace)
        fusebottom = isomorphism(fusionspace, fuse(fusionspace))
        op = (fusetop ⊗ fusetop) * op_new_unfused * (fusebottom ⊗ fusebottom)
        num_sites /= 2
    end
    if num_sites != 1
        msg = "`num_sites` needs to be a power of 2"
        throw(ArgumentError(msg))
    end
    return op
end

function block_op(op::MERA.SquareTensorMap{1}, num_sites)
    while num_sites > 1
        V = space(op, 1)
        eye = id(V)
        opleft = op ⊗ eye
        opright = eye ⊗ op
        op_new_unfused = opleft + opright
        fusionspace = domain(op_new_unfused)
        fusetop = isomorphism(fuse(fusionspace), fusionspace)
        fusebottom = isomorphism(fusionspace, fuse(fusionspace))
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
    XXZ_hamiltonian(J_xy, J_z; symmetry=:none, block_size=1)

Return the local Hamiltonian term for the XXZ model: J_xy*XX + J_xy*YY + J_z*ZZ.

`symmetry` should be `:none` or `:group`, and determmines whether the Hamiltonian should be
an explicitly U(1) symmetric `TensorMap` or a dense one. `block_size` determines how many
sites to block together, and should be a power of 2. If sites are blocked, then the
Hamiltonian is divided by a constant to make the energy per site be the same as for the
original.
"""
function XXZ_hamiltonian(J_xy, J_z; symmetry=:none, block_size=1)
    if symmetry == :U1 || symmetry == :group
        V = ℂ[U₁](-1=>1, 1=>1)
        Z = TensorMap(zeros, Float64, V ← V)
        Z.data[U₁(1)] .= 1.0
        Z.data[U₁(-1)] .= -1.0
        ZZ = Z ⊗ Z
        XXplusYY = TensorMap(zeros, Float64, V ⊗ V ← V ⊗ V)
        XXplusYY.data[U₁(0)] .= [0.0 2.0; 2.0 0.0]
    elseif symmetry == :none
        V = ℂ^2
        Z = TensorMap(zeros, Float64, V ← V)
        Z.data .= [1.0 0.0; 0.0 -1.0]
        ZZ = Z ⊗ Z
        XXplusYY = TensorMap(zeros, Float64, V ⊗ V ← V ⊗ V)
        XXplusYY.data[2,3] = 2.0
        XXplusYY.data[3,2] = 2.0
    else
        error("Unknown symmetry: $symmetry")
    end
    H = J_xy*XXplusYY + J_z*ZZ
    H = block_op(H, block_size)
    H = H / block_size
    return H
end

"""
    ising_hamiltonian(h=1.0; symmetry=:none, block_size=1)

Return the local Hamiltonian term for the Ising model: -XX - h*Z

`symmetry` should be `:none`, `:group`, or `:anyons` and determmines whether the Hamiltonian
should be an explicitly Z2 symmetric or anyonic `TensorMap`, or a dense one. `block_size`
determines how many sites to block together, and should be a power of 2. If sites are
blocked, then the Hamiltonian is divided by a constant to make the energy per site be the
same as for the original.
"""
function ising_hamiltonian(h=1.0; symmetry=:none, block_size=1)
    if symmetry == :Z2
        V = ℂ[ℤ₂](0=>1, 1=>1)
        # Pauli Z
        Z = TensorMap(zeros, Float64, V ← V)
        Z.data[ℤ₂(0)] .= 1.0
        Z.data[ℤ₂(1)] .= -1.0
        eye = id(V)
        ZI = Z ⊗ eye
        IZ = eye ⊗ Z
        # Pauli XX
        XX = TensorMap(zeros, Float64, V ⊗ V ← V ⊗ V)
        XX.data[ℤ₂(0)] .= [0.0 1.0; 1.0 0.0]
        XX.data[ℤ₂(1)] .= [0.0 1.0; 1.0 0.0]
        H = -(XX + h/2 * (ZI+IZ))
    elseif symmetry == :anyons
        V = RepresentationSpace{IsingAnyon}(:I => 0, :ψ => 0, :σ => 1)
        H = TensorMap(zeros, Float64, V ⊗ V ← V ⊗ V)
        H.data[IsingAnyon(:I)] .= 1.0
        H.data[IsingAnyon(:ψ)] .= -1.0
    elseif symmetry == :none
        V = ℂ^2
        # Pauli matrices
        X = TensorMap(zeros, Float64, V ← V)
        Z = TensorMap(zeros, Float64, V ← V)
        eye = id(V)
        X.data .= [0.0 1.0; 1.0 0.0]
        Z.data .= [1.0 0.0; 0.0 -1.0]
        XX = X ⊗ X
        ZI = Z ⊗ eye
        IZ = eye ⊗ Z
        H = -(XX + h/2 * (ZI+IZ))
    else
        error("Unknown symmetry: $symmetry")
    end
    H = block_op(H, block_size)
    H = H / block_size
    return H
end

"""
    magop(; symmetry = :none, block_size = 1)

Return the magnetization operator for the Ising model, blocked over `block_size` sites.
"""
function magop(; symmetry = :none, block_size = 1)
    if symmetry == :none 
        V = ℂ^2
    elseif symmetry == :group || symmetry == :Z2
        V = Z2Space(0 => 1, 1 => 1)
    else
        msg = "Unknown symmetry: $symmetry"
        raise(ArgumentError(msg))
    end
    X = TensorMap(zeros, Float64, V ← V)
    if symmetry == :none
        X.data .= [0.0 1.0; 1.0 0.0]
    end
    magop = block_op(X, block_size)
    return magop
end

"""
    store_mera(path, m)

Store the MERA `m` on disk at `path`.

The storage format is a JLD2 file of `pseudoserialize(m)`.  We use pseudoserialization to
both work around limitations of JLD(2) and improve compatibility between different versions
of MERA.jl

See also: [`load_mera`](@ref)
"""
function store_mera(path, m)
    deser = pseudoserialize(m)
    @save path deser
end

"""
    load_mera(path)

Load the MERA at stored at `path`.

The storage format is a JLD2 file of `pseudoserialize(m)`.  We use pseudoserialization to
both work around limitations of JLD(2) and improve compatibility between different versions
of MERA.jl

See also: [`store_mera`](@ref)
"""
function load_mera(path)
    @load path deser
    m = depseudoserialize(deser...)
    return m
end

end  # module
