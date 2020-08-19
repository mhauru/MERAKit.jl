module MERA

using TensorKit
using TensorKitManifolds
using KrylovKit
using OptimKit
using TupleTools
using NamedTupleTools
using Printf
using LinearAlgebra
using Logging

export GenericMERA
export Layer
export SimpleLayer
export TernaryMERA
export BinaryMERA
export ModifiedBinaryMERA
export TernaryLayer
export BinaryLayer
export ModifiedBinaryLayer
export SquareTensorMap
export ascend
export ascended_operator
export scale_invariant_operator_sum
export descend
export fixedpoint_densitymatrix
export environment
export remove_symmetry
export layertype
export baselayertype
export operatortype
export get_layer
export get_disentangler
export get_isometry
export num_translayers
export outputspace
export inputspace
export internalspace
export causal_cone_width
export scalefactor
export densitymatrix
export densitymatrices
export densitymatrix_entropies
export random_MERA
export randomlayer
export scalingdimensions
export replace_layer
export release_transitionlayer
export expand_bonddim
export expand_internal_bonddim
export expect
export minimize_expectation
export pseudoserialize
export depseudoserialize
export projectisometric
export projectisometric!
export reset_storage
export tensorwise_sum
export tensorwise_scale
export gradient
export retract
export transport!
export inner

include("tensortools.jl")
include("layer.jl")
include("meracache.jl")
include("genericmera.jl")
include("simplelayer.jl")
include("ternarylayer.jl")
include("binarylayer.jl")
include("modifiedbinaryop.jl")
include("modifiedbinarylayer.jl")

# properties of instances
layertype(m::Union{GenericMERA,MERACache,Layer}) = layertype(typeof(m))
baselayertype(m::Union{GenericMERA,MERACache,Layer}) = baselayertype(typeof(m))
operatortype(m::Union{GenericMERA,MERACache,Layer}) = operatortype(typeof(m))
scalefactor(m::Union{GenericMERA,MERACache,Layer}) = scalefactor(typeof(m))
causal_cone_width(m::Union{GenericMERA,MERACache,Layer}) = causal_cone_width(typeof(m))
Base.eltype(m::Union{GenericMERA,MERACache,Layer}) = eltype(typeof(m))


end  # module
