module IsingAnyons
export IsingAnyon

using TensorKit
import TensorKit.⊗, TensorKit.dim
import TensorKit.Nsymbol, TensorKit.Fsymbol, TensorKit.Rsymbol
import TensorKit.FusionStyle, TensorKit.BraidingStyle
import TensorKit.fusiontensor

struct IsingAnyon <: Sector
    s::Symbol
    function IsingAnyon(s)
        if !(s in (:I, :σ, :ψ))
            throw(ValueError("Unknown IsingAnyon $s."))
        end
        new(s)
    end
end

const all_anyons = (IsingAnyon(:I), IsingAnyon(:σ), IsingAnyon(:ψ))

function Base.convert(::Type{Int}, a::IsingAnyon)
    if a.s == :I
        return 1
    elseif a.s == :σ
        return 2
    else
        return 3
    end
end

Base.convert(::Type{IsingAnyon}, s::Symbol) = IsingAnyon(s)
Base.convert(::Type{Symbol}, a::IsingAnyon) = a.s

Base.one(::Type{IsingAnyon}) = IsingAnyon(:I)
Base.conj(s::IsingAnyon) = s
function ⊗(a::IsingAnyon, b::IsingAnyon)
    if a.s == :I
        return (b,)
    elseif b.s == :I
        return (a,)
    elseif a.s == :ψ && b.s == :ψ
        return (IsingAnyon(:I),)
    elseif (a.s == :σ && b.s == :ψ) || (a.s == :ψ && b.s == :σ)
        return (IsingAnyon(:σ),)
    elseif a.s == :σ && b.s == :σ
        return (IsingAnyon(:I), IsingAnyon(:ψ))
    end
end

dim(a::IsingAnyon) = a.s == :σ ? sqrt(2) : 1.0
# TODO This is cheating.
#dim(s::IsingAnyon) = length(triplets(s))

Base.@pure FusionStyle(::Type{IsingAnyon}) = SimpleNonAbelian()
# TODO BraidingStyle should really be Anyonic(), but that's unimplemented at
# the moment.
Base.@pure BraidingStyle(::Type{IsingAnyon}) = Bosonic()

function Nsymbol(a::IsingAnyon, b::IsingAnyon, c::IsingAnyon)
    return ((a.s == :I && b.s == c.s)
            || (b.s == :I && a.s == c.s)
            || (c.s == :I && a.s == b.s)
            || (a.s == :σ && b.s == :σ && c.s == :ψ)
            || (a.s == :σ && b.s == :ψ && c.s == :σ)
            || (a.s == :ψ && b.s == :σ && c.s == :σ)
           )
end

function Fsymbol(a::IsingAnyon, b::IsingAnyon, c::IsingAnyon,
                 d::IsingAnyon, e::IsingAnyon, f::IsingAnyon)
    Nsymbol(a, b, e) || return 0.0
    Nsymbol(e, c, d) || return 0.0
    Nsymbol(b, c, f) || return 0.0
    Nsymbol(a, f, d) || return 0.0
    if a.s == b.s == c.s == d.s == :σ
        if e.s == f.s == :ψ
            return -1.0/sqrt(2.0)
        else
            return 1.0/sqrt(2.0)
        end
    end
    if e.s == f.s == :σ
        if a.s == c.s == :σ && b.s == d.s == :ψ
            return -1.0
        elseif a.s == c.s == :ψ && b.s == d.s == :σ
            return -1.0
        end
    end
    return 1.0
end

function Rsymbol(a::IsingAnyon, b::IsingAnyon, c::IsingAnyon)
    Nsymbol(a, b, c) || return 0.
    return 1.0
end

function triplets(a::IsingAnyon)
    trips = Tuple{IsingAnyon, IsingAnyon, IsingAnyon}[]
    for b in all_anyons, c in a ⊗ b
            push!(trips, (a, b, c))
    end
    return trips
end

function fusiontensor(b::IsingAnyon, c::IsingAnyon, f::IsingAnyon, v::Nothing = nothing)
    trips_b, trips_c, trips_f = map(triplets, (b, c, f))
    Db, Dc, Df = map(length, (trips_b, trips_c, trips_f))
    C = zeros(Float64, Db, Dc, Df)
    for e in all_anyons
        for a in e ⊗ b, d in e ⊗ c
            elem = Fsymbol(a, b, c, d, e, f)
            indb = first(indexin(((b, e, a),), trips_b))
            indc = first(indexin(((c, d, e),), trips_c))
            indf = first(indexin(((f, a, d),), trips_f))
            if indb != nothing && indc != nothing && indf != nothing
                C[indb, indc, indf] = elem
            end
        end
    end
    return C
end

Base.show(io::IO, ::Type{IsingAnyon}) = print(io, "IsingAnyon")
function Base.show(io::IO, s::IsingAnyon)
    return get(io, :compact, false) ? print(io, s.s) : print(io, "IsingAnyon(", s.s, ")")
end

Base.hash(s::IsingAnyon, h::UInt) = hash(s.s, h)
Base.isless(a::IsingAnyon, b::IsingAnyon) = isless(convert(Int, a), convert(Int, b))

end
