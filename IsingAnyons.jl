module IsingAnyons
export IsingAnyon

using TensorKit
import TensorKit.⊗, TensorKit.dim, TensorKit.SectorSet
import TensorKit.Nsymbol, TensorKit.Fsymbol, TensorKit.Rsymbol
import TensorKit.FusionStyle, TensorKit.BraidingStyle

struct IsingAnyon <: Sector
    s::Symbol
    function IsingAnyon(s)
        if !(s in (:I, :ψ, :σ))
            throw(ValueError("Unknown IsingAnyon $s."))
        end
        new(s)
    end
end

Base.one(::Type{IsingAnyon}) = IsingAnyon(:I)
Base.conj(s::IsingAnyon) = s
function ⊗(s1::IsingAnyon, s2::IsingAnyon)
    if s1.s == :I
        return SectorSet{IsingAnyon}((s2,))
    elseif s2.s == :I
        return SectorSet{IsingAnyon}((s1,))
    elseif s1.s == :ψ && s2.s == :ψ
        return SectorSet{IsingAnyon}((IsingAnyon(:I),))
    elseif (s1.s == :σ && s2.s == :ψ) || (s1.s == :ψ && s2.s == :σ)
        return SectorSet{IsingAnyon}((IsingAnyon(:σ),))
    elseif s1.s == :σ && s2.s == :σ
        return SectorSet{IsingAnyon}((IsingAnyon(:I), IsingAnyon(:ψ)))
    end
end

Base.convert(::Type{IsingAnyon}, s::Symbol) = IsingAnyon(s)

dim(s::IsingAnyon) = s == :σ ? sqrt(2) : 1

Base.@pure FusionStyle(::Type{IsingAnyon}) = SimpleNonAbelian()
# TODO BraidingStyle should really be Anyonic(), but that's unimplemented at
# the moment.
Base.@pure BraidingStyle(::Type{IsingAnyon}) = Bosonic()

function Nsymbol(sa::IsingAnyon, sb::IsingAnyon, sc::IsingAnyon)
    return ((sa.s == :I && sb.s == sc.s)
            || (sb.s == :I && sa.s == sc.s)
            || (sc.s == :I && sa.s == sb.s)
            || (sa.s == sb.s && sc.s == :I)
            || (sa.s == :σ && sb.s == :σ && sc.s == :ψ)
            || (sa.s == :σ && sb.s == :ψ && sc.s == :σ)
            || (sa.s == :ψ && sb.s == :σ && sc.s == :σ)
           )
end

function Fsymbol(sa::IsingAnyon, sb::IsingAnyon, sc::IsingAnyon,
                 sd::IsingAnyon, se::IsingAnyon, sf::IsingAnyon) 
    Nsymbol(sa, sb, se) || return 0.0
    Nsymbol(se, sc, sd) || return 0.0
    Nsymbol(sb, sc, sf) || return 0.0
    Nsymbol(sa, sf, sd) || return 0.0
    if sa.s == sb.s == sc.s == sd.s == :σ
        if se.s == sf.s == :ψ
            return -1.0/sqrt(2.0)
        else
            return 1.0/sqrt(2.0)
        end
    end
    if se.s == sf.s == :σ
        if sa.s == sc.s == :σ && sb.s == sd.s == :ψ
            return -1.0
        elseif sa.s == sc.s == :ψ && sb.s == sd.s == :σ
            return -1.0
        end
    end
    return 1.0
end

function Rsymbol(sa::IsingAnyon, sb::IsingAnyon, sc::IsingAnyon)
    Nsymbol(sa, sb, sc) || return 0.
    return 1.0
end

Base.show(io::IO, ::Type{IsingAnyon}) = print(io, "IsingAnyon")
function Base.show(io::IO, s::IsingAnyon)
    return get(io, :compact, false) ? print(io, s.s) : print(io, "IsingAnyon(", s.s, ")")
end

Base.hash(s::IsingAnyon, h::UInt) = hash(s.s, h)
function Base.isless(s1::IsingAnyon, s2::IsingAnyon)
    if s1.s == :I
        return s2.s != :I
    elseif s1.s == :ψ
        return s2.s == :σ
    else
        return false
    end
end

end
