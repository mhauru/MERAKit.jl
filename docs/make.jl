using Documenter
using MERA

makedocs(sitename = "MERA.jl", modules = [MERA])
deploydocs(repo = "github.com/mhauru/MERA.jl.git",)
