module AbstractCosmologicalEmulators

using Base: @kwdef
using Adapt
using Lux
using SimpleChains

export AbstractTrainedEmulators, LuxEmulator, SimpleChainsEmulator
export maximin_input!, inv_maximin_output!, run_emulator, get_emulator_description,
init_emulator

include("core.jl")
include("initialization.jl")
include("utils.jl")

end # module AbstractCosmologicalEmulators
