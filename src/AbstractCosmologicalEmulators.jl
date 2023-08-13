module AbstractCosmologicalEmulators

using Base: @kwdef
using Lux
using SimpleChains

export AbstractTrainedEmulators, SimpleChainsEmulator
export maximin_input!, inv_maximin_output!, run_emulator, instantiate_NN,
get_emulator_description

include("core.jl")
include("initialization.jl")
include("utils.jl")

end # module AbstractCosmologicalEmulators
