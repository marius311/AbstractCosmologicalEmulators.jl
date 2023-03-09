module AbstractEmulator

using Base: @kwdef
using LoopVectorization
using SimpleChains
#TODO: add the FluxEmulators
export AbstractTrainedEmulators, SimpleChainsEmulator
export maximin_input!, inv_maximin_output!, run_emulator

function maximin_input!(input, in_MinMax)
    println("Pippo!")
    param_dim = size(x)[1]
    for i in 1:param_dim
        input[i] .-= in_MinMax[i,1]
        input[i] ./= (in_MinMax[i,2]-in_MinMax[i,1])
    end
end

function inv_maximin_output!(x, out_MinMax)
    for i in eachindex(x)
        x[i] *= (out_MinMax[i,2]-out_MinMax[i,1])
        x[i] += out_MinMax[i,1]
    end
end

abstract type AbstractTrainedEmulators end

@kwdef mutable struct SimpleChainsEmulator <: AbstractTrainedEmulators
    Architecture
    Weights
end

function run_emulator(input, trained_emulator::SimpleChainsEmulator)
    return trained_emulator.Architecture(input, trained_emulator.Weights)
end

end # module AbstractEmulator
