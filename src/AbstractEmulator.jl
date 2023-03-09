module AbstractEmulator

using Base: @kwdef
using LoopVectorization
using SimpleChains
#TODO: add the FluxEmulators
export AbstractTrainedEmulators, SimpleChainsEmulator
export maximin_input!, inv_maximin_output!, run_emulator

function maximin_input!(input::Vector, in_MinMax)
    for i in eachindex(input)
        input[i] -= in_MinMax[i,1]
        input[i] /= (in_MinMax[i,2]-in_MinMax[i,1])
    end
end

function inv_maximin_output!(output::Vector, out_MinMax)
    for i in eachindex(output)
        output[i] *= (out_MinMax[i,2]-out_MinMax[i,1])
        output[i] += out_MinMax[i,1]
    end
end

function maximin_input!(input::Matrix, in_MinMax)
    dim_pars, _ = size(input)
    for i in 1:dim_pars
        input[i,:] .-= in_MinMax[i,1]
        input[i,:] ./= (in_MinMax[i,2]-in_MinMax[i,1])
    end
end

function inv_maximin_output!(output::Matrix, out_MinMax)
    dim_out, _ = size(output)
    for i in 1:dim_out
        output[i,:] .*= (out_MinMax[i,2]-out_MinMax[i,1])
        output[i,:] .+= out_MinMax[i,1]
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
