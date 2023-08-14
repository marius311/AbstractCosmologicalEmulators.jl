abstract type AbstractTrainedEmulators end

@kwdef mutable struct SimpleChainsEmulator <: AbstractTrainedEmulators
    Architecture
    Weights
    Description::Dict = Dict()
end

function run_emulator(input, emulator::SimpleChainsEmulator)
    return emulator.Architecture(input, emulator.Weights)
end

#@kwdef mutable struct LuxEmulator <: AbstractTrainedEmulators
#    Model
#    Device::Lux.AbstractLuxDevice
#end

@kwdef mutable struct LuxEmulator <: AbstractTrainedEmulators
    Model
    Parameters
    State
    Device::Lux.AbstractLuxDevice
end

dev_gpu(Lux.apply(new_model, dev_gpu(x), tstate.parameters, tstate.states)[1])

function run_emulator(input, emulator::LuxEmulator)
    return emulator.Device(Lux.apply(emulator.Model, emulator.Device(input),
                           emulator.Parameters, emulator.States)[1])
end
