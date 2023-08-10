module AbstractCosmologicalEmulators

using Base: @kwdef
using SimpleChains

export AbstractTrainedEmulators, SimpleChainsEmulator
export maximin_input!, inv_maximin_output!, run_emulator, instantiate_NN,
get_emulator_description

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
    Description::Dict = Dict()
end

function run_emulator(input, emulator::SimpleChainsEmulator)
    return emulator.Architecture(input, emulator.Weights)
end

function _create_layer(input_dict::Dict)
    if input_dict["activation_function"] == "tanh"
        act_func = SimpleChains.tanh
    elseif input_dict["activation_function"] == "relu"
        act_func = SimpleChains.relu
    else
        error("Error in the Activation Function! You choose "*
        string(input_dict["activation_function"])*" which we do not support!")
    end
    return TurboDense(act_func, Int(input_dict["n_neurons"]))
end

function _create_hidden_layers_tuple(input_dict::Dict)
    n_hidden_layers = input_dict["n_hidden_layers"]
    hid_lay_tuple = (_create_layer(input_dict["layers"]["layer_"*string(i)]) for i in 1:n_hidden_layers)
    return hid_lay_tuple
end

function instantiate_NN(input_dict::Dict)
    hidden_layer_tuple = _create_hidden_layers_tuple(input_dict)
    return SimpleChain(static(input_dict["n_input_features"]), hidden_layer_tuple...,
    TurboDense(identity, input_dict["n_output_features"]))
end

function get_emulator_description(input_dict::Dict)
    if haskey(input_dict, "parameters")
        println("The parameters the model has been trained are, in the following order: "*input_dict["parameters"]*".")
    else
        @warn "We don't know which parameters were included in the emulators training space. Use this trained emulator with caution!"
    end
    if haskey(input_dict, "author")
        println("The emulator has been trained by "*input_dict["author"]*".")
    end
    if haskey(input_dict, "author_email")
        println(input_dict["author"]*" email is "*input_dict["author_email"]*".")
    end
    if haskey(input_dict, "miscellanea")
        println(input_dict["miscellanea"])
    end
    return nothing
end

end # module AbstractCosmologicalEmulators
