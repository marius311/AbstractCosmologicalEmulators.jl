function _create_layer(input_dict::Dict)
    if input_dict["activation_function"] == "tanh"
        act_func = SimpleChains.tanh
    elseif input_dict["activation_function"] == "relu"
        act_func = SimpleChains.relu
    else
        error("Error in the Activation Function. You choose "*
        string(input_dict["activation_function"])*" which we do not support.")
    end
    return TurboDense(act_func, Int(input_dict["n_neurons"]))
end

function _create_hidden_layers_tuple(input_dict::Dict)
    n_hidden_layers = input_dict["n_hidden_layers"]
    return (_create_layer(input_dict["layers"]["layer_"*string(i)])
                     for i in 1:n_hidden_layers)
end

function _create_layer_lux(activation_function, n_in::Int, n_out::Int)
    if activation_function == "tanh"
        act_func = Lux.tanh
    elseif activation_function == "relu"
        act_func = Lux.relu
    else
        error("Error in the Activation Function. You choose "*
        string(activation_function)*" which we do not support.")
    end
    return Dense(n_in => n_out, act_func)
end

function _create_hidden_layers_tuple_lux(input_dict::Dict)
    n_hidden_layers = input_dict["n_hidden_layers"]
    in_array, out_array = create_in_out_arrays(input_dict)
    intermediate = (_create_layer_lux(
        input_dict["layers"]["layer_"*string(j)]["activation_function"],
        in_array[j], out_array[j]) for j in 1:n_hidden_layers)
    return (intermediate..., Dense(in_array[end],out_array[end]))
end

function instantiate_NN(input_dict::Dict)
    hidden_layer_tuple = _create_hidden_layers_tuple(input_dict)
    return SimpleChain(static(input_dict["n_input_features"]), hidden_layer_tuple...,
    TurboDense(identity, input_dict["n_output_features"]))
end

function instantiate_NN_lux(input_dict::Dict)
    hidden_layer_tuple = _create_hidden_layers_tuple_lux(input_dict)
    return Chain(hidden_layer_tuple...)
end

function create_weight_bias(i::Int, n_in::Int, n_out::Int, weight_bias, NN_dict::Dict)
    weight = reshape(weight_bias[i:i+n_out*n_in-1], n_out, n_in)
    bias = weight_bias[i+n_out*n_in:i+n_out*n_in+n_out-1]
    i += n_out*n_in+n_out-1+1
    return (weight = weight, bias = bias)
end

function create_in_out_arrays(NN_dict::Dict)
    n = NN_dict["n_hidden_layers"]
    in_array  = zeros(Int64, n+1)
    out_array = zeros(Int64, n+1)
    in_array[1] = NN_dict["n_input_features"]
    out_array[end] = NN_dict["n_output_features"]
    for i in 1:n
        in_array[i+1] = NN_dict["layers"]["layer_"*string(i)]["n_neurons"]
        out_array[i] = NN_dict["layers"]["layer_"*string(i)]["n_neurons"]
    end
    return in_array, out_array
end

function create_i_array(in_array::Vector, out_array::Vector)
    i_array = similar(in_array)
    i_array[1] = 1
    for i in 1:length(i_array)-1
        i_array[i+1] = i_array[i]+in_array[i]*out_array[i]+out_array[i]
    end
    @info i_array
    return i_array
end

function create_lux_params(NN_dict::Dict, weights)
    in_array, out_array = create_in_out_arrays(NN_dict)
    i_array = create_i_array(in_array, out_array)
    params = [create_weight_bias(i_array[j], in_array[j], out_array[j], weights, NN_dict) for j in 1:NN_dict["n_hidden_layers"]+1]
    layer = [Symbol("layer_"*string(j)) for j in 1:NN_dict["n_hidden_layers"]+1]
    return (; zip(layer, params)...)
end

function create_lux_states(NN_dict::Dict)
    params = [NamedTuple() for j in 1:NN_dict["n_hidden_layers"]+1]
    layer = [Symbol("layer_"*string(j)) for j in 1:NN_dict["n_hidden_layers"]+1]
    return (; zip(layer, params)...)
end

function create_lux_params_states(NN_dict::Dict, weights)
    return create_lux_params(NN_dict, weights), create_lux_states(NN_dict)

end

function instantiate_lux_emulator(NN_dict::Dict, weight)
    params, states = create_lux_params_states(NN_dict, weight)
    model = instantiate_NN_lux(input_dict::Dict)
    return LuxEmulator(Model = model, Parameters = params, State = states,
    Device = Lux.cpu_device())
end
