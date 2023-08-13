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

function instantiate_NN(input_dict::Dict)
    hidden_layer_tuple = _create_hidden_layers_tuple(input_dict)
    return SimpleChain(static(input_dict["n_input_features"]), hidden_layer_tuple...,
    TurboDense(identity, input_dict["n_output_features"]))
end
