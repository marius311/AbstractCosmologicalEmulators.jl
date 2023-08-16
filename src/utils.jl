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

function get_emulator_description(input_dict::Dict)
    if haskey(input_dict, "parameters")
        println("The parameters the model has been trained are, in the following order: "*input_dict["parameters"]*".")
    else
        @warn "We do not know which parameters were included in the emulators training space. Use this trained emulator with caution!"
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

function get_emulator_description(emu::AbstractTrainedEmulators)
    get_emulator_description(emu.Description)
end
