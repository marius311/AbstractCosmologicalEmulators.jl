using AbstractCosmologicalEmulators
using JSON
using SimpleChains
using Static
using Test

m = 100
n = 300
InMinMax = hcat(zeros(m), ones(m))
mlpd = SimpleChain(
  static(6),
  TurboDense(tanh, 64),
  TurboDense(tanh, 64),
  TurboDense(relu, 64),
  TurboDense(tanh, 64),
  TurboDense(tanh, 64),
  TurboDense(identity, 40)
)

NN_dict = JSON.parsefile(pwd()*"/testNN.json")
weights = SimpleChains.init_params(mlpd)
sc_emu = SimpleChainsEmulator(Architecture = mlpd, Weights = weights,
                              Description = NN_dict["emulator_description"])

@testset "AbstractEmulators test" begin
    x = rand(m)
    y = rand(m, n)

    X = deepcopy(x)
    Y = deepcopy(y)
    maximin_input!(x, InMinMax)
    maximin_input!(y, InMinMax)
    @test any(x .>=0 .&& x .<=1)
    @test any(y .>=0 .&& y .<=1)
    inv_maximin_output!(x, InMinMax)
    inv_maximin_output!(y, InMinMax)
    @test any(x .== X)
    @test any(y .== Y)
    input = randn(6)
    stack_input = hcat(input, input)
    @test isapprox(run_emulator(input, sc_emu), run_emulator(stack_input, sc_emu)[:,1])
    @test AbstractCosmologicalEmulators._get_nn_simplechains(NN_dict) == mlpd
    lux_emu = init_emulator(NN_dict, weights, LuxEmulator)
    sc_emu_check = init_emulator(NN_dict, weights, SimpleChainsEmulator)
    @test sc_emu_check.Architecture == sc_emu.Architecture
    @test sc_emu_check.Weights == sc_emu.Weights
    @test sc_emu_check.Description == sc_emu.Description
    NN_dict["layers"]["layer_1"]["activation_function"]= "adremxud"
    @test_throws ErrorException AbstractCosmologicalEmulators._get_nn_simplechains(NN_dict)
    @test_throws ErrorException AbstractCosmologicalEmulators._get_nn_lux(NN_dict)
    get_emulator_description(NN_dict["emulator_description"])
    @test_logs (:warn, "We do not know which parameters were included in the emulators training space. Use this trained emulator with caution!") AbstractCosmologicalEmulators.get_emulator_description(Dict("pippo" => "franco"))
    @test_logs (:warn, "No emulator description found!") AbstractCosmologicalEmulators._get_emulator_description_dict(Dict("pippo" => "franco"))
    @test isapprox(run_emulator(input, sc_emu), run_emulator(input, lux_emu))
end
