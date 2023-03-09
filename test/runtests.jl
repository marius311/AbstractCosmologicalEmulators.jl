using AbstractEmulator
using SimpleChains
using Static
using Test

m = 100
n = 300
InMinMax = hcat(zeros(m), ones(m))

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
end
