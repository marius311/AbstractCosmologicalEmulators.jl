using AbstractEmulator
using SimpleChains
using Static
using Test

m = 100
n = 200
a = -2
b = 5
InMinMax = hcat(a .*ones(m), b .*ones(m))

@testset "AbstractEmulators test" begin
    x = rand(a:b, m)
    y = rand(a:b, m, n)
    X = deepcopy(x)
    Y = deepcopy(y)
    maximin_input!(x, InMinMax)
    maximin_input!(y, InMinMax)
    @test any(x .>=0 .& x .<=1)
    @test any(y .>=0 .& x .<=1)
    inv_maximin_output!(x, InMinMax)
    inv_maximin_output!(y, InMinMax)
    @test any(x .== X)
    @test any(y .== Y)
end
