using AbstractEmulator
using SimpleChains
using Static
using Test

n = 100
m = 200
a = -2
b = 5
InMinMax = hcat(a .*ones(n), b .*ones(n))

@testset "AbstractEmulators test" begin
    x = randn(a:b, n)
    maximin_input!(x, InMinMax)
    @test any(x .>=0 .& x .<=1)
end
