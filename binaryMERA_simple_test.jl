using Test
include("binaryMERA_simple.jl")

testeval = true
testthrows = true

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Simple tests

if testeval
    chi = 3
    layers = 4
    m = build_random_MERA(chi, layers)
    top = m.top
    @show norm(top)

    u2 = m.uw_list[2][1]
    chieye = collect(Diagonal(ones(chi)))
    @tensor twositeeye[-1,-2,-11,-12] := chieye[-1,-11] * chieye[-2,-12]
    @tensor uuconj[-1,-2,-11,-12] := u2[-1,-2,1,2] * conj(u2)[-11,-12,1,2]
    @show norm(uuconj - twositeeye)

    println()

    for i in 1:layers
        randomop1 = random_complex_tensor(chi, 6)
        randomop2 = random_complex_tensor(chi, 6)
        #opdown = desc_threesite(randomop, m; startscale=i+1, endscale=i)
        #opdownup = asc_threesite(opdown, m; startscale=i, endscale=i+1)
        #@show norm(randomop - opdownup)/norm(randomop)
        u, w = m.uw_list[i]
        down1 = binary_descend_threesite(randomop1, u, w; pos=:a)
        up2 = binary_ascend_threesite(randomop2, u, w; pos=:a)
        @tensor e1 := down1[1,2,3,11,12,13] * randomop2[11,12,13,1,2,3]
        @tensor e2 := up2[1,2,3,11,12,13] * randomop1[11,12,13,1,2,3]
        @show abs(e1[1] - e2[2]) / abs(e1[1])
    end

    println()

    for i in 1:(layers+1)
        @tensor(
                threesiteeye[-1,-2,-3,-11,-12,-13] :=
                chieye[-1,-11] * chieye[-2,-12] * chieye[-3,-13]
               )
        expectationvalue = expect(threesiteeye, m; evalscale=i)
        @show expectationvalue, i
    end

    println()

    randomop = random_complex_tensor(chi, 6)
    for i in 1:(layers+1)
        expectationvalue = expect(randomop, m; evalscale=i)
        @show expectationvalue, i
    end
end

if testthrows
    # Test some error throwing
    println()
    uw_list = m.uw_list
    top = m.top
    randlayer = rand(1:layers)
    u, w = uw_list[randlayer]

    for l in 1:layers
        println("Checking error throws, layer = $l.")
        uw_list_mod = copy(uw_list)
        w_mod = w[1:chi-1, :, :]
        uw_list_mod[l] = (u, w_mod)
        @test_throws ArgumentError MERA(uw_list_mod, top)

        uw_list_mod = copy(uw_list)
        u_mod = u[:, 1:chi-1, :, :]
        uw_list_mod[l] = (u_mod, w)
        @test_throws ArgumentError MERA(uw_list_mod, top)

        top_mod = m.top[:, 1:chi-1, :]
        @test_throws ArgumentError MERA(uw_list, top_mod)

        m_copy = MERA(copy(uw_list), copy(top))
        @test_throws ArgumentError set_u!(m_copy, u_mod, l)

        m_copy = MERA(copy(uw_list), copy(top))
        @test_throws ArgumentError set_w!(m_copy, w_mod, l)

        m_copy = MERA(copy(uw_list), copy(top))
        @test_throws ArgumentError set_top!(m_copy, top_mod)
    end
    println()
end
