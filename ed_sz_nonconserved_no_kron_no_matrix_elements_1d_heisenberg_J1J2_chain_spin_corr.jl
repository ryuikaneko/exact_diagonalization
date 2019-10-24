#versioninfo()

#import Pkg; Pkg.add("Arpack")
#import Pkg; Pkg.add("ArgParse")
#import Pkg; Pkg.add("LinearMaps")


using LinearAlgebra
using Arpack
using SparseArrays
using ArgParse
using LinearMaps


function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--N"
            help = "set Nsize (should be >=4)"
            arg_type = Int
            default = 8
        "--J1"
            help = "set J1"
            arg_type = Float64
            default = 1.0
        "--J2"
            help = "set J2"
            arg_type = Float64
            default = 0.0
    end
    return parse_args(s)
end

function make_hamiltonian(Jxx,Jzz,list_isite1,list_isite2,N,Nint,Nhilbert)
#    listki = ones(Int,(Nint+1)*Nhilbert)
#    loc = ones(Int,(Nint+1)*Nhilbert)
#    elemnt = zeros((Nint+1)*Nhilbert)
#    for k in 0:Nint
#        for i in 0:Nhilbert-1
#            listki[k*Nhilbert+i+1] = i+1
#        end
#    end
    function get_vec!(vecnew::AbstractVector,vec::AbstractVector)
        lngth = length(vec)
        length(vecnew) == lngth || throw(DimensionMismatch())
        for i in 0:Nhilbert-1
            vecnew[i+1] = 0.0
        end
        for k in 0:Nint-1
            isite1 = list_isite1[k+1]
            isite2 = list_isite2[k+1]
            is1 = 1<<isite1
            is2 = 1<<isite2
            is12 = is1 + is2
            wght = 2.0*Jxx[k+1]
            diag = Jzz[k+1]
            for i in 0:Nhilbert-1
                ibit = i & is12
#                loc[Nint*Nhilbert+i+1] = i+1
                if (ibit==0 || ibit==is12)
                    vecnew[i+1] += +diag*vec[i+1]
#                    elemnt[Nint*Nhilbert+i+1] += diag
                else
                    vecnew[i+1] += -diag*vec[i+1]
#                    elemnt[Nint*Nhilbert+i+1] -= diag
                    iexchg = xor(i,is12)
                    vecnew[i+1] += wght*vec[iexchg+1]
#                    elemnt[k*Nhilbert+i+1] = wght
#                    loc[k*Nhilbert+i+1] = iexchg+1
                end
            end
        end
        return vecnew
    end
    (vecnew,vec) -> get_vec!(vecnew,vec)
#    HamCSC = dropzeros(sparse(listki,loc,elemnt))
#    return HamCSC
end

function calc_zcorr(Nhilbert,Ncorr,list_corr_isite1,list_corr_isite2,psi)
    szz = zeros(Float64,Ncorr)
    for k in 0:Ncorr-1 # loop for all bonds for correlations
        isite1 = list_corr_isite1[k+1]
        isite2 = list_corr_isite2[k+1]
        is1 = 1<<isite1
        is2 = 1<<isite2
        is12 = is1 + is2
        corr = 0.0
        for i in 0:Nhilbert-1 # loop for all spin configurations
            ibit = i & is12
            if (ibit==0 || ibit==is12) # if (spin1,spin2) = (00) or (11): factor = +1
                factor = +1.0
            else # if (spin1,spin2) = (01) or (10): factor = -1
                factor = -1.0
            end
            corr += factor*abs(psi[i+1])^2
        end
        szz[k+1] = 0.25 * corr
        if (isite1==isite2)
            szz[k+1] = 0.25
        end
    end
    return szz
end

function calc_xcorr(Nhilbert,Ncorr,list_corr_isite1,list_corr_isite2,psi)
    sxx = zeros(Float64,Ncorr)
    for k in 0:Ncorr-1 # loop for all bonds for correlations
        isite1 = list_corr_isite1[k+1]
        isite2 = list_corr_isite2[k+1]
        is1 = 1<<isite1
        is2 = 1<<isite2
        is12 = is1 + is2
        corr = 0.0
        for i in 0:Nhilbert-1 # loop for all spin configurations with fixed Sz
            ibit = i & is12
            if (ibit==is1 || ibit==is2) # if (spin1,spin2) = (10) or (01)
                iexchg = xor(i,is12)
                corr += real(psi[i+1]*conj(psi[iexchg+1])) # psi[i]: real
            end
        end
        sxx[k+1] = 0.25 * corr
        if (isite1==isite2)
            sxx[k+1] = 0.25
        end
    end
    return sxx
end

function make_lattice(N,J1,J2)
#    Jxx = []
#    Jzz = []
#    list_isite1 = []
#    list_isite2 = []
    Jxx = Float64[]
    Jzz = Float64[]
    list_isite1 = Int[]
    list_isite2 = Int[]
    Nint = 0
    for i in 0:N-1
        site1 = i
        site2 = mod((i+1),N)
        site3 = mod((i+2),N)
#
        push!(list_isite1,site1)
        push!(list_isite2,site2)
        push!(Jxx,J1)
        push!(Jzz,J1)
        Nint += 1
#
        push!(list_isite1,site1)
        push!(list_isite2,site3)
        push!(Jxx,J2)
        push!(Jzz,J2)
        Nint += 1
    end
    return Jxx, Jzz, list_isite1, list_isite2, Nint
end

function main()
    parsed_args = parse_commandline()
#    println("Parsed args:")
#    for (arg,val) in parsed_args
#        println("  $arg  =>  $val")
#    end

#    N = 16 # should be N>=4
#    J1 = 1.0
#    J2 = 0.0

    N = parsed_args["N"]
    J1 = parsed_args["J1"]
    J2 = parsed_args["J2"]
    Nhilbert = 2^N

    println("J1=",J1)
    println("J2=",J2)
    println("N=",N)
    println("Nhilbert=",Nhilbert)

    println("")
    @time Jxx, Jzz, list_isite1, list_isite2, Nint = make_lattice(N,J1,J2)
    println(Jxx)
    println(Jzz)
    println(list_isite1)
    println(list_isite2)
    println("Nint=",Nint)

    println("")
#    @time Ham = make_hamiltonian(Jxx,Jzz,list_isite1,list_isite2,N,Nint,Nhilbert)
    @time get_vec_LM = make_hamiltonian(Jxx,Jzz,list_isite1,list_isite2,N,Nint,Nhilbert)
    @time Ham = LinearMap(get_vec_LM,Nhilbert;ismutating=true,issymmetric=true)
    #println(Ham)
    #HamElm = SparseArrays.sparse(Ham)
    #println(HamElm)

    #@time ene,vec = eigs(Ham, nev=5, which=:LM)
    @time ene,vec = eigs(Ham, nev=5, which=:SR)
    #println("# GS energy:",ene[1])
    println("# energy:",ene[1]," ",ene[2]," ",ene[3]," ",ene[4]," ",ene[5])
    #vec_sgn = sign(argmax(vec[:,1]))
    #println("# GS wave function:")
    #for i in 0:Nhilbert-1
    #    ii = list_1[i+1]
    #    binii = string(ii,base=2,pad=N)
    #    println(i," ",vec[i+1,1]*vec_sgn," ",binii)
    #end

    println("")
    Ncorr = N # number of total correlations
    list_corr_isite1 = [0 for k in 0:Ncorr-1] # site 1
    list_corr_isite2 = [k for k in 0:Ncorr-1] # site 2
    println(list_corr_isite1)
    println(list_corr_isite2)  
    psi = vec[:,1] # choose the ground state
    @time szz = calc_zcorr(Nhilbert,Ncorr,list_corr_isite1,list_corr_isite2,psi)
    @time sxx = calc_xcorr(Nhilbert,Ncorr,list_corr_isite1,list_corr_isite2,psi)
    ss = szz+sxx+sxx   
    stot2 = N*sum(ss)
    println("# szz:",szz)
    println("# sxx:",sxx)
    println("# ss:",ss)
    println("# stot(stot+1):",stot2)

end


main()
