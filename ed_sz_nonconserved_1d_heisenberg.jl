#versioninfo()

#import Pkg; Pkg.add("Arpack")
#import Pkg; Pkg.add("ArgParse")


using LinearAlgebra
using Arpack
using SparseArrays
using ArgParse


function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--N"
            help = "set Nsize (should be >=4)"
            arg_type = Int
            default = 8
    end
    return parse_args(s)
end

function make_spin()
    S0 = sparse([1,2],[1,2],[1,1])
    Sx = sparse([1,2],[2,1],[1,1])
    Sy = sparse([1,2],[2,1],[-1im,1im])
    Sz = sparse([1,2],[1,2],[1,-1])
    return S0, Sx, Sy, Sz
end

function make_interaction_list(N,Nbond)
    list_site1 = [i for i in 1:N]
    list_site2 = [mod(i,N)+1 for i in 1:N]
    list_Jxx = ones(N)
    list_Jyy = ones(N)
    list_Jzz = ones(N)
    return list_site1, list_site2, list_Jxx, list_Jyy, list_Jzz
end

function make_hamiltonian(S0,Sx,Sy,Sz,N,Nbond,
    list_site1,list_site2,list_Jxx,list_Jyy,list_Jzz)
    Ham = spzeros(2^N,2^N)
    for bond in 1:Nbond
        i1 = list_site1[bond]
        i2 = list_site2[bond]
        Jxx = list_Jxx[bond]
        Jyy = list_Jyy[bond]
        Jzz = list_Jzz[bond]
        SxSx = 1
        SySy = 1
        SzSz = 1
        for site in 1:N
            if (site==i1 || site==i2)
                SxSx = kron(SxSx,Sx)
                SySy = kron(SySy,Sy)
                SzSz = kron(SzSz,Sz)
            else
                SxSx = kron(SxSx,S0)
                SySy = kron(SySy,S0)
                SzSz = kron(SzSz,S0)
            end
#        println(bond,site,SxSx)
#        println(bond,site,SySy)
#        println(bond,site,SzSz)
        end
#        Ham += Jxx * SxSx + Jyy * SySy + Jzz * SzSz
        Ham += real(Jxx * SxSx + Jyy * SySy + Jzz * SzSz)
    end
#    println(Ham)
    return Ham
end

function main()
    parsed_args = parse_commandline()
#    println("Parsed args:")
#    for (arg,val) in parsed_args
#        println("  $arg  =>  $val")
#    end

    N = parsed_args["N"]
    Nbond = N
    println("N=",N)

    S0, Sx, Sy, Sz = make_spin()
    println("S0=",Matrix(S0))
    println("Sx=",Matrix(Sx))
    println("Sy=",Matrix(Sy))
    println("Sz=",Matrix(Sz))

    list_site1, list_site2, list_Jxx, list_Jyy, list_Jzz = make_interaction_list(N,Nbond)
    println("list_site1=",list_site1)
    println("list_site2=",list_site2)
    println("list_Jxx=",list_Jxx)
    println("list_Jyy=",list_Jyy)
    println("list_Jzz=",list_Jzz)

    @time Ham = make_hamiltonian(S0,Sx,Sy,Sz,N,Nbond,list_site1,list_site2,list_Jxx,list_Jyy,list_Jzz)

    @time ene,vec = eigs(Ham, nev=5, which=:LM)
    println("# energy:",ene[1]," ",ene[2]," ",ene[3]," ",ene[4]," ",ene[5])
end


main()
