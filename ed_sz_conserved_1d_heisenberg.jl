versioninfo()

import Pkg; Pkg.add("Arpack")
import Pkg; Pkg.add("ArgParse")


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
        "--Sz"
            help = "set Sz"
            arg_type = Int
            default = 0
    end
    return parse_args(s)
end

function snoob(x)
    next = 0
    if(x>0)
        smallest = x & -(x)
        ripple = x + smallest
        ones = xor(x,ripple)
        ones = div((ones >> 2),smallest)
        next = ripple | ones
    end
    return next
end

function count_bit(n)
    count = 0
    while (n>0)
        count += n & 1
        n >>= 1
    end
    return count 
end

function init_parameters(N,Sz)
    Nup = div(N,2) + Sz
    Nhilbert = binomial(N,Nup)
    ihfbit = 1 << (div(N,2))
    irght = ihfbit-1
    ilft = xor(((1<<N)-1),irght)
    iup = (1<<(N-Nup))-1
    return Nup, Nhilbert, ihfbit, irght, ilft, iup
end

function make_list(N,Nup,Nhilbert,ihfbit,irght,ilft,iup)
    list_1 = zeros(Int,Nhilbert)
    list_ja = zeros(Int,ihfbit)
    list_jb = zeros(Int,ihfbit)
    ii = iup
    ja = 0
    jb = 0
    ia_old = ii & irght
    ib_old = div((ii & ilft),ihfbit)
    list_1[1] = ii
    list_ja[ia_old+1] = ja
    list_jb[ib_old+1] = jb
    ii = snoob(ii)
    for i in 1:Nhilbert-1
        ia = ii & irght
        ib = div((ii & ilft),ihfbit)
        if ib == ib_old
            ja += 1
        else
            jb += ja+1
            ja = 0
        end
        list_1[i+1] = ii
        list_ja[ia+1] = ja
        list_jb[ib+1] = jb
        ia_old = ia
        ib_old = ib
        ii = snoob(ii)
    end
    return list_1, list_ja, list_jb
end

function get_ja_plus_jb(ii,irght,ilft,ihfbit,list_ja,list_jb)
    ia = ii & irght
    ib = div((ii & ilft),ihfbit)
    ja = list_ja[ia+1]
    jb = list_jb[ib+1]
    return ja+jb
end

function make_hamiltonian(J1,D1,N,Nhilbert,irght,ilft,ihfbit,list_1,list_ja,list_jb)
    listki = ones(Int,(N+1)*Nhilbert)
    loc = ones(Int,(N+1)*Nhilbert)
    elemnt = zeros((N+1)*Nhilbert)
    for k in 0:N
        for i in 0:Nhilbert-1
            listki[k*Nhilbert+i+1] = i+1
        end
    end
    for k in 0:N-1
        isite1 = k
        isite2 = mod((k+1),N)
        is1 = 1<<isite1
        is2 = 1<<isite2
        is0 = is1 + is2
        wght = -2.0*J1[k+1]
        diag = wght*0.5*D1[k+1]
        for i in 0:Nhilbert-1
            ii = list_1[i+1]
            ibit = ii & is0
            if (ibit==0 || ibit==is0)
                elemnt[N*Nhilbert+i+1] -= diag
                loc[N*Nhilbert+i+1] = i+1
            else
                elemnt[N*Nhilbert+i+1] += diag
                loc[N*Nhilbert+i+1] = i+1
                iexchg = xor(ii,is0)
                newcfg = get_ja_plus_jb(iexchg,irght,ilft,ihfbit,list_ja,list_jb)
                elemnt[k*Nhilbert+i+1] = -wght
                loc[k*Nhilbert+i+1] = newcfg+1
            end
        end
    end
    HamCSC = dropzeros(sparse(listki,loc,elemnt))
    return HamCSC
end

function main()
    parsed_args = parse_commandline()
#    println("Parsed args:")
#    for (arg,val) in parsed_args
#        println("  $arg  =>  $val")
#    end

#    N = 16 # should be N>=4
#    Sz = 0

    N = parsed_args["N"]
    Sz = parsed_args["Sz"]

    Nup, Nhilbert, ihfbit, irght, ilft, iup = init_parameters(N,Sz)

    binirght = string(irght,base=2,pad=N)
    binilft = string(ilft,base=2,pad=N)
    biniup = string(iup,base=2,pad=N)

    println("N=",N)
    println("Sz=",Sz)
    println("Nup=",Nup)
    println("Nhilbert=",Nhilbert)
    println("ihfbit=",ihfbit)
    println("irght,binirght=",irght," ",binirght)
    println("ilft,binilft=",ilft," ",binilft)
    println("iup,biniup=",iup," ",biniup)

    @time list_1, list_ja, list_jb = make_list(N,Nup,Nhilbert,ihfbit,irght,ilft,iup)

    #println("list_1=",list_1)
    #println("list_ja=",list_ja)
    #println("list_jb=",list_jb)

    #println("")
    #println("i ii binii ja+jb")
    #for i in 0:Nhilbert-1
    #    ii = list_1[i+1]
    #    binii = string(ii,base=2,pad=N)
    #    ind = get_ja_plus_jb(ii,irght,ilft,ihfbit,list_ja,list_jb)
    #    println(i," ",ii," ",binii," ",ind)
    #end

    J1 = ones(N) # J_{ij}>0: AF
    D1 = ones(N) # D_{ij}>0: AF

    @time Ham = make_hamiltonian(J1,D1,N,Nhilbert,irght,ilft,ihfbit,list_1,list_ja,list_jb)
    #println(Ham)

    @time ene,vec = eigs(Ham, nev=5, which=:LM)
    #println("# GS energy:",ene[1])
    println("# energy:",ene[1]," ",ene[2]," ",ene[3]," ",ene[4]," ",ene[5])
    #vec_sgn = sign(argmax(vec[:,1]))
    #println("# GS wave function:")
    #for i in 0:Nhilbert-1
    #    ii = list_1[i+1]
    #    binii = string(ii,base=2,pad=N)
    #    println(i," ",vec[i+1,1]*vec_sgn," ",binii)
    #end
end


main()
