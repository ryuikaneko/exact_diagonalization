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
        "--J1"
            help = "set J1"
            arg_type = Float64
            default = 1.0
        "--J2"
            help = "set J2"
            arg_type = Float64
            default = 0.0
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

function make_hamiltonian(Jxx,Jzz,list_isite1,list_isite2,N,Nint,Nhilbert,irght,ilft,ihfbit,list_1,list_ja,list_jb)
    listki = ones(Int,(Nint+1)*Nhilbert)
    loc = ones(Int,(Nint+1)*Nhilbert)
    elemnt = zeros((Nint+1)*Nhilbert)
    for k in 0:Nint
        for i in 0:Nhilbert-1
            listki[k*Nhilbert+i+1] = i+1
        end
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
            ii = list_1[i+1]
            ibit = ii & is12
            loc[Nint*Nhilbert+i+1] = i+1
            if (ibit==0 || ibit==is12)
                elemnt[Nint*Nhilbert+i+1] += diag
            else
                elemnt[Nint*Nhilbert+i+1] -= diag
                iexchg = xor(ii,is12)
                newcfg = get_ja_plus_jb(iexchg,irght,ilft,ihfbit,list_ja,list_jb)
                elemnt[k*Nhilbert+i+1] = wght
                loc[k*Nhilbert+i+1] = newcfg+1
            end
        end
    end
    HamCSC = dropzeros(sparse(listki,loc,elemnt))
    return HamCSC
end

function calc_zcorr(Nhilbert,Ncorr,list_corr_isite1,list_corr_isite2,psi,list_1)
    szz = zeros(Float64,Ncorr)
    for k in 0:Ncorr-1 # loop for all bonds for correlations
        isite1 = list_corr_isite1[k+1]
        isite2 = list_corr_isite2[k+1]
        is1 = 1<<isite1
        is2 = 1<<isite2
        is12 = is1 + is2
        corr = 0.0
        for i in 0:Nhilbert-1 # loop for all spin configurations with fixed Sz
            ii = list_1[i+1]
            ibit = ii & is12 # find sgmz.sgmz|uu> = |uu> or sgmz.sgmz|dd> = |dd>
            if (ibit==0 || ibit==is12) # if (spin1,spin2) = (00) or (11): factor = +1
                factor = +1.0
            else # if (spin1,spin2) = (01) or (10): factor = -1
                factor = -1.0
            end
            corr += factor*psi[i+1]^2 # psi[i]: real
        end
        szz[k+1] = 0.25 * corr
        if (isite1==isite2)
            szz[k+1] = 0.25
        end
    end
    return szz
end

function calc_xcorr(Nhilbert,Ncorr,list_corr_isite1,list_corr_isite2,psi,irght,ilft,ihfbit,list_1,list_ja,list_jb)
    sxx = zeros(Float64,Ncorr)
    for k in 0:Ncorr-1 # loop for all bonds for correlations
        isite1 = list_corr_isite1[k+1]
        isite2 = list_corr_isite2[k+1]
        is1 = 1<<isite1
        is2 = 1<<isite2
        is12 = is1 + is2
        corr = 0.0
        for i in 0:Nhilbert-1 # loop for all spin configurations with fixed Sz
            ii = list_1[i+1]
            ibit = ii & is12 # find sgmz.sgmz|ud> = -|ud> or sgmz.sgmz|du> = -|du>
            if (ibit==is1 || ibit==is2) # if (spin1,spin2) = (10) or (01)
                iexchg = xor(ii,is12) # find S+.S-|du> = |ud> or S-.S+|ud> = |du>
                newcfg = get_ja_plus_jb(iexchg,irght,ilft,ihfbit,list_ja,list_jb)
                corr += psi[i+1]*psi[newcfg+1] # psi[i]: real
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
    Jxx = []
    Jzz = []
    list_isite1 = []
    list_isite2 = []
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
#    Sz = 0
#    J1 = 1.0
#    J2 = 0.0

    N = parsed_args["N"]
    Sz = parsed_args["Sz"]
    J1 = parsed_args["J1"]
    J2 = parsed_args["J2"]

    Nup, Nhilbert, ihfbit, irght, ilft, iup = init_parameters(N,Sz)

    binirght = string(irght,base=2,pad=N)
    binilft = string(ilft,base=2,pad=N)
    biniup = string(iup,base=2,pad=N)

    println("J1=",J1)
    println("J2=",J2)
    println("N=",N)
    println("Sz=",Sz)
    println("Nup=",Nup)
    println("Nhilbert=",Nhilbert)
    println("ihfbit=",ihfbit)
    println("irght,binirght=",irght," ",binirght)
    println("ilft,binilft=",ilft," ",binilft)
    println("iup,biniup=",iup," ",biniup)

    println("")
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

    @time Jxx, Jzz, list_isite1, list_isite2, Nint = make_lattice(N,J1,J2)
    println(Jxx)
    println(Jzz)
    println(list_isite1)
    println(list_isite2)
    println("Nint=",Nint)

    println("")
    @time Ham = make_hamiltonian(Jxx,Jzz,list_isite1,list_isite2,N,Nint,Nhilbert,irght,ilft,ihfbit,list_1,list_ja,list_jb)
    #println(Ham)

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
    @time szz = calc_zcorr(Nhilbert,Ncorr,list_corr_isite1,list_corr_isite2,psi,list_1)
    @time sxx = calc_xcorr(Nhilbert,Ncorr,list_corr_isite1,list_corr_isite2,psi,irght,ilft,ihfbit,list_1,list_ja,list_jb)
    ss = szz+sxx+sxx   
    stot2 = N*sum(ss)
    println("# szz:",szz)
    println("# sxx:",sxx)
    println("# ss:",ss)
    println("# stot(stot+1):",stot2)

end


main()
