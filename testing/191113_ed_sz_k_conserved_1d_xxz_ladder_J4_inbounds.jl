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
        "--Lx"
            help = "set Lx (should be >=2)"
            arg_type = Int
            default = 4
        "--Ly"
            help = "set Ly (=2)"
            arg_type = Int
            default = 2
        "--Jleg"
            help = "set Jleg"
            arg_type = Float64
            default = 1.0
        "--Jrung"
            help = "set Jrung"
            arg_type = Float64
            default = 1.0
        "--Jising"
            help = "set Jising"
            arg_type = Float64
            default = 1.0
        "--J4"
            help = "set J4"
            arg_type = Float64
            default = 0.0
        "--twoSz"
            help = "set twoSz"
            arg_type = Int
            default = 0
        "--momkx"
            help = "set momkx"
            arg_type = Int
            default = 0
        "--momky"
            help = "set momky"
            arg_type = Int
            default = 0
    end
    return parse_args(s)
end

function get_next_same_nup_state(state)
    next = 0
    if (state>0)
        smallest = state & -(state)
        ripple = state + smallest
        ones = xor(state,ripple)
        ones = div((ones >> 2),smallest)
        next = ripple | ones
    end
    return next
end

function shift_child_BA2AB(s,a,b)
    return ((s<<b)&(((1<<a)-1)<<b))|((s>>a)&((1<<b)-1))
end

function shift_child_CBA2CAB(s,a,b,c)
    left = s&(((1<<c)-1)<<(a+b))
    right = s&((1<<(a+b))-1)
    right = shift_child_BA2AB(right,a,b)
    return left|right
end

function shift_child_DCBA2DBCA(s,a,b,c,d)
    left = s&(((1<<d)-1)<<(a+b+c))
    right = s&((1<<a)-1)
    mid = (s>>a)&((1<<(b+c))-1)
    mid = shift_child_BA2AB(mid,b,c)
    mid = mid<<a
    return left|(mid|right)
end

function shift_y_1spin(state,Lx,Ly,Ns)
    return shift_child_BA2AB(state,Ns-Lx,Lx)
end

function shift_x_1spin(state,Lx,Ly,Ns)
    s = shift_child_CBA2CAB(state,Lx-1,1,Ns-Lx)
    for n in 1:Ly-1
        s = shift_child_DCBA2DBCA(s,n*Lx,Lx-1,1,Ns-(n+1)*Lx)
    end
    return s
end

function check_state(state,Lx,Ly,Ns)
    list_allstate = Int[]
    r = state
    t = state
    push!(list_allstate,t)
    for i in 0:Lx-2
        t = shift_x_1spin(t,Lx,Ly,Ns)
        push!(list_allstate,t)
        if (t < r)
            r = t
        end
    end
    for j in 0:Ly-2
        t = shift_y_1spin(t,Lx,Ly,Ns)
        for i in 0:Lx-1
            t = shift_x_1spin(t,Lx,Ly,Ns)
            push!(list_allstate,t)
            if (t < r)
                r = t
            end
        end
    end
    D = length(unique(list_allstate)) ## remove duplication and count length of a list
#    println(D," ",list_allstate)
    if (r == state)
        return +D
    else
        return -D
    end
end

function calc_exp(Lx,Ly,momkx,momky)
#    return [[exp(-1im*2.0*pi*(expx*momkx/Lx+expy*momky/Ly)) for expy in 0:Ly-1] for expx in 0:Lx-1]
#    return [exp(-1im*2.0*pi*(expx*momkx/Lx+expy*momky/Ly)) for expy in 0:Ly-1, expx in 0:Lx-1]
    @inbounds return [exp(-1im*2.0*pi*(expx*momkx/Lx+expy*momky/Ly)) for expx in 0:Lx-1, expy in 0:Ly-1]
end

function calc_sum_exp(r,Lx,Ly,Ns,expk)
    t = r
#    F2 = expk[0,0]
    F2 = expk[1,1]
    @inbounds for i in 0:Lx-2
        t = shift_x_1spin(t,Lx,Ly,Ns)
        if (t == r)
#            F2 += expk[i+1,0]
            F2 += expk[i+2,1]
        end
    end
    @inbounds for j in 0:Ly-2
        t = shift_y_1spin(t,Lx,Ly,Ns)
        for i in 0:Lx-1
            t = shift_x_1spin(t,Lx,Ly,Ns)
            if (t == r)
#                F2 += expk[i,j+1]
                F2 += expk[i+1,j+2]
            end
        end
    end
    return F2
end

function make_basis(Lx,Ly,Ns,nup,Nbinom,momkx,momky,expk)
#function make_basis(Lx,Ly,Ns,nup,Nbinom,momkx,momky)
    list_state = Int[]
    list_R = Int[]
    list_F2 = Complex{Float64}[]
    first = (1<<(Ns-nup))-1
    last = ((1<<(Ns-nup))-1)<<(nup)
    Nrep = 0
    state = first
    for i in 0:Nbinom-1
        R = check_state(state,Lx,Ly,Ns)
        if (R>=0)
            push!(list_state,state)
            push!(list_R,R)
            Nrep += 1
            F2 = calc_sum_exp(state,Lx,Ly,Ns,expk)
            push!(list_F2,F2)
        end
        state = get_next_same_nup_state(state)
    end
    return list_state, list_R, list_F2, Nrep
#    return list_state, list_R, Nrep
end

function find_representative(state,Lx,Ly,Ns)
    r = state
    t = state
    expx = 0
    expy = 0
    for i in 0:Lx-2
        t = shift_x_1spin(t,Lx,Ly,Ns)
        if (t < r)
            r = t
            expx = i+1
        end
    end
    for j in 0:Ly-2
        t = shift_y_1spin(t,Lx,Ly,Ns)
        for i in 0:Lx-1
            t = shift_x_1spin(t,Lx,Ly,Ns)
            if (t < r)
                r = t
                expx = i
                expy = j+1
            end
        end
    end
    return r, expx, expy
end

function get_spin(state,site)
    return (state>>site)&1
end

function flip_2spins(state,i1,i2)
    return xor(state,((1<<i1)+(1<<i2)))
end

function find_state_2(state,list_1,maxind)
    imin = 0
    imax = maxind-1
    i = div(imin+imax,2) ## make "i" in the loop global
    while true
        i = div(imin+imax,2)
#        if (state < list_1[i])
        if (state < list_1[i+1])
            imax = i-1
#        elseif (state > list_1[i])
        elseif (state > list_1[i+1])
            imin = i+1
        else
            break
        end
        if (imin > imax)
            return -1
        end
    end
    return i
end

function make_hamiltonian_child(Jxx,Jzz,list_site1,list_site2,Nbond1,
    J4xx,J4zz,J4xz,list_site3,list_site4,list_site5,list_site6,Nbond2,
    Nrep,list_state,list_sqrtnorm,Lx,Ly,Ns,momkx,momky,expk)
    Nbond = Nbond1 + 3*Nbond2
#
    listki = ones(Int,(Nbond+1)*Nrep)
    loc = ones(Int,(Nbond+1)*Nrep)
    elemnt = zeros(Complex{Float64},(Nbond+1)*Nrep)
    @inbounds for k in 0:Nbond
        for i in 0:Nrep-1
            listki[k*Nrep+i+1] = i+1
        end
    end
##
## index of loc[]
##   Nbond*Nrep+a: diag 
##   i*Nrep+a: offdiag such as [s2,s1] = 01, 10
##   (Nbond1+i)*Nrep+a: offdiag such as [s6,s5,s4,s3] = 0100, 1000, 0111, 1011
##   (Nbond1+Nbond2+i)*Nrep+a: offdiag such as [s6,s5,s4,s3] = 0001, 0010, 1101, 1110
##   (Nbond1+2*Nbond2+i)*Nrep+a: offdiag such as [s6,s5,s4,s3] = 0101, 0110, 1001, 1010
##
    @inbounds for a in 0:Nrep-1
        sa = list_state[a+1]
        loc[Nbond*Nrep+a+1] = a+1
        for i in 0:Nbond1-1
            i1 = list_site1[i+1]
            i2 = list_site2[i+1]
            wght = 2.0*Jxx[i+1]
            diag = Jzz[i+1]
#            loc[Nbond*Nrep+a+1] = a+1
            if get_spin(sa,i1) == get_spin(sa,i2)
                elemnt[Nbond*Nrep+a+1] += diag
            else
                elemnt[Nbond*Nrep+a+1] -= diag
                bb = flip_2spins(sa,i1,i2)
                sb, expx, expy = find_representative(bb,Lx,Ly,Ns)
                b = find_state_2(sb,list_state,Nrep)
                if (b>=0)
                    elemnt[i*Nrep+a+1] += wght*list_sqrtnorm[b+1]/list_sqrtnorm[a+1]*expk[expx+1,expy+1]
                    loc[i*Nrep+a+1] = b+1
                end
            end
        end
        for i in 0:Nbond2-1
            i3 = list_site3[i+1]
            i4 = list_site4[i+1]
            i5 = list_site5[i+1]
            i6 = list_site6[i+1]
            czz = J4zz[i+1]
            cxz = 2.0*J4xz[i+1]
            cxx = 4.0*J4xx[i+1]
#            loc[Nbond*Nrep+a+1] = a+1
            if get_spin(sa,i3) == get_spin(sa,i4) && get_spin(sa,i5) == get_spin(sa,i6)
                elemnt[Nbond*Nrep+a+1] += czz
            elseif get_spin(sa,i3) == get_spin(sa,i4) && get_spin(sa,i5) != get_spin(sa,i6)
                elemnt[Nbond*Nrep+a+1] -= czz
#
                bb = flip_2spins(sa,i5,i6)
                sb, expx, expy = find_representative(bb,Lx,Ly,Ns)
                b = find_state_2(sb,list_state,Nrep)
                if (b>=0)
                    elemnt[(Nbond1+i)*Nrep+a+1] += cxz*list_sqrtnorm[b+1]/list_sqrtnorm[a+1]*expk[expx+1,expy+1]
                    loc[(Nbond1+i)*Nrep+a+1] = b+1
                end
            elseif get_spin(sa,i3) != get_spin(sa,i4) && get_spin(sa,i5) == get_spin(sa,i6)
                elemnt[Nbond*Nrep+a+1] -= czz
#
                bb = flip_2spins(sa,i3,i4)
                sb, expx, expy = find_representative(bb,Lx,Ly,Ns)
                b = find_state_2(sb,list_state,Nrep)
                if (b>=0)
                    elemnt[(Nbond1+Nbond2+i)*Nrep+a+1] += cxz*list_sqrtnorm[b+1]/list_sqrtnorm[a+1]*expk[expx+1,expy+1]
                    loc[(Nbond1+Nbond2+i)*Nrep+a+1] = b+1
                end
            elseif get_spin(sa,i3) != get_spin(sa,i4) && get_spin(sa,i5) != get_spin(sa,i6)
                elemnt[Nbond*Nrep+a+1] += czz
#
                bb = flip_2spins(sa,i3,i4)
                sb, expx, expy = find_representative(bb,Lx,Ly,Ns)
                b = find_state_2(sb,list_state,Nrep)
                if (b>=0)
                    elemnt[(Nbond1+Nbond2+i)*Nrep+a+1] += -cxz*list_sqrtnorm[b+1]/list_sqrtnorm[a+1]*expk[expx+1,expy+1]
                    loc[(Nbond1+Nbond2+i)*Nrep+a+1] = b+1
                end
#
                bb = flip_2spins(sa,i5,i6)
                sb, expx, expy = find_representative(bb,Lx,Ly,Ns)
                b = find_state_2(sb,list_state,Nrep)
                if (b>=0)  
                    elemnt[(Nbond1+i)*Nrep+a+1] += -cxz*list_sqrtnorm[b+1]/list_sqrtnorm[a+1]*expk[expx+1,expy+1]
                    loc[(Nbond1+i)*Nrep+a+1] = b+1
                end
#
                bbtmp = flip_2spins(sa,i3,i4)
                bb = flip_2spins(bbtmp,i5,i6)
                sb, expx, expy = find_representative(bb,Lx,Ly,Ns)
                b = find_state_2(sb,list_state,Nrep)
                if (b>=0)
                    elemnt[(Nbond1+2*Nbond2+i)*Nrep+a+1] += cxx*list_sqrtnorm[b+1]/list_sqrtnorm[a+1]*expk[expx+1,expy+1]
                    loc[(Nbond1+2*Nbond2+i)*Nrep+a+1] = b+1
                end
            end
        end
    end
    return elemnt, listki, loc
end

##
## ...- L+1 - L+2 - L+3 -...- 2L -...
##       |     |     |         |
## ...-  0  -  1  -  2  -...-  L -...
##
## - 0 - 2 -     - ix+L - ix+L+1 -
##   |   |   -->     |       |
## - 1 - 3 -     -  ix  -  ix+1  -
##
function make_lattice(Lx,Ly,Jleg,Jrung,Jising,J4)
    Jxx = Float64[]
    Jzz = Float64[]
    J4xx = Float64[]
    J4zz = Float64[]
    J4xz = Float64[]
    list_isite1 = Int[]
    list_isite2 = Int[]
    list_isite3 = Int[]
    list_isite4 = Int[]
    list_isite5 = Int[]
    list_isite6 = Int[]
    Nint = 0
    Nint2 = 0
    @inbounds for ix in 0:Lx-1
        site0 = ix+Lx
        site1 = ix
        site2 = (ix+1)%Lx+Lx
        site3 = (ix+1)%Lx
#
## 2-body terms
        push!(list_isite1,site0)
        push!(list_isite2,site2)
        push!(Jxx,Jleg)
        push!(Jzz,Jleg*Jising)
        Nint += 1
#
        push!(list_isite1,site1)
        push!(list_isite2,site3)
        push!(Jxx,Jleg)
        push!(Jzz,Jleg*Jising)
        Nint += 1
#
        push!(list_isite1,site0)
        push!(list_isite2,site1)
        push!(Jxx,Jrung)
        push!(Jzz,Jrung*Jising)
        Nint += 1
#
## 4-body terms
        push!(list_isite3,site0)
        push!(list_isite4,site2)
        push!(list_isite5,site1)
        push!(list_isite6,site3)
        push!(J4xx,J4)
        push!(J4zz,J4)
        push!(J4xz,J4)
        Nint2 += 1
#
    end
    return Jxx, Jzz, list_isite1, list_isite2, Nint,
        J4xx, J4zz, J4xz, list_isite3, list_isite4, list_isite5, list_isite6, Nint2
end

#function make_hamiltonian(Nrep,elemnt,listki,loc)
function make_hamiltonian(elemnt,listki,loc)
    return dropzeros(sparse(listki,loc,elemnt))
end


function main()
    parsed_args = parse_commandline()
    Lx = parsed_args["Lx"]
    Ly = parsed_args["Ly"]
    twoSz = parsed_args["twoSz"]
    Jleg = parsed_args["Jleg"]
    Jrung = parsed_args["Jrung"]
    Jising = parsed_args["Jising"]
    J4 = parsed_args["J4"]
    momkx = parsed_args["momkx"]
    momky = parsed_args["momky"]
    Ns = Lx*Ly
    nup = div(Ns+twoSz,2)
    Nbinom = binomial(Ns,nup)

    println("# make list")
    println("Lx=",Lx)
    println("Ly=",Ly)
    println("Jleg=",Jleg)
    println("Jrung=",Jrung)
    println("Jising=",Jising)
    println("J4=",J4)
    println("Ns=",Ns)
    println("twoSz=",twoSz)
    println("nup=",nup)
    println("Nbinom=",Nbinom)
    println("momkx=",momkx)
    println("momky=",momky)
    @time expk = calc_exp(Lx,Ly,momkx,momky)
#    println(expk)
#    println(expk[0+1,0+1])
#    println(expk[0+1,1+1])
#    println(expk[1+1,0+1])
#    println(expk[1+1,1+1])
#    println(expk[2+1,0+1])
#    println(expk[2+1,1+1])
#    println(expk[3+1,0+1])
#    println(expk[3+1,1+1])
    @time list_state, list_R, list_F2, Nrep = make_basis(Lx,Ly,Ns,nup,Nbinom,momkx,momky,expk)
#    list_state, list_R, Nrep = make_basis(Lx,Ly,Ns,nup,Nbinom,momkx,momky)
#    println(list_state)
#    println(list_R)
#    println(list_F2)
#    println(abs.(list_F2))
#    println(abs.(list_F2).>1e-12)
    mask = abs.(list_F2).>1e-12
#    println(mask)
    println("Nrep=",Nrep)
    println("# num of of excluded elements from |F|^2:",Nrep-sum(mask))
    Nrep = sum(mask)
    println("Nrep=",Nrep)
    list_state = list_state[mask]
    list_R = list_R[mask]
    list_F2 = list_F2[mask]
    list_sqrtnorm = sqrt.(list_R.*list_F2)
#    println(list_state)
#    println(list_R)
#    println(list_F2)
#    println(list_sqrtnorm)
    println()

    println("# make interactions")
    @time Jxx, Jzz, list_site1, list_site2, Nbond1,
        J4xx, J4zz, J4xz, list_site3, list_site4, list_site5, list_site6, Nbond2 =
        make_lattice(Lx,Ly,Jleg,Jrung,Jising,J4)
##
## make 2-body interactions for 1/4*(sigma^z.sigma^z + 2(S^+.S^- + S^-.S^+))
    Jxx = Jxx/4.0
    Jzz = Jzz/4.0
##
## make 4-body interactions for 1/16*(sigma^z.sigma^z + 2(S^+.S^- + S^-.S^+))^2
    J4xx = J4xx/16.0
    J4zz = J4zz/16.0
    J4xz = J4xz/16.0
##
    println("Jxx=",Jxx)
    println("Jzz=",Jzz)
    println("list_site1=",list_site1)
    println("list_site2=",list_site2)
    println("Nbond1=",Nbond1)
    println("J4xx=",J4xx)
    println("J4zz=",J4zz)
    println("J4xz=",J4xz)
    println("list_site3=",list_site3)
    println("list_site4=",list_site4)
    println("list_site5=",list_site5)
    println("list_site6=",list_site6)
    println("Nbond2=",Nbond2)
    println()

    println("# make Hamiltonian")
    @time elemnt, listki, loc =
        make_hamiltonian_child(Jxx,Jzz,list_site1,list_site2,Nbond1,
        J4xx,J4zz,J4xz,list_site3,list_site4,list_site5,list_site6,Nbond2,
        Nrep,list_state,list_sqrtnorm,Lx,Ly,Ns,momkx,momky,expk)
#    println(elemnt)
#    println(listki)
#    println(loc)
#    @time Ham = make_hamiltonian(Nrep,elemnt,listki,loc)
    @time Ham = make_hamiltonian(elemnt,listki,loc)
#    println(Ham)
    println()

    println("# diag Hamiltonian")
    Neig = 5
#    @time ene,vec = eigs(Ham,nev=min(Neig,Nrep-1),which=:SR)
    @time ene,vec = eigs(Ham,nev=min(Neig,Nrep-1),which=:SR,ritzvec=false)
    ene = sort(real(ene))/Ns
    println("# energy:",ene[1]," ",ene[2]," ",ene[3]," ",ene[4]," ",ene[5])
    println()

end


main()
