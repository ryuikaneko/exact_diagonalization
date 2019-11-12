#!/usr/bin/env python

# coding:utf-8
from __future__ import print_function
import math
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import argparse
import time
#
from numba import jit

def parse_args():
    parser = argparse.ArgumentParser(description='Calculate the ground state of S=1/2 Heisenberg ladder for a given Sz')
    parser.add_argument('-Lx',metavar='Lx',dest='Lx',type=int,default=8,help='set Lx')
    parser.add_argument('-Ly',metavar='Ly',dest='Ly',type=int,default=2,help='set Ly=2')
    parser.add_argument('-Jleg',metavar='Jleg',dest='Jleg',type=float,default=1.0,help='set Jleg')
    parser.add_argument('-Jrung',metavar='Jrung',dest='Jrung',type=float,default=1.0,help='set Jrung')
    parser.add_argument('-Jising',metavar='Jising',dest='Jising',type=float,default=1.0,help='set Jising (=1: Heisenberg)')
    parser.add_argument('-J4',metavar='J4',dest='J4',type=float,default=0.0,help='set J4')
#    parser.add_argument('-nup',metavar='nup',dest='nup',type=int,default=8,help='set nup')
    parser.add_argument('-twosz',metavar='twosz',dest='twosz',type=int,default=0,help='set twosz')
    parser.add_argument('-momkx',metavar='momkx',dest='momkx',type=int,default=0,help='set momkx')
    parser.add_argument('-momky',metavar='momky',dest='momky',type=int,default=0,help='set momky')
    return parser.parse_args()

def num2bit(state,L):
    return np.binary_repr(state,L)

def bit2num(bit):
    return int(bit,2)

def binomial(n,r):
    return math.factorial(n) // (math.factorial(n - r) * math.factorial(r))

@jit(nopython=True)
def get_next_same_nup_state(state):
    next = 0
    if(state>0):      
        smallest = state & -(state)
        ripple = state + smallest
        ones = state ^ ripple
        ones = (ones >> 2) // smallest
        next = ripple | ones
    return next

@jit(nopython=True)
def shift_child_BA2AB(s,a,b):
    return ((s<<b)&(((1<<a)-1)<<b))|((s>>a)&((1<<b)-1))

@jit(nopython=True)
def shift_child_CBA2CAB(s,a,b,c):
    left = s&(((1<<c)-1)<<(a+b))
    right = s&((1<<(a+b))-1)
    right = shift_child_BA2AB(right,a,b)
    return left|right

@jit(nopython=True)
def shift_child_DCBA2DBCA(s,a,b,c,d):
    left = s&(((1<<d)-1)<<(a+b+c))
    right = s&((1<<a)-1)
    mid = (s>>a)&((1<<(b+c))-1)
    mid = shift_child_BA2AB(mid,b,c)
    mid = mid<<a
    return left|(mid|right)

@jit(nopython=True)
def shift_y_1spin(state,Lx,Ly,Ns):
    return shift_child_BA2AB(state,Ns-Lx,Lx)

@jit(nopython=True)
def shift_x_1spin(state,Lx,Ly,Ns):
    s = shift_child_CBA2CAB(state,Lx-1,1,Ns-Lx)
    for n in range(1,Ly):
        s = shift_child_DCBA2DBCA(s,n*Lx,Lx-1,1,Ns-(n+1)*Lx)
    return s

@jit(nopython=True)
def check_state(state,Lx,Ly,Ns):
    list_allstate = []
    r = state
    t = state
    list_allstate.append(t)
    for i in range(Lx-1):
        t = shift_x_1spin(t,Lx,Ly,Ns)
        list_allstate.append(t)
        if (t < r):
            r = t
    for j in range(Ly-1):
        t = shift_y_1spin(t,Lx,Ly,Ns)
        for i in range(Lx):
            t = shift_x_1spin(t,Lx,Ly,Ns)
            list_allstate.append(t)
            if (t < r):
                r = t
    D = len(set(list_allstate)) ## remove duplication by "set" and count length of a list
#    print(D,list_allstate)
    if r == state:
        return +D
    else:
        return -D

@jit(nopython=True)
def calc_sum_exp(r,Lx,Ly,Ns,expk):
#    list_expx = []
#    list_expy = []
    t = r
#    list_expx.append(0)
#    list_expy.append(0)
#    Nlist = 1
    F2 = expk[0,0]
    for i in range(Lx-1):
        t = shift_x_1spin(t,Lx,Ly,Ns)
        if (t == r):
#            list_expx.append(i+1)
#            list_expy.append(0)
#            Nlist += 1
            F2 += expk[i+1,0]
    for j in range(Ly-1):
        t = shift_y_1spin(t,Lx,Ly,Ns)
        for i in range(Lx):
            t = shift_x_1spin(t,Lx,Ly,Ns)
            if (t == r):
#                list_expx.append(i)
#                list_expy.append(j+1)
#                Nlist += 1
                F2 += expk[i,j+1]
#    return list_expx, list_expy, Nlist
    return F2

@jit(nopython=True)
def find_representative(state,Lx,Ly,Ns):
    r = state
    t = state
    expx = 0
    expy = 0
    for i in range(Lx-1):
        t = shift_x_1spin(t,Lx,Ly,Ns)
        if (t < r):
            r = t
            expx = i+1
    for j in range(Ly-1):
        t = shift_y_1spin(t,Lx,Ly,Ns)
        for i in range(Lx):
            t = shift_x_1spin(t,Lx,Ly,Ns)
            if (t < r):
                r = t
                expx = i
                expy = j+1
    return r, expx, expy

#def make_basis(Lx,Ly,Ns,nup,Nbinom,momkx,momky):
@jit(nopython=True)
def make_basis(Lx,Ly,Ns,nup,Nbinom,momkx,momky,expk):
    list_state = []
    list_R = []
    list_F2 = []
    first = (1<<(Ns-nup))-1
    last = ((1<<(Ns-nup))-1)<<(nup)
#    print("# first:",first,num2bit(first,Ns))
#    print("# last:",last,num2bit(last,Ns))
    Nrep = 0
    state = first
    for i in range(Nbinom):
        R = check_state(state,Lx,Ly,Ns)
        if (R>=0):
            list_state.append(state)
            list_R.append(R)
            Nrep += 1
#            list_expx, list_expy, Nlist = calc_sum_exp(state,Lx,Ly,Ns)
            F2 = calc_sum_exp(state,Lx,Ly,Ns,expk)
#            F2 = np.abs(np.sum(np.array(\
#                [ np.exp(-1j*2.0*np.pi*(float(xi*momkx)/Lx+float(yi*momky)/Ly)) for xi,yi in zip(list_expx,list_expy) ]\
#                ),dtype=np.complex128))**2
#            F2 = abs(sum([ np.exp(-1j*2.0*np.pi*(float(xi*momkx)/Lx+float(yi*momky)/Ly)) for xi,yi in zip(list_expx,list_expy) ]))**2
#            F2 = np.abs(np.sum(np.array([ expk[xi,yi] for xi,yi in zip(list_expx,list_expy) ],dtype=np.complex128)))**2
#
#            F2tmp = [expk[xi,yi] for xi,yi in zip(list_expx,list_expy)]
#            F2 = np.abs(np.sum(np.array(F2tmp,dtype=np.complex128)))**2
#
#            F2 = np.abs(np.sum(np.array([ expk[list_expx[i],list_expy[i]] for i in range(Nlist) ],dtype=np.complex128)))**2
#
            list_F2.append(F2)
#            print(state,list_expx,list_expy,F2)
        state = get_next_same_nup_state(state)
    return list_state, list_R, list_F2, Nrep

@jit(nopython=True)
def get_spin(state,site):
    return (state>>site)&1

@jit(nopython=True)
def flip_2spins(state,i1,i2):
    return state^((1<<i1)+(1<<i2))

@jit(nopython=True)
def find_state_2(state,list_1,maxind):
    imin = 0
    imax = maxind-1
    while True:
        i = (imin+imax)//2
#        print(i,imin,imax,maxind,state,list_1[i])
        if (state < list_1[i]):
            imax = i-1
        elif (state > list_1[i]):
            imin = i+1
        else:
            break
        if (imin > imax):
            return -1
    return i

@jit(nopython=True)
def calc_exp(Lx,Ly,momkx,momky):
    return np.array([[np.exp(-1j*2.0*np.pi*(float(expx*momkx)/Lx+float(expy*momky)/Ly)) \
        for expy in range(Ly)] for expx in range(Lx)])

#def make_hamiltonian_child(Jxx,Jzz,Nbond,list_site1,list_site2,Nrep,list_state,list_sqrtnorm,Lx,Ly,Ns,momkx,momky,expk):
@jit(nopython=True)
def make_hamiltonian_child(Jxx,Jzz,list_site1,list_site2,Nbond1,\
    J4xx,J4zz,J4xz,list_site3,list_site4,list_site5,list_site6,Nbond2,\
    Nrep,list_state,list_sqrtnorm,Lx,Ly,Ns,momkx,momky,expk):
    Nbond = Nbond1 + 3*Nbond2
    listki = np.array([i for k in range(Nbond+1) for i in range(Nrep)],dtype=np.int64)
    loc = np.zeros((Nbond+1)*Nrep,dtype=np.int64)
    elemnt = np.zeros((Nbond+1)*Nrep,dtype=np.complex128)
##
## index of loc[]
##   Nbond*Nrep+a: diag
##   i*Nrep+a: offdiag such as [s2,s1] = 01, 10
##   (Nbond1+i)*Nrep+a: offdiag such as [s6,s5,s4,s3] = 0100, 1000, 0111, 1011
##   (Nbond1+Nbond2+i)*Nrep+a: offdiag such as [s6,s5,s4,s3] = 0001, 0010, 1101, 1110
##   (Nbond1+2*Nbond2+i)*Nrep+a: offdiag such as [s6,s5,s4,s3] = 0101, 0110, 1001, 1010
##
    for a in range(Nrep):
        sa = list_state[a]
        loc[Nbond*Nrep+a] = a
        for i in range(Nbond1):
            i1 = list_site1[i]
            i2 = list_site2[i]
            wght = 2.0*Jxx[i]
            diag = Jzz[i]
#            loc[Nbond*Nrep+a] = a
            if get_spin(sa,i1) == get_spin(sa,i2):
                elemnt[Nbond*Nrep+a] += diag
            else:
                elemnt[Nbond*Nrep+a] -= diag
                bb = flip_2spins(sa,i1,i2)
                sb, expx, expy = find_representative(bb,Lx,Ly,Ns)
                b = find_state_2(sb,list_state,Nrep)
                if b>=0:
                    elemnt[i*Nrep+a] += wght*list_sqrtnorm[b]/list_sqrtnorm[a]*expk[expx,expy]
                    loc[i*Nrep+a] = b
        for i in range(Nbond2):
            i3 = list_site3[i]
            i4 = list_site4[i]
            i5 = list_site5[i]
            i6 = list_site6[i]
            czz = J4zz[i]
            cxz = 2.0*J4xz[i]
            cxx = 4.0*J4xx[i]
#            loc[Nbond*Nrep+a] = a
            if get_spin(sa,i3) == get_spin(sa,i4) and get_spin(sa,i5) == get_spin(sa,i6):
                elemnt[Nbond*Nrep+a] += czz
            elif get_spin(sa,i3) == get_spin(sa,i4) and get_spin(sa,i5) != get_spin(sa,i6):
                elemnt[Nbond*Nrep+a] -= czz
#
                bb = flip_2spins(sa,i5,i6)
                sb, expx, expy = find_representative(bb,Lx,Ly,Ns)
                b = find_state_2(sb,list_state,Nrep)
                if b>=0:
                    elemnt[(Nbond1+i)*Nrep+a] += cxz*list_sqrtnorm[b]/list_sqrtnorm[a]*expk[expx,expy]
                    loc[(Nbond1+i)*Nrep+a] = b
            elif get_spin(sa,i3) != get_spin(sa,i4) and get_spin(sa,i5) == get_spin(sa,i6):
                elemnt[Nbond*Nrep+a] -= czz
#
                bb = flip_2spins(sa,i3,i4)
                sb, expx, expy = find_representative(bb,Lx,Ly,Ns)
                b = find_state_2(sb,list_state,Nrep)
                if b>=0:
                    elemnt[(Nbond1+Nbond2+i)*Nrep+a] += cxz*list_sqrtnorm[b]/list_sqrtnorm[a]*expk[expx,expy]
                    loc[(Nbond1+Nbond2+i)*Nrep+a] = b
            elif get_spin(sa,i3) != get_spin(sa,i4) and get_spin(sa,i5) != get_spin(sa,i6):
                elemnt[Nbond*Nrep+a] += czz
#
                bb = flip_2spins(sa,i3,i4)
                sb, expx, expy = find_representative(bb,Lx,Ly,Ns)
                b = find_state_2(sb,list_state,Nrep)
                if b>=0:
                    elemnt[(Nbond1+Nbond2+i)*Nrep+a] += -cxz*list_sqrtnorm[b]/list_sqrtnorm[a]*expk[expx,expy]
                    loc[(Nbond1+Nbond2+i)*Nrep+a] = b
#
                bb = flip_2spins(sa,i5,i6)
                sb, expx, expy = find_representative(bb,Lx,Ly,Ns)
                b = find_state_2(sb,list_state,Nrep)
                if b>=0:
                    elemnt[(Nbond1+i)*Nrep+a] += -cxz*list_sqrtnorm[b]/list_sqrtnorm[a]*expk[expx,expy]
                    loc[(Nbond1+i)*Nrep+a] = b
#
                bbtmp = flip_2spins(sa,i3,i4)
                bb = flip_2spins(bbtmp,i5,i6)
                sb, expx, expy = find_representative(bb,Lx,Ly,Ns)
                b = find_state_2(sb,list_state,Nrep)
                if b>=0:
                    elemnt[(Nbond1+2*Nbond2+i)*Nrep+a] += cxx*list_sqrtnorm[b]/list_sqrtnorm[a]*expk[expx,expy]
                    loc[(Nbond1+2*Nbond2+i)*Nrep+a] = b
    return elemnt, listki, loc

def make_hamiltonian(Nrep,elemnt,listki,loc):
    return scipy.sparse.csr_matrix((elemnt,(listki,loc)),shape=(Nrep,Nrep),dtype=np.complex128)

##
## ...- L+1 - L+2 - L+3 -...- 2L -...
##       |     |     |         |
## ...-  0  -  1  -  2  -...-  L -...
##
## - 0 - 2 -     - ix+L - ix+L+1 -
##   |   |   -->     |       |
## - 1 - 3 -     -  ix  -  ix+1  -
##
@jit(nopython=True)
def make_lattice(Lx,Ly,Jleg,Jrung,Jising,J4):
    Jxx = []
    Jzz = []
    J4xx = []
    J4zz = []
    J4xz = []
    list_isite1 = []
    list_isite2 = []
    list_isite3 = []
    list_isite4 = []
    list_isite5 = []
    list_isite6 = []
    Nint = 0
    Nint2 = 0
    for ix in range(Lx):
        site0 = ix+Lx
        site1 = ix
        site2 = (ix+1)%Lx+Lx
        site3 = (ix+1)%Lx
#
## 2-body terms
        list_isite1.append(site0)
        list_isite2.append(site2)
        Jxx.append(Jleg)
        Jzz.append(Jleg*Jising)
        Nint += 1
#
        list_isite1.append(site1)
        list_isite2.append(site3)
        Jxx.append(Jleg)
        Jzz.append(Jleg*Jising)
        Nint += 1
#
        list_isite1.append(site0)
        list_isite2.append(site1)
        Jxx.append(Jrung)
        Jzz.append(Jrung*Jising)
        Nint += 1
#
## 4-body terms
        list_isite3.append(site0)
        list_isite4.append(site2)
        list_isite5.append(site1)
        list_isite6.append(site3)
        J4xx.append(J4)
        J4zz.append(J4)
        J4xz.append(J4)
        Nint2 += 1
#
#    return Jxx, Jzz, list_isite1, list_isite2, Nint
    return Jxx, Jzz, list_isite1, list_isite2, Nint, \
        J4xx, J4zz, J4xz, list_isite3, list_isite4, list_isite5, list_isite6, Nint2


def main():
    args = parse_args()
    Lx = args.Lx
    Ly = args.Ly
    Ns = Lx*Ly
    Jleg = args.Jleg
    Jrung = args.Jrung
    Jising = args.Jising
    J4 = args.J4
#    nup = args.nup
    twosz = args.twosz
    momkx = args.momkx
    momky = args.momky
    nup = (Ns + twosz)//2
    Nbinom = binomial(Ns,nup)

    print("# make list")
    start = time.time()
    print("Lx=",Lx)
    print("Ly=",Ly)
    print("Jleg=",Jleg)
    print("Jrung=",Jrung)
    print("Jising=",Jising)
    print("J4=",J4)
    print("Ns=",Ns)
    print("twoSz=",twosz)
    print("Nup=",nup)
    print("momkx=",momkx)
    print("momky=",momky)
    end = time.time()
    expk = calc_exp(Lx,Ly,momkx,momky)
#    list_state, list_R, list_F2, Nrep = make_basis(Lx,Ly,Ns,nup,Nbinom,momkx,momky)
    list_state, list_R, list_F2, Nrep = make_basis(Lx,Ly,Ns,nup,Nbinom,momkx,momky,expk)
    mask = abs(np.array(list_F2)) > 1e-16 # exclude |F|^2==0
#    print(mask)
    print("# num of of excluded elements from |F|^2:",Nrep-np.sum(mask))
    Nrep = np.sum(mask)
    list_state = np.array(list_state)[mask]
    list_R = np.array(list_R)[mask]
    list_F2 = np.array(list_F2)[mask]
    list_sqrtnorm = np.sqrt(list_R*list_F2)
    print("Nrep=",Nrep)
#    print("# ind state_num state_bit R=num_diff_states |F|^2 sqrt(Na)")
#    for i in range(Nrep):
#        print(i,list_state[i],num2bit(list_state[i],Ns),list_R[i],list_F2[i],list_sqrtnorm[i])
    end = time.time()
    print("# time:",end-start)
    print()

    print("# make interactions")
    start = time.time()
#    Jxx, Jzz, list_site1, list_site2, Nbond = make_lattice(Lx,Ly,Jleg,Jrung,Jising)
    Jxx, Jzz, list_site1, list_site2, Nbond1, \
        J4xx, J4zz, J4xz, list_site3, list_site4, list_site5, list_site6, Nbond2 \
        = make_lattice(Lx,Ly,Jleg,Jrung,Jising,J4)
##
## make 2-body interactions for 1/4*(sigma^z.sigma^z + 2(S^+.S^- + S^-.S^+))
    Jxx = np.array(Jxx)/4.0
    Jzz = np.array(Jzz)/4.0
    list_site1 = np.array(list_site1)
    list_site2 = np.array(list_site2)
##
## make 4-body interactions for 1/16*(sigma^z.sigma^z + 2(S^+.S^- + S^-.S^+))^2
    J4xx = np.array(J4xx)/16.0
    J4zz = np.array(J4zz)/16.0
    J4xz = np.array(J4xz)/16.0
    list_site3 = np.array(list_site3)
    list_site4 = np.array(list_site4)
    list_site5 = np.array(list_site5)
    list_site6 = np.array(list_site6)
##
    print("Jxx=",Jxx)
    print("Jzz=",Jzz)
    print("list_site1=",list_site1)
    print("list_site2=",list_site2)
    print("Nbond1=",Nbond1)
    print("J4xx=",J4xx)
    print("J4zz=",J4zz)
    print("J4xz=",J4xz)
    print("list_site3=",list_site3)
    print("list_site4=",list_site4)
    print("list_site5=",list_site5)
    print("list_site6=",list_site6)
    print("Nbond2=",Nbond2)
    end = time.time()
    print("# time:",end-start)
    print()

    print("# make Hamiltonian")
    start = time.time()
#    expk = calc_exp(Lx,Ly,momkx,momky)
    elemnt, listki, loc = \
        make_hamiltonian_child(Jxx,Jzz,list_site1,list_site2,Nbond1,\
        J4xx,J4zz,J4xz,list_site3,list_site4,list_site5,list_site6,Nbond2,\
        Nrep,list_state,list_sqrtnorm,Lx,Ly,Ns,momkx,momky,expk)
    Ham = make_hamiltonian(Nrep,elemnt,listki,loc)
#    print(Ham)
    end = time.time()
    print("# time:",end-start)
    print()

    print("# diag Hamiltonian")
    start = time.time()
    Neig = 5
#    ene,vec = scipy.linalg.eigh(Ham.todense(),eigvals=(0,min(Neig,Nrep-1)))
#    ene,vec = scipy.linalg.eigh(Ham,eigvals=(0,min(Neig,Nrep-1)))
    ene,vec = scipy.sparse.linalg.eigsh(Ham,which='SA',k=min(Neig,Nrep-1))
#    ene = ene/Ns/4
    ene = ene/Ns
    end = time.time()
#    print ("energy:",ene)
    print("# energy:",ene[0],ene[1],ene[2],ene[3],ene[4])
    print("# time:",end-start)
    print()

if __name__ == "__main__":
    main()
