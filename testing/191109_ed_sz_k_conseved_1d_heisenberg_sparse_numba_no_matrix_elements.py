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
#import os
#import sys
#
from numba import jit

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-L',metavar='L',dest='L',type=int,default=16,help='set L')
#    parser.add_argument('-nup',metavar='nup',dest='nup',type=int,default=8,help='set nup')
    parser.add_argument('-twosz',metavar='twosz',dest='twosz',type=int,default=0,help='set twosz')
    parser.add_argument('-momk',metavar='momk',dest='momk',type=int,default=0,help='set momk')
    return parser.parse_args()

def num2bit(state,L):
    return np.binary_repr(state,L)

## https://stackoverflow.com/questions/8928240/convert-base-2-binary-number-string-to-int
def bit2num(bit):
    return int(bit,2)

## http://lptms.u-psud.fr/membres/groux/Test/ED/ED_Lecture1.html
def show_state01(state,L): # show spins from left to right
    return "|"+"".join([i for i in num2bit(state,L)[::-1]])+">" # use 0,1 with ket
#    return "".join([i for i in num2bit(state,L)[::-1]]) # use 0,1

def show_state(state,L): # show spins from left to right
    return "|"+"".join([ str('+') if i==str(0) else str('-') for i in num2bit(state,L)[::-1]])+">" # use +,- with ket
#    return "".join([ str('+') if i==str(0) else str('-') for i in num2bit(state,L)[::-1]]) # use +,-

## https://github.com/alexwie/ed_basics/blob/master/hamiltonian_hb_staggered.py
@jit(nopython=True)
def get_spin(state,site):
    return (state>>site)&1

## http://lptms.u-psud.fr/membres/groux/Test/ED/ED_Lecture1.html
def get_spin_alternative(state,site):
    return (state&(1<<site))>>site

## http://tccm.pks.mpg.de/?page_id=871
## https://www.pks.mpg.de/~frankp/comp-phys/
## https://www.pks.mpg.de/~frankp/comp-phys/exact_diagonalization_conserve.py
##
## http://lptms.u-psud.fr/membres/groux/Test/ED/ED_Lecture2.html
#def shift_1spin(state,L):
#    bs = bin(state)[2:].zfill(L)
#    return int(bs[1:]+bs[0],2)

@jit(nopython=True)
def shift_1spin(state,L):
    return ((state<<1)&(1<<L)-2)|((state>>(L-1))&1)

## http://lptms.u-psud.fr/membres/groux/Test/ED/ED_Lecture2.html
#def shift_1spin_inv(state,L):
#    first = get_spin(state,0)
#    return (state>>1)|(first<<(L-1))

@jit(nopython=True)
def shift_1spin_inv(state,L):
    return ((state<<(L-1))&(1<<(L-1)))|((state>>1)&((1<<(L-1))-1))

def shift_spin(state,L,shift):
    n2b = num2bit(state,L)
    bit = n2b[shift:]+n2b[0:shift]
#    print(n2b,"=",n2b[0:shift],"+",n2b[shift:],"-->",bit)
    return bit2num(bit)

## https://en.wikipedia.org/wiki/Circular_shift
## https://stackoverflow.com/questions/6223137/verifying-ctypes-type-precision-in-python
## http://lptms.u-psud.fr/membres/groux/Test/ED/ED_Lecture2.html
#
#def shift_spins_inv(state,shift,L):
#    CHAR_BIT = os.sysconf('SC_CHAR_BIT')
#    print(CHAR_BIT)
#    mask = CHAR_BIT * sys.getsizeof(state) - 1
#    shift &= mask
#    return (state>>shift)|(state<<(-shift&mask))
##
## https://en.wikipedia.org/wiki/Bitwise_operation#In_high-level_languages
#def shift_spin(state,shift):
#    return (state<<shift)|(state>>(-shift&31))

## https://qiita.com/phdax/items/3064de264c7933bab2f5
## https://web.archive.org/web/20190108235115/https://www.hackersdelight.org/hdcodetxt/pop.c.txt
## http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
## https://stackoverflow.com/questions/9829578/fast-way-of-counting-non-zero-bits-in-positive-integer
def count_upspins(state):
    count = state
## 32 bits
    count = (count & 0x55555555) + ((count >> 1) & 0x55555555)
    count = (count & 0x33333333) + ((count >> 2) & 0x33333333)
    count = (count & 0x0F0F0F0F) + ((count >> 4) & 0x0F0F0F0F)
    count = (count & 0x00FF00FF) + ((count >> 8) & 0x00FF00FF)
    count = (count & 0x0000FFFF) + ((count >>16) & 0x0000FFFF)
## 64 bits
#    count = (count & 0x5555555555555555) + ((count & 0xAAAAAAAAAAAAAAAA) >> 1)
#    count = (count & 0x3333333333333333) + ((count & 0xCCCCCCCCCCCCCCCC) >> 2)
#    count = (count & 0x0F0F0F0F0F0F0F0F) + ((count & 0xF0F0F0F0F0F0F0F0) >> 4)
#    count = (count & 0x00FF00FF00FF00FF) + ((count & 0xFF00FF00FF00FF00) >> 8)
#    count = (count & 0x0000FFFF0000FFFF) + ((count & 0xFFFF0000FFFF0000) >> 16)
#    count = (count & 0x00000000FFFFFFFF) + ((count & 0xFFFFFFFF00000000) >> 32)
    return count

def count_upspins_alternative(state):
    count = 0
    while(state):
        count += state & 1
        state >>= 1
    return count

def binomial(n,r):
    return math.factorial(n) // (math.factorial(n - r) * math.factorial(r))

## https://web.archive.org/web/20190109000030/http://www.hackersdelight.org/hdcodetxt/snoob.c.txt
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

## http://graphics.stanford.edu/~seander/bithacks.html#NextBitPermutation
## https://github.com/alexwie/ed_basics/blob/master/hamiltonian_hb_staggered.py
def get_next_same_nup_state_alternative(state):
    next = 0
    if(state>0):
        t = (state | (state - 1)) + 1
        next = t | ((((t & -t) // (state & -state)) >> 1) - 1)
    return next

def init_parameters(L,nup):
    Nhilbert = binomial(L,nup)
    ihfbit = 1 << (L//2)
    irght = ihfbit-1
    ilft = ((1<<L)-1) ^ irght
    iup = (1<<(L-nup))-1
    return Nhilbert, ihfbit, irght, ilft, iup

def make_list_same_nup(Nhilbert,ihfbit,irght,ilft,iup):
    list_1 = np.zeros(Nhilbert,dtype=int)
    list_ja = np.zeros(ihfbit,dtype=int)
    list_jb = np.zeros(ihfbit,dtype=int)
    ii = iup
    ja = 0
    jb = 0
    ia_old = ii & irght
    ib_old = (ii & ilft) // ihfbit
    list_1[0] = ii
    list_ja[ia_old] = ja
    list_jb[ib_old] = jb
    ii = get_next_same_nup_state(ii)
    for i in range(1,Nhilbert):
        ia = ii & irght
        ib = (ii & ilft) // ihfbit
        if (ib == ib_old):
            ja += 1
        else:
            jb += ja+1
            ja = 0
        list_1[i] = ii
        list_ja[ia] = ja
        list_jb[ib] = jb
        ia_old = ia
        ib_old = ib
        ii = get_next_same_nup_state(ii)
    return list_1, list_ja, list_jb

def get_ja_plus_jb(ii,irght,ilft,ihfbit,list_ja,list_jb):
    ia = ii & irght
    ib = (ii & ilft) // ihfbit
    ja = list_ja[ia]
    jb = list_jb[ib]
    return ja+jb

## http://physics.bu.edu/~sandvik/vietri/dia.pdf
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
def check_state(state,nup,momk,L):
#    R = -1
#    if (count_upspins(state) != nup): return R ## fixed sz
    t = state
    for i in range(L):
        t = shift_1spin(t,L)
        if (t < state):
#            return R
            return -1
        elif (t == state):
            if (np.mod(momk,L//(i+1)) != 0):
#                return R
                return -1
            else:
                return i+1

@jit(nopython=True)
def find_representative(state,L):
    rep = state
    tmp = state
    exponent = 0
    for i in range(L):
        tmp = shift_1spin(tmp,L)
        if (tmp < rep):
            rep = tmp
            exponent = i+1
    return rep, exponent

@jit(nopython=True)
def flip_2spins(state,i1,i2):
    return state^((1<<i1)+(1<<i2))
#    return state^(2**i1+2**i2)

@jit(nopython=True)
def make_basis(L,nup,momk,Nbinom):
    list_state = []
    list_R = []
#    list_sqrtR = []
    first = (1<<(L-nup))-1
    last = ((1<<(L-nup))-1)<<(nup)
#    print("# first:",first,num2bit(first,L))
#    print("# last:",last,num2bit(last,L))
    Nrep = 0
    state = first
    for i in range(Nbinom):
        R = check_state(state,nup,momk,L)
        if (R>=0):
            list_state.append(state)
            list_R.append(R)
#            list_sqrtR.append(np.sqrt(R))
            Nrep += 1
        state = get_next_same_nup_state(state)
#    return list_state, list_R, Nrep
#    return list_state, list_sqrtR, Nrep
#    return np.array(list_state,dtype=np.int64), np.array(list_R,dtype=np.float64), Nrep
#    return np.array(list_state,dtype=np.int64), np.array(list_sqrtR,dtype=np.float64), Nrep
    return list_state, list_R, Nrep

def calc_exp(L,momk):
    return np.array([np.exp(-1j*exponent*2.0*np.pi*momk/L) for exponent in range(L)])

#def make_hamiltonian(Nbond,list_site1,list_site2,Nrep,list_state,list_R,L,momk):
#def make_hamiltonian(Nbond,list_site1,list_site2,Nrep,list_state,list_sqrtR,L,momk):
#def make_hamiltonian(Nbond,list_site1,list_site2,Nrep,list_state,list_sqrtR,L,momk,expk):
#@jit(nopython=True)
#def make_hamiltonian_child(Nbond,list_site1,list_site2,Nrep,list_state,list_sqrtR,L,momk,expk):
def make_hamiltonian(Nbond,list_site1,list_site2,Nrep,list_state,list_sqrtR,L,momk,expk):
#    listki = np.array([i for k in range(Nbond+1) for i in range(Nrep)],dtype=np.int64)
#    loc = np.zeros((Nbond+1)*Nrep,dtype=np.int64)
#    elemnt = np.zeros((Nbond+1)*Nrep,dtype=np.complex128)
#    Ham = np.zeros((Nrep,Nrep),dtype=complex)
    @jit(nopython=True)
    def get_vec(vec):
        vecnew = np.zeros(Nrep,dtype=np.complex128)
        for a in range(Nrep):
            sa = list_state[a]
            for i in range(Nbond):
                i1 = list_site1[i]
                i2 = list_site2[i]
#                loc[Nbond*Nrep+a] = a
                if get_spin(sa,i1) == get_spin(sa,i2):
#                    Ham[a,a] += 0.25
#                    elemnt[Nbond*Nrep+a] += 0.25
                    vecnew[a] += +0.25*vec[a]
                else:
#                    Ham[a,a] -= 0.25
#                    elemnt[Nbond*Nrep+a] -= 0.25
                    vecnew[a] += -0.25*vec[a]
                    bb = flip_2spins(sa,i1,i2)
                    sb, exponent = find_representative(bb,L)
                    b = find_state_2(sb,list_state,Nrep)
                    if b>=0:
#                        Ham[a,b] += 0.5*np.sqrt(float(list_R[a])/float(list_R[b]))*np.exp(-1j*exponent*2.0*np.pi*momk/L)
#                        Ham[a,b] += 0.5*list_sqrtR[a]/list_sqrtR[b]*np.exp(-1j*exponent*2.0*np.pi*momk/L)
#                        Ham[a,b] += 0.5*list_sqrtR[a]/list_sqrtR[b]*expk[exponent]
#                        elemnt[i*Nrep+a] += 0.5*list_sqrtR[a]/list_sqrtR[b]*expk[exponent]
#                        loc[i*Nrep+a] = b
                        vecnew[a] += vec[b]*0.5*list_sqrtR[a]/list_sqrtR[b]*expk[exponent]
## https://stackoverflow.com/questions/19420171/sparse-matrix-in-numba
## Unknown attribute 'csr_matrix' of type Module
#    Ham = scipy.sparse.csr_matrix((elemnt,(listki,loc)),shape=(Nrep,Nrep),dtype=np.complex128)
#    return Ham
#    return elemnt, listki, loc
        vec = vecnew[:]
        return vec
    return get_vec

#def make_hamiltonian(Nrep,elemnt,listki,loc):
#    return scipy.sparse.csr_matrix((elemnt,(listki,loc)),shape=(Nrep,Nrep),dtype=np.complex128)


def main():
    args = parse_args()
    L = args.L
#    nup = args.nup
    twosz = args.twosz
    momk = args.momk
    nup = (L + twosz)//2
    Nbinom = binomial(L,nup)

    start = time.time()
    print("# make basis: sector of twosz=",twosz)
#    list_state, list_R, Nrep = make_basis(L,nup,momk)
#    list_state, list_sqrtR, Nrep = make_basis(L,nup,momk)
#    list_state, list_sqrtR, Nrep = make_basis(L,nup,momk,Nbinom)
    list_state, list_R, Nrep = make_basis(L,nup,momk,Nbinom)
    list_state = np.array(list_state,dtype=np.int64)
    list_R = np.array(list_R,dtype=np.int64)
    list_sqrtR = np.sqrt(list_R)
    print("# L=",L,", nup=",nup,"twosz =",twosz,", momk=",momk,", Nrep=",Nrep)
    print("# show first and last bases")
#    print("# ind state_num state_bit period_R")
#    print("# ind state_num state_bit period_sqrtR")
    print("# ind state_num state_bit period_R period_sqrtR")
#    for i in range(Nrep):
    for i in range(0,Nrep,Nrep-1):
#        print(i,list_state[i],num2bit(list_state[i],L),list_R[i])
#        print(i,list_state[i],num2bit(list_state[i],L),list_sqrtR[i])
        print(i,list_state[i],num2bit(list_state[i],L),list_R[i],list_sqrtR[i])
    end = time.time()
    print("# time:",end-start)
    print()

    start = time.time()
    end = time.time()
    print("# make interactions")
    Nbond = L
#    list_site1 = [i for i in range(Nbond)]
#    list_site2 = [(i+1)%L for i in range(Nbond)]
    list_site1 = np.array([i for i in range(Nbond)],dtype=np.int64)
    list_site2 = np.array([(i+1)%L for i in range(Nbond)],dtype=np.int64)
    print("list_site1=",list_site1)
    print("list_site2=",list_site2)
    end = time.time()
    print("# time:",end-start)
    print()

    start = time.time()
    end = time.time()
    print("# make Hamiltonian")
    expk = calc_exp(L,momk)
#    Ham = make_hamiltonian(Nbond,list_site1,list_site2,Nrep,list_state,list_R,L,momk)
#    Ham = make_hamiltonian(Nbond,list_site1,list_site2,Nrep,list_state,list_sqrtR,L,momk)
#    Ham = make_hamiltonian(Nbond,list_site1,list_site2,Nrep,list_state,list_sqrtR,L,momk,expk)
#    elemnt, listki, loc = make_hamiltonian_child(Nbond,list_site1,list_site2,Nrep,list_state,list_sqrtR,L,momk,expk)
#    Ham = make_hamiltonian(Nrep,elemnt,listki,loc)
    get_vec = make_hamiltonian(Nbond,list_site1,list_site2,Nrep,list_state,list_sqrtR,L,momk,expk)
    Ham = scipy.sparse.linalg.LinearOperator((Nrep,Nrep),matvec=get_vec)
#    print(Ham)
    end = time.time()
    print("# time:",end-start)
    print()

    start = time.time()
    end = time.time()
    print("# diag Hamiltonian")
    Neig = 5
#    ene,vec = scipy.linalg.eigh(Ham,eigvals=(0,min(Neig,Nrep-1)))
#    ene,vec = scipy.sparse.linalg.eigsh(Ham,which='SA',k=min(Neig,Nrep-1))
    ene = scipy.sparse.linalg.eigsh(Ham,which='SA',k=min(Neig,Nrep-1),return_eigenvectors=False)
    ene4 = 4.0*ene
    ene = np.sort(ene4)
    end = time.time()
#    print ("energy:",ene)
#    print ("energy:",4.0*ene)
#    print ("energy:",ene[0],ene[1],ene[2],ene[3],ene[4])
    print ("L k energy:",L,momk,ene[0],ene[1],ene[2],ene[3],ene[4])
    print("# time:",end-start)
    print()

if __name__ == "__main__":
    main()
