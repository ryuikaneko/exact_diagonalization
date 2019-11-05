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

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-L',metavar='L',dest='L',type=int,default=6,help='set L')
    parser.add_argument('-nup',metavar='nup',dest='nup',type=int,default=3,help='set nup')
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
def shift_1spin(state,L):
    bs = bin(state)[2:].zfill(L)
    return int(bs[1:]+bs[0],2)

## http://lptms.u-psud.fr/membres/groux/Test/ED/ED_Lecture2.html
def shift_1spin_inv(state,L):
    first = get_spin(state,0)
    return (state>>1)|(first<<(L-1))

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

def check_state(state,nup,k,L):
    R = -1
#    if (count_upspins(state) != nup): return R ## fixed sz
    t = state
    for i in range(L):
        t = shift_1spin(t,L)
        if (t < state):
            return R
        elif (t == state):
            if (np.mod(k,L//(i+1)) != 0):
                return R
            else:
                return i+1

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

def flip_2spins(state,i1,i2):
    return state^((1<<i1)+(1<<i2))
#    return state^(2**i1+2**i2)


def main():
    args = parse_args()
    L = args.L
    nup = args.nup
#    nup = L//2

    print("# check state, make basis")
    list_state = [[] for i in range(L)]
    list_R = [[] for i in range(L)]
    Nrep = np.zeros(L,dtype=int)
    for k in range(L):
        Nrep[k] = 0
        for state in range(2**L):
            if (count_upspins(state) == nup): ## fixed sz
                R = check_state(state,nup,k,L)
                if (R>=0):
                    list_state[k].append(state)
                    list_R[k].append(R)
                    Nrep[k] += 1
    print(list_state)
    print(list_R)
    for k in range(len(list_state)):
        print("#")
        print("# L=",L,", nup=",nup,", k=",k,", Nrep=",Nrep[k])
        print("# int_momentum_k i state bit period_R")
        for i in range(len(list_state[k])):
            state = list_state[k][i]
            R = list_R[k][i]
            print(k,i,state,num2bit(state,L),R)
    print()

    print("# check state, make basis (faster)")
    list_state = [[] for i in range(L)]
    list_R = [[] for i in range(L)]
    Nrep = np.zeros(L,dtype=int)
    first = (1<<(L-nup))-1
    last = ((1<<(L-nup))-1)<<(nup)
    print("# first:",first,num2bit(first,L))
    print("# last:",last,num2bit(last,L))
    for k in range(L):
        Nrep[k] = 0
        state = first
        for i in range(binomial(L,nup)):
            R = check_state(state,nup,k,L)
            if (R>=0):
                list_state[k].append(state)
                list_R[k].append(R)
                Nrep[k] += 1
            state = get_next_same_nup_state(state)
    print(list_state)
    print(list_R)
    for k in range(len(list_state)):
        print("#")
        print("# L=",L,", nup=",nup,", k=",k,", Nrep=",Nrep[k])
        print("# int_momentum_k i state bit period_R")
        for i in range(len(list_state[k])):
            state = list_state[k][i]
            R = list_R[k][i]
            print(k,i,state,num2bit(state,L),R)
    print()

    print("# find representative")
    list_repstate = []
    list_exponent = []
    print("#")
    print("# i state bit repstate repbit exponent")
    i = 0
    for state in range(2**L):
        if (count_upspins(state) == nup): ## fixed sz
            rep, exponent = find_representative(state,L)
            list_repstate.append(rep)
            list_exponent.append(exponent)
            print(k,i,state,num2bit(state,L),rep,num2bit(rep,L),exponent)
            i += 1
    print(list_repstate)
    print(list_exponent)
    print()

    print("# find representative (faster)")
    list_repstate = []
    list_exponent = []
    print("#")
    print("# i state bit repstate repbit exponent")
    first = (1<<(L-nup))-1
    last = ((1<<(L-nup))-1)<<(nup)
    print("# first:",first,num2bit(first,L))
    print("# last:",last,num2bit(last,L))
    state = first
    for i in range(binomial(L,nup)):
        rep, exponent = find_representative(state,L)
        list_repstate.append(rep)
        list_exponent.append(exponent)
        print(k,i,state,num2bit(state,L),rep,num2bit(rep,L),exponent)
        state = get_next_same_nup_state(state)
    print(list_repstate)
    print(list_exponent)
    print()

    print("# Hamiltonian")
    Nbond = L
    list_site1 = [i for i in range(Nbond)]
    list_site2 = [(i+1)%L for i in range(Nbond)]
    print("list_site1=",list_site1)
    print("list_site2=",list_site2)
    for k in range(L):
        Nhilbert = Nrep[k]
        Ham = np.zeros((Nhilbert,Nhilbert),dtype=complex)
        print("#")
        print("# int_momentum_k=",k)
        print("# Nhilbert=",Nhilbert)
        print("# bond_i,ind_a,sa,num2bit(sa,L),bb,num2bit(bb,L),sb,num2bit(sb,L),ind_b,exponent")
        for a in range(Nhilbert):
            sa = list_state[k][a]
            for i in range(Nbond):
                i1 = list_site1[i]
                i2 = list_site2[i]
                if get_spin(sa,i1) == get_spin(sa,i2):
                    Ham[a,a] += 0.25
                else:
                    Ham[a,a] -= 0.25
                    bb = flip_2spins(sa,i1,i2)
                    sb, exponent = find_representative(bb,L)
                    b = find_state_2(sb,list_state[k],Nhilbert)
                    if b>=0:
                        print(i,a,sa,num2bit(sa,L),bb,num2bit(bb,L),sb,num2bit(sb,L),b,exponent)
                        Ham[a,b] += 0.5*np.sqrt(float(list_R[k][a])/float(list_R[k][b]))*np.exp(-1j*exponent*2.0*np.pi*k/L)
        print("# Ham")
        print(Ham)
    print()

if __name__ == "__main__":
    main()
