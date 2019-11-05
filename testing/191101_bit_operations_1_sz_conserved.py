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
def find_state(state,list_1,maxind):
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
    return i


def main():
    args = parse_args()
    L = args.L

    print("# show state")
    print("# num num2bit ket bit2num")
    for state in range(L):
        spins = show_state(state,L)
        n2b = num2bit(state,L)
        b2n = bit2num(n2b)
        print(state,n2b,spins,b2n)
    print("...")
    for state in range(2**L-L,2**L):
        spins = show_state(state,L)
        n2b = num2bit(state,L)
        b2n = bit2num(n2b)
        print(state,n2b,spins,b2n)
    print()

    print("# get spin")
    state = 2**(L//2)+6
    for site in range(L):
        spins01 = show_state01(state,L)
        spins = show_state(state,L)
        spin = get_spin(state,site)
        print(site,"bit (from right to left) of state",state,"=",num2bit(state,L),\
            "is",spin)
    print()

    print("# get spin alternative")
    state = 2**(L//2)+6
    for site in range(L):
        spins01 = show_state01(state,L)
        spins = show_state(state,L)
        spin = get_spin(state,site)
        print(site,"bit (from right to left) of state",state,"=",num2bit(state,L),\
            "is",spin)
    print()

    print("# permutation")
    state = 2**(L//2)+6
    for shift in range(L):
        newstate = shift_1spin(state,L)
        print(shift,num2bit(state,L),"--> shift 1 -->",num2bit(newstate,L))
        state = newstate
    print()

    print("# inverse permutation")
    state = 2**(L//2)+6
    for shift in range(L):
        newstate = shift_1spin_inv(state,L)
        print(shift,num2bit(state,L),"--> shift -1 -->",num2bit(newstate,L))
        state = newstate
    print()

    print("# permutation (given shift)")
    state = 2**(L//2)+6
    for shift in range(L+1):
        newstate = shift_spin(state,L,shift)
        print(num2bit(state,L),"--> shift",shift,"-->",num2bit(newstate,L))
    print()

    print("# count up spins")
    for state in range(L):
        nup = count_upspins(state)
        print(state,num2bit(state,L),nup)
    print("...")
    for state in range(2**L-L,2**L):
        nup = count_upspins(state)
        print(state,num2bit(state,L),nup)
    print()

    print("# count up spins alternative")
    for state in range(L):
        nup = count_upspins_alternative(state)
        print(state,num2bit(state,L),nup)
    print("...")
    for state in range(2**L-L,2**L):
        nup = count_upspins_alternative(state)
        print(state,num2bit(state,L),nup)
    print()

    print("# next same nup state")
    nup = L//2-1
    first = (1<<(L-nup))-1
    last = ((1<<(L-nup))-1)<<(nup)
    print("# first:",first,num2bit(first,L))
    print("# last:",last,num2bit(last,L))
    state = first
    for i in range(binomial(L,nup)):
        print(i,state,num2bit(state,L))
        state = get_next_same_nup_state(state)
    print()

    print("# next same nup state alternative")
    nup = L//2-1
    first = (1<<(L-nup))-1
    last = ((1<<(L-nup))-1)<<(nup)
    print("# first:",first,num2bit(first,L))
    print("# last:",last,num2bit(last,L))
    state = first
    for i in range(binomial(L,nup)):
        print(i,state,num2bit(state,L))
        state = get_next_same_nup_state_alternative(state)
    print()

    print("# find same nup states in list_1 by list_ja and list_jb")
    print("# ind_state num_state(=list_1) bit_state bit_state(left) bit_state(right) ",end="")
    print("ib ia list_jb[ib] list_ja[ia] state2ind")
    nup = L//2-1
    Nhilbert, ihfbit, irght, ilft, iup = init_parameters(L,nup)
    list_1, list_ja, list_jb = make_list_same_nup(Nhilbert,ihfbit,irght,ilft,iup)
    for i in range(Nhilbert):
        ii = list_1[i]
        bit = num2bit(ii,L)
        ind = get_ja_plus_jb(ii,irght,ilft,ihfbit,list_ja,list_jb)
        ia = ii & irght
        ib = (ii & ilft) // ihfbit
        print(i,ii,bit,num2bit(ib,L//2),num2bit(ia,L//2),ib,ia,list_jb[ib],list_ja[ia],ind)
    print()

## http://physics.bu.edu/~sandvik/vietri/dia.pdf
## http://lptms.u-psud.fr/membres/groux/Test/ED/ED_Lecture2.html
    print("# find same nup states in list_1 by a bisection method")
    for i in range(Nhilbert):
        ii = list_1[i]
        bit = num2bit(ii,L)
        ind = find_state(ii,list_1,list_1.size)
        print("[bisection] ",i,ii,bit,ind)
#        print(i,ii,bit,ind)
    print()

## https://github.com/alexwie/ed_basics/blob/master/hamiltonian_hb_xxz.py
##
## https://stackoverflow.com/questions/51127209/numpy-ndarray-object-has-no-attribute-index
## https://stackoverflow.com/questions/21488005/how-to-find-the-index-of-an-array-within-an-array
##
## ind1 by list.index might be faster?
## https://stackoverflow.com/questions/18452591/fast-python-numpy-where-functionality
## https://stackoverflow.com/questions/5913671/complexity-of-list-indexx-in-python
##
## practically, np.where seems faster than list.index
##
    print("# find same nup states in list_1 by np.where")
    nup = L//2-1
    Nhilbert, ihfbit, irght, ilft, iup = init_parameters(L,nup)
    list_1, list_ja, list_jb = make_list_same_nup(Nhilbert,ihfbit,irght,ilft,iup)
    for i in range(Nhilbert):
        ii = list_1[i]
        bit = num2bit(ii,L)
        ind = np.where(list_1==ii)[0].item()
        print("[np.where] ",i,ii,bit,ind)
#        print(i,ii,bit,ind)
    print()

    print("# find same nup states in list_1 by list.index")
    nup = L//2-1
    Nhilbert, ihfbit, irght, ilft, iup = init_parameters(L,nup)
    list_1, list_ja, list_jb = make_list_same_nup(Nhilbert,ihfbit,irght,ilft,iup)
    for i in range(Nhilbert):
        ii = list_1[i]
        bit = num2bit(ii,L)
        ind = list_1.tolist().index(ii)
        print("[list.index] ",i,ii,bit,ind)
#        print(i,ii,bit,ind)
    print()

## speed
## list_j > bisection > np.where > list.index

if __name__ == "__main__":
    main()
