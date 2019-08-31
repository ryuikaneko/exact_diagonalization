#!/usr/bin/env python

# coding:utf-8
from __future__ import print_function
import math
import numpy as np
#import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Calculate the ground state of S=1/2 Heisenberg chain for a given Sz')
    parser.add_argument('-N', metavar='N',dest='N', type=int, default=8, help='set Nsize (should be >=4)')
    parser.add_argument('-Sz', metavar='Sz',dest='Sz', type=int, default=0, help='set Sz')
    return parser.parse_args()

def snoob(x):
    next = 0
    if(x>0):
        smallest = x & -(x)
        ripple = x + smallest
        ones = x ^ ripple
        ones = (ones >> 2) // smallest
        next = ripple | ones
    return next

def binomial(n,r):
    return math.factorial(n) // (math.factorial(n - r) * math.factorial(r))

def count_bit(n):
    count = 0
    while (n): 
        count += n & 1
        n >>= 1
    return count 

def init_parameters(N,Sz):
    Nup = N//2 + Sz
    Nhilbert = binomial(N,Nup)
    ihfbit = 1 << (N//2)
    irght = ihfbit-1
    ilft = ((1<<N)-1) ^ irght
    iup = (1<<(N-Nup))-1
    return Nup, Nhilbert, ihfbit, irght, ilft, iup

def make_list(N,Nup,Nhilbert,ihfbit,irght,ilft,iup):
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
    ii = snoob(ii)
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
        ii = snoob(ii)
    return list_1, list_ja, list_jb

def get_ja_plus_jb(ii,irght,ilft,ihfbit,list_ja,list_jb):
    ia = ii & irght
    ib = (ii & ilft) // ihfbit
    ja = list_ja[ia]
    jb = list_jb[ib]
    return ja+jb

def make_hamiltonian(J1,D1,N,Nhilbert,irght,ilft,ihfbit,list_1,list_ja,list_jb):
    listki = np.zeros((N+1)*Nhilbert,dtype=int)
    loc = np.zeros((N+1)*Nhilbert,dtype=int)
    elemnt = np.zeros((N+1)*Nhilbert,dtype=float)
    listki = [i for k in range(N+1) for i in range(Nhilbert)]
    for k in range(N):
        isite1 = k
        isite2 = (k+1)%N
        is1 = 1<<isite1
        is2 = 1<<isite2
        is0 = is1 + is2
        wght = -2.0*J1[k]
        diag = wght*0.5*D1[k]
        for i in range(Nhilbert):
            ii = list_1[i]
            ibit = ii & is0
            if (ibit==0 or ibit==is0):
                elemnt[N*Nhilbert+i] -= diag
                loc[N*Nhilbert+i] = i
            else:
                elemnt[N*Nhilbert+i] += diag
                loc[N*Nhilbert+i] = i
                iexchg = ii ^ is0
                newcfg = get_ja_plus_jb(iexchg,irght,ilft,ihfbit,list_ja,list_jb)
                elemnt[k*Nhilbert+i] = -wght
                loc[k*Nhilbert+i] = newcfg
    HamCSR = scipy.sparse.csr_matrix((elemnt,(listki,loc)),shape=(Nhilbert,Nhilbert))
    return HamCSR

def main():
    args = parse_args()
    N = args.N
    Sz = args.Sz
    Nup, Nhilbert, ihfbit, irght, ilft, iup = init_parameters(N,Sz)
    binirght = np.binary_repr(irght,width=N)
    binilft = np.binary_repr(ilft,width=N)
    biniup = np.binary_repr(iup,width=N)
    print("N=",N)
    print("Sz=",Sz)
    print("Nup=",Nup)
    print("Nhilbert=",Nhilbert)
    print("ihfbit=",ihfbit)
    print("irght,binirght=",irght,binirght)
    print("ilft,binilft=",ilft,binilft)
    print("iup,biniup=",iup,biniup)
    start = time.time()
    list_1, list_ja, list_jb = make_list(N,Nup,Nhilbert,ihfbit,irght,ilft,iup)
    end = time.time()
    print (end - start)
#    print("list_1=",list_1)
#    print("list_ja=",list_ja)
#    print("list_jb=",list_jb)
    print("")
#    print("i ii binii ja+jb")
#    for i in range(Nhilbert):
#        ii = list_1[i]
#        binii = np.binary_repr(ii,width=N)
#        ind = get_ja_plus_jb(ii,irght,ilft,ihfbit,list_ja,list_jb)
#        print(i,ii,binii,ind)
    J1 = np.ones(N,dtype=float) # J_{ij}>0: AF
    D1 = np.ones(N,dtype=float) # D_{ij}>0: AF
    start = time.time()
    HamCSR = make_hamiltonian(J1,D1,N,Nhilbert,irght,ilft,ihfbit,list_1,list_ja,list_jb)
    end = time.time()
    print (end - start)
#    print (HamCSR)
    start = time.time()
    ene,vec = scipy.sparse.linalg.eigsh(HamCSR,k=5)
    end = time.time()
    print (end - start)
    #print ("# GS energy:",ene[0])
    print ("# energy:",ene[0],ene[1],ene[2],ene[3],ene[4])
#    vec_sgn = np.sign(np.amax(vec[:,0]))
#    print ("# GS wave function:")
#    for i in range (Nhilbert):
#        ii = list_1[i]
#        binii = np.binary_repr(ii,width=N)
#        print (i,vec[i,0]*vec_sgn,binii)

if __name__ == "__main__":
    main()
