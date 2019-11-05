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
    parser = argparse.ArgumentParser(description='Calculate the ground state of S=1/2 Heisenberg chain')
    parser.add_argument('-N', metavar='N',dest='N', type=int, default=10, help='set Nsize (should be >=4)')
    parser.add_argument('-Hx', metavar='Hx',dest='Hx', type=float, default=1.0, help='set transverse field Hx (default: critical point @ Hx/Jzz=1)')
    parser.add_argument('-Jzz', metavar='Jzz',dest='Jzz', type=float, default=1.0, help='set Ising interaction Jzz')
    return parser.parse_args()

def make_hamiltonian(list_Hx,list_Jzz,list_isite0,list_isite1,list_isite2,N,Nint1,Nint2,Nhilbert):
    Nint = Nint1 + Nint2
    listki = np.zeros((Nint+1)*Nhilbert,dtype=int)
    loc = np.zeros((Nint+1)*Nhilbert,dtype=int)
    elemnt = np.zeros((Nint+1)*Nhilbert,dtype=float)
    for k in range(Nint1): # loop for all 1-body interactions
        isite0 = list_isite0[k]
        is0 = 1<<isite0
        wght = list_Hx[k]
        for i in range(Nhilbert): # loop for all spin configurations
            iexchg = i ^ is0 # spin0: (0)-->(1), (1)-->(0)
            listki[k*Nhilbert+i] = i # store diag index (row)
            loc[k*Nhilbert+i] = iexchg # store offdiag index (col)
            elemnt[k*Nhilbert+i] += wght # store Hx
    for k in range(Nint2): # loop for all 2-body interactions
        isite1 = list_isite1[k]
        isite2 = list_isite2[k]
        is1 = 1<<isite1
        is2 = 1<<isite2
        is12 = is1 + is2
        diag = list_Jzz[k]
        for i in range(Nhilbert): # loop for all spin configurations
            ibit = i & is12
            listki[Nint*Nhilbert+i] = i # store diag index (row)
            loc[Nint*Nhilbert+i] = i # store diag index (col)
            if (ibit==0 or ibit==is12): # if (spin1,spin2) = (00) or (11)
                elemnt[Nint*Nhilbert+i] += diag # store +Jzz
            else: # if (spin1,spin2) = (01) or (10)
                elemnt[Nint*Nhilbert+i] -= diag # store -Jzz
    HamCSR = scipy.sparse.csr_matrix((elemnt,(listki,loc)),shape=(Nhilbert,Nhilbert))
#    print(HamCSR)
    return HamCSR

def calc_zcorr(Nhilbert,Ncorr,list_corr_isite1,list_corr_isite2,psi):
    szz = np.zeros(Ncorr,dtype=float)
    for k in range(Ncorr): # loop for all bonds for correlations
        isite1 = list_corr_isite1[k]
        isite2 = list_corr_isite2[k]
        is1 = 1<<isite1
        is2 = 1<<isite2
        is12 = is1 + is2
        corr = 0.0
        for i in range(Nhilbert): # loop for all spin configurations
            ibit = i & is12
            if (ibit==0 or ibit==is12): # if (spin1,spin2) = (00) or (11): factor = +1
                factor = +1.0
            else: # if (spin1,spin2) = (01) or (10): factor = -1
                factor = -1.0
            corr += factor*np.abs(psi[i])**2
        szz[k] = 0.25 * corr
        if (isite1==isite2):
            szz[k] = 0.25
    return szz

## TFIsing: <sx.sx> != <sy.sy>
##
#def calc_xcorr(Nhilbert,Ncorr,list_corr_isite1,list_corr_isite2,psi):
#    sxx = np.zeros(Ncorr,dtype=float)
#    for k in range(Ncorr): # loop for all bonds for correlations
#        isite1 = list_corr_isite1[k]
#        isite2 = list_corr_isite2[k]
#        is1 = 1<<isite1
#        is2 = 1<<isite2
#        is12 = is1 + is2
#        corr = 0.0
#        for i in range(Nhilbert): # loop for all spin configurations
#            ibit = i & is12
#            if (ibit==is1 or ibit==is2): # if (spin1,spin2) = (10) or (01)
#                iexchg = i ^ is12
#                corr += np.real(np.conj(psi[iexchg])*psi[i])
#        sxx[k] = 0.25 * corr
#        if (isite1==isite2):
#            sxx[k] = 0.25
#    return sxx

def make_lattice(N,Hx,Jzz):
    list_Hx = []
    list_Jzz = []
    list_isite0 = []
    list_isite1 = []
    list_isite2 = []
    Nint1 = 0
    Nint2 = 0
    for i in range(N):
        site0 = i
        site1 = i
        site2 = (i+1)%N
#
        list_isite0.append(site0)
        list_Hx.append(Hx)
        Nint1 += 1
#
        list_isite1.append(site1)
        list_isite2.append(site2)
        list_Jzz.append(Jzz)
        Nint2 += 1
    return np.array(list_Hx), np.array(list_Jzz), \
        np.array(list_isite0), np.array(list_isite1), np.array(list_isite2), \
        Nint1, Nint2

def main():
    args = parse_args()
    N = args.N
    Hx = args.Hx
    Jzz = args.Jzz
    N1bond = N
    N2bond = N   
    Nhilbert = 2**N
    print("N=",N)
    print("N1bond=",N1bond)
    print("N2bond=",N2bond)
    print("Jzz=",Jzz)
    print("Hx=",Hx)
    print("Nhilbert=",Nhilbert)
    print("")

    list_Hx, list_Jzz, list_isite0, list_isite1, list_isite2, Nint1, Nint2 = \
        make_lattice(N,Hx,Jzz)
    print("list_isite0=",list_isite0)
    print("list_isite1=",list_isite1)
    print("list_isite2=",list_isite2)
    print("list_Hx=",list_Hx)
    print("list_Jzz=",list_Jzz)
    print("Nint1=",Nint1)
    print("Nint2=",Nint2)

    start = time.time()
    HamCSR = make_hamiltonian(list_Hx,list_Jzz,list_isite0,list_isite1,list_isite2,N,Nint1,Nint2,Nhilbert)
    end = time.time()
    print(end - start)
    start = time.time()
    ene,vec = scipy.sparse.linalg.eigsh(HamCSR,which='SA',k=5)
    end = time.time()
    print(end - start)
#    print("# GS energy:",ene[0])
    print("# energy:",ene[0],ene[1],ene[2],ene[3],ene[4])
#    vec_sgn = np.sign(np.amax(vec[:,0]))
#    print("# GS wave function:")
#    for i in range(Nhilbert):
#        bini = np.binary_repr(i,width=N)
#        print(i,vec[i,0]*vec_sgn,bini)
#
    print("")

    Ncorr = N # number of total correlations
    list_corr_isite1 = np.array([0 for k in range(Ncorr)]) # site 1
    list_corr_isite2 = np.array([k for k in range(Ncorr)]) # site 2
    print(list_corr_isite1)
    print(list_corr_isite2)
    psi = vec[:,0] # choose the ground state
    start = time.time()
    szz = calc_zcorr(Nhilbert,Ncorr,list_corr_isite1,list_corr_isite2,psi)
#    sxx = calc_xcorr(Nhilbert,Ncorr,list_corr_isite1,list_corr_isite2,psi)
#    ss = szz+sxx+sxx
#    stot2 = N*np.sum(ss)
    end = time.time()
    print(end - start)
    print("# szz:",szz)
#    print("# sxx:",sxx)
#    print("# ss:",ss)
#    print("# stot(stot+1):",stot2)

if __name__ == "__main__":
    main()
