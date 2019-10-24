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
    parser.add_argument('-N', metavar='N',dest='N', type=int, default=8, help='set Nsize (should be >=4)')
    parser.add_argument('-J1', metavar='J1',dest='J1', type=float, default=1.0, help='set J1')
    parser.add_argument('-J2', metavar='J2',dest='J2', type=float, default=0.0, help='set J2')
    return parser.parse_args()

def make_hamiltonian(Jxx,Jzz,list_isite1,list_isite2,N,Nint,Nhilbert):
    listki = np.zeros((Nint+1)*Nhilbert,dtype=int)
    loc = np.zeros((Nint+1)*Nhilbert,dtype=int)
    elemnt = np.zeros((Nint+1)*Nhilbert,dtype=float)
    listki = [i for k in range(Nint+1) for i in range(Nhilbert)]
    for k in range(Nint): # loop for all interactions
        isite1 = list_isite1[k]
        isite2 = list_isite2[k]
        is1 = 1<<isite1
        is2 = 1<<isite2
        is12 = is1 + is2
        wght = 2.0*Jxx[k]
        diag = Jzz[k]
        for i in range(Nhilbert): # loop for all spin configurations
            ibit = i & is12
            loc[Nint*Nhilbert+i] = i # store diag index
            if (ibit==0 or ibit==is12): # if (spin1,spin2) = (00) or (11)
                elemnt[Nint*Nhilbert+i] += diag # store +Jzz
#                print("# diag k(interactions) i(Hilbert)",k,i)
#                print("# diag i   ",np.binary_repr(i,width=N))
#                print("# diag is12",np.binary_repr(is12,width=N))
#                print("# diag ibit",np.binary_repr(ibit,width=N))
            else: # if (spin1,spin2) = (01) or (10)
                elemnt[Nint*Nhilbert+i] -= diag # store -Jzz
                iexchg = i ^ is12
                elemnt[k*Nhilbert+i] = wght # store 2*Jxx
                loc[k*Nhilbert+i] = iexchg # store offdiag index
#                print("# offdiag k(interactions) i(Hilbert)",k,i)
#                print("# offdiag i   ",np.binary_repr(i,width=N))
#                print("# offdiag is12",np.binary_repr(is12,width=N))
#                print("# offdiag iexc",np.binary_repr(iexchg,width=N))
    HamCSR = scipy.sparse.csr_matrix((elemnt,(listki,loc)),shape=(Nhilbert,Nhilbert))
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

def calc_xcorr(Nhilbert,Ncorr,list_corr_isite1,list_corr_isite2,psi):
    sxx = np.zeros(Ncorr,dtype=float)
    for k in range(Ncorr): # loop for all bonds for correlations
        isite1 = list_corr_isite1[k]
        isite2 = list_corr_isite2[k]
        is1 = 1<<isite1
        is2 = 1<<isite2
        is12 = is1 + is2
        corr = 0.0
        for i in range(Nhilbert): # loop for all spin configurations
            ibit = i & is12
            if (ibit==is1 or ibit==is2): # if (spin1,spin2) = (10) or (01)
                iexchg = i ^ is12
                corr += np.real(np.conj(psi[iexchg])*psi[i])
        sxx[k] = 0.25 * corr
        if (isite1==isite2):
            sxx[k] = 0.25
    return sxx

def make_lattice(N,J1,J2):
    Jxx = []
    Jzz = []
    list_isite1 = []
    list_isite2 = []
    Nint = 0
    for i in range(N):
        site1 = i
        site2 = (i+1)%N
        site3 = (i+2)%N
#
        list_isite1.append(site1)
        list_isite2.append(site2)
        Jxx.append(J1)
        Jzz.append(J1)
        Nint += 1
#
        list_isite1.append(site1)
        list_isite2.append(site3)
        Jxx.append(J2)
        Jzz.append(J2)
        Nint += 1
    return Jxx, Jzz, list_isite1, list_isite2, Nint

def main():
    args = parse_args()
    N = args.N
    J1 = args.J1
    J2 = args.J2
    Nhilbert = 2**N
    print("J1=",J1)
    print("J2=",J2)
    print("N=",N)
    print("Nhilbert=",Nhilbert)
    print("")

    Jxx, Jzz, list_isite1, list_isite2, Nint = make_lattice(N,J1,J2)
    print (Jxx)
    print (Jzz)
    print (list_isite1)
    print (list_isite2)
    print("Nint=",Nint)

    start = time.time()
    HamCSR = make_hamiltonian(Jxx,Jzz,list_isite1,list_isite2,N,Nint,Nhilbert)
    end = time.time()
    print (end - start)
#    print (HamCSR)
    start = time.time()
#    ene,vec = scipy.sparse.linalg.eigsh(HamCSR,k=5)
    ene,vec = scipy.sparse.linalg.eigsh(HamCSR,which='SA',k=5)
    end = time.time()
    print (end - start)
#    print ("# GS energy:",ene[0])
    print ("# energy:",ene[0],ene[1],ene[2],ene[3],ene[4])
#    vec_sgn = np.sign(np.amax(vec[:,0]))
#    print ("# GS wave function:")
#    for i in range(Nhilbert):
#        bini = np.binary_repr(i,width=N)
#        print (i,vec[i,0]*vec_sgn,bini)
#
    print("")

    Ncorr = N # number of total correlations
    list_corr_isite1 = [0 for k in range(Ncorr)] # site 1
    list_corr_isite2 = [k for k in range(Ncorr)] # site 2
    print (list_corr_isite1)
    print (list_corr_isite2)
    psi = vec[:,0] # choose the ground state
    start = time.time()
    szz = calc_zcorr(Nhilbert,Ncorr,list_corr_isite1,list_corr_isite2,psi)
    sxx = calc_xcorr(Nhilbert,Ncorr,list_corr_isite1,list_corr_isite2,psi)
    ss = szz+sxx+sxx
    stot2 = N*np.sum(ss)
    end = time.time()
    print (end - start)
    print ("# szz:",szz)
    print ("# sxx:",sxx)
    print ("# ss:",ss)
    print ("# stot(stot+1):",stot2)

if __name__ == "__main__":
    main()
