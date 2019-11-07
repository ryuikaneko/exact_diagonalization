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
#
from numba import jit

def parse_args():
    parser = argparse.ArgumentParser(description='Calculate the ground state of S=1/2 Heisenberg chain for a given Sz')
    parser.add_argument('-N', metavar='N',dest='N', type=int, default=8, help='set Nsize (should be >=4)')
    parser.add_argument('-J1', metavar='J1',dest='J1', type=float, default=1.0, help='set J1')
    parser.add_argument('-J2', metavar='J2',dest='J2', type=float, default=0.0, help='set J2')
    parser.add_argument('-Sz', metavar='Sz',dest='Sz', type=int, default=0, help='set Sz')
    return parser.parse_args()

@jit(nopython=True)
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

#@jit(nopython=True)
#def count_bit(n):
#    count = 0
#    while (n): 
#        count += n & 1
#        n >>= 1
#    return count 

def init_parameters(N,Sz):
    Nup = N//2 + Sz
    Nhilbert = binomial(N,Nup)
    ihfbit = 1 << (N//2)
    irght = ihfbit-1
    ilft = ((1<<N)-1) ^ irght
    iup = (1<<(N-Nup))-1
    return Nup, Nhilbert, ihfbit, irght, ilft, iup

@jit(nopython=True)
def make_list(N,Nup,Nhilbert,ihfbit,irght,ilft,iup):
    list_1 = np.zeros(Nhilbert,dtype=np.int64)
    list_ja = np.zeros(ihfbit,dtype=np.int64)
    list_jb = np.zeros(ihfbit,dtype=np.int64)
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

@jit(nopython=True)
def get_ja_plus_jb(ii,irght,ilft,ihfbit,list_ja,list_jb):
    ia = ii & irght
    ib = (ii & ilft) // ihfbit
    ja = list_ja[ia]
    jb = list_jb[ib]
    return ja+jb

def make_hamiltonian(Jxx,Jzz,list_isite1,list_isite2,N,Nint,Nhilbert,irght,ilft,ihfbit,list_1,list_ja,list_jb):
#    listki = np.zeros((Nint+1)*Nhilbert,dtype=np.int64)
#    loc = np.zeros((Nint+1)*Nhilbert,dtype=np.int64)
#    elemnt = np.zeros((Nint+1)*Nhilbert,dtype=np.float64)
#    listki = [i for k in range(Nint+1) for i in range(Nhilbert)]
    @jit(nopython=True)
    def get_vec(vec):
        vecnew = np.zeros(Nhilbert,dtype=np.float64)
        for k in range(Nint): # loop for all interactions
            isite1 = list_isite1[k]
            isite2 = list_isite2[k]
            is1 = 1<<isite1
            is2 = 1<<isite2
            is12 = is1 + is2
            wght = 2.0*Jxx[k]
            diag = Jzz[k]
            for i in range(Nhilbert): # loop for all spin configurations with fixed Sz
                ii = list_1[i]
                ibit = ii & is12 # find sgmz.sgmz|uu> = |uu> or sgmz.sgmz|dd> = |dd>
#                loc[Nint*Nhilbert+i] = i # store diag index
                if (ibit==0 or ibit==is12): # if (spin1,spin2) = (00) or (11): sgmz.sgmz only
                    vecnew[i] += +diag*vec[i]
#                    elemnt[Nint*Nhilbert+i] += diag # store +Jzz
                else: # if (spin1,spin2) = (01) or (10): sgmz.sgmz and (S+.S- or S-.S+)
#                    elemnt[Nint*Nhilbert+i] -= diag # store -Jzz
                    vecnew[i] += -diag*vec[i]
                    iexchg = ii ^ is12 # find S+.S-|du> = |ud> or S-.S+|ud> = |du>
                    newcfg = get_ja_plus_jb(iexchg,irght,ilft,ihfbit,list_ja,list_jb)
                    vecnew[i] += vec[newcfg]*wght
#                    elemnt[k*Nhilbert+i] = wght # store 2*Jxx
#                    loc[k*Nhilbert+i] = newcfg # store offdiag index
#        HamCSR = scipy.sparse.csr_matrix((elemnt,(listki,loc)),shape=(Nhilbert,Nhilbert))
#        return HamCSR
        vec = vecnew[:]
        return vec
    return get_vec

@jit(nopython=True)
def calc_zcorr(Nhilbert,Ncorr,list_corr_isite1,list_corr_isite2,psi,list_1):
    szz = np.zeros(Ncorr,dtype=np.float64)
    for k in range(Ncorr): # loop for all bonds for correlations
        isite1 = list_corr_isite1[k]
        isite2 = list_corr_isite2[k]
        is1 = 1<<isite1
        is2 = 1<<isite2
        is12 = is1 + is2
        corr = 0.0
        for i in range(Nhilbert): # loop for all spin configurations with fixed Sz
            ii = list_1[i]
            ibit = ii & is12 # find sgmz.sgmz|uu> = |uu> or sgmz.sgmz|dd> = |dd>
            if (ibit==0 or ibit==is12): # if (spin1,spin2) = (00) or (11): factor = +1
                factor = +1.0
            else: # if (spin1,spin2) = (01) or (10): factor = -1
                factor = -1.0
            corr += factor*psi[i]**2 # psi[i]: real
        szz[k] = 0.25 * corr
        if (isite1==isite2):
            szz[k] = 0.25
    return szz

@jit(nopython=True)
def calc_xcorr(Nhilbert,Ncorr,list_corr_isite1,list_corr_isite2,psi,irght,ilft,ihfbit,list_1,list_ja,list_jb):
    sxx = np.zeros(Ncorr,dtype=np.float64)
    for k in range(Ncorr): # loop for all bonds for correlations
        isite1 = list_corr_isite1[k]
        isite2 = list_corr_isite2[k]
        is1 = 1<<isite1
        is2 = 1<<isite2
        is12 = is1 + is2
        corr = 0.0
        for i in range(Nhilbert): # loop for all spin configurations with fixed Sz
            ii = list_1[i]
            ibit = ii & is12 # find sgmz.sgmz|ud> = -|ud> or sgmz.sgmz|du> = -|du>
            if (ibit==is1 or ibit==is2): # if (spin1,spin2) = (10) or (01)
                iexchg = ii ^ is12 # find S+.S-|du> = |ud> or S-.S+|ud> = |du>
                newcfg = get_ja_plus_jb(iexchg,irght,ilft,ihfbit,list_ja,list_jb)
                corr += psi[i]*psi[newcfg] # psi[i]: real
        sxx[k] = 0.25 * corr
        if (isite1==isite2):
            sxx[k] = 0.25
    return sxx

#def make_lattice(N,J1,J2):
#    Nint = N # number of total interactions
#    Jxx = np.ones(Nint,dtype=np.float64) # Jxx_{ij}>0: AF
#    Jzz = np.ones(Nint,dtype=np.float64) # Jzz_{ij}>0: AF
#    list_isite1 = [k for k in range(N)] # n.n. site 1
#    list_isite2 = [(k+1)%N for k in range(N)] # n.n. site 2
#    return Jxx, Jzz, list_isite1, list_isite2, Nint

@jit(nopython=True)
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
    Sz = args.Sz
    J1 = args.J1
    J2 = args.J2

    print("# make list")
    start = time.time()
    Nup, Nhilbert, ihfbit, irght, ilft, iup = init_parameters(N,Sz)
    binirght = np.binary_repr(irght,width=N)
    binilft = np.binary_repr(ilft,width=N)
    biniup = np.binary_repr(iup,width=N)
    print("J1=",J1)
    print("J2=",J2)
    print("N=",N)
    print("Sz=",Sz)
    print("Nup=",Nup)
    print("Nhilbert=",Nhilbert)
    print("ihfbit=",ihfbit)
    print("irght,binirght=",irght,binirght)
    print("ilft,binilft=",ilft,binilft)
    print("iup,biniup=",iup,biniup)
    list_1, list_ja, list_jb = make_list(N,Nup,Nhilbert,ihfbit,irght,ilft,iup)
    end = time.time()
#    print("list_1=",list_1)
#    print("list_ja=",list_ja)
#    print("list_jb=",list_jb)
#    print("")
#    print("i ii binii ja+jb")
#    for i in range(Nhilbert):
#        ii = list_1[i]
#        binii = np.binary_repr(ii,width=N)
#        ind = get_ja_plus_jb(ii,irght,ilft,ihfbit,list_ja,list_jb)
#        print(i,ii,binii,ind)
    print("# time=",end - start)
    print("")

    print("# make lattice")
    start = time.time()
    Jxx, Jzz, list_isite1, list_isite2, Nint = make_lattice(N,J1,J2)
    Jxx = np.array(Jxx)
    Jzz = np.array(Jzz)
    list_isite1 = np.array(list_isite1)
    list_isite2 = np.array(list_isite2)
    print(Jxx)
    print(Jzz)
    print(list_isite1)
    print(list_isite2)
    print("Nint=",Nint)
    end = time.time()
    print("# time=",end - start)
    print("")

    print("# make Hamiltonian")
    start = time.time()
#    HamCSR = make_hamiltonian(Jxx,Jzz,list_isite1,list_isite2,N,Nint,Nhilbert,irght,ilft,ihfbit,list_1,list_ja,list_jb)
#    print(HamCSR)
    get_vec = make_hamiltonian(Jxx,Jzz,list_isite1,list_isite2,N,Nint,Nhilbert,irght,ilft,ihfbit,list_1,list_ja,list_jb)
    Ham = scipy.sparse.linalg.LinearOperator((Nhilbert,Nhilbert),matvec=get_vec)
    print(Ham)
    end = time.time()
    print("# time=",end - start)
    print("")

    print("# diagonalization")
    start = time.time()
##    ene,vec = scipy.sparse.linalg.eigsh(HamCSR,k=5)
#    ene,vec = scipy.sparse.linalg.eigsh(HamCSR,which='SA',k=5)
    ene,vec = scipy.sparse.linalg.eigsh(Ham,which='SA',k=5)
    end = time.time()
    #print("# GS energy:",ene[0])
    print("# energy:",ene[0],ene[1],ene[2],ene[3],ene[4])
#    vec_sgn = np.sign(np.amax(vec[:,0]))
#    print("# GS wave function:")
#    for i in range (Nhilbert):
#        ii = list_1[i]
#        binii = np.binary_repr(ii,width=N)
#        print(i,vec[i,0]*vec_sgn,binii)
#
    print("# time=",end - start)
    print("")

    print("# calc correlations")
    start = time.time()
#    Ncorr = N/2+1 # number of total correlations
    Ncorr = N # number of total correlations
    list_corr_isite1 = np.array([0 for k in range(Ncorr)],dtype=np.int64) # site 1
    list_corr_isite2 = np.array([k for k in range(Ncorr)],dtype=np.int64) # site 2
    print(list_corr_isite1)
    print(list_corr_isite2)
    psi = vec[:,0] # choose the ground state
    start = time.time()
    szz = calc_zcorr(Nhilbert,Ncorr,list_corr_isite1,list_corr_isite2,psi,list_1)
    sxx = calc_xcorr(Nhilbert,Ncorr,list_corr_isite1,list_corr_isite2,psi,irght,ilft,ihfbit,list_1,list_ja,list_jb)
    ss = szz+sxx+sxx
    stot2 = N*np.sum(ss)
    end = time.time()
    print("# szz:",szz)
    print("# sxx:",sxx)
    print("# ss:",ss)
    print("# stot(stot+1):",stot2)
    print("# time=",end - start)
    print("")

if __name__ == "__main__":
    main()
