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
    parser = argparse.ArgumentParser(description='Calculate the ground state of S=1/2 Heisenberg ladder for a given Sz')
    parser.add_argument('-Lx', metavar='Lx',dest='Lx', type=int, default=8, help='set Lx (should be >=2)')
    parser.add_argument('-Ly', metavar='Ly',dest='Ly', type=int, default=2, help='set Ly=2')
    parser.add_argument('-Jleg', metavar='Jleg',dest='Jleg', type=float, default=1.0, help='set Jleg')
    parser.add_argument('-Jrung', metavar='Jrung',dest='Jrung', type=float, default=1.0, help='set Jrung')
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

#def make_hamiltonian(Jxx,Jzz,list_isite1,list_isite2,N,Nint,Nhilbert,irght,ilft,ihfbit,list_1,list_ja,list_jb):
@jit(nopython=True)
def make_hamiltonian_child(Jxx,Jzz,list_isite1,list_isite2,N,Nint,Nhilbert,irght,ilft,ihfbit,list_1,list_ja,list_jb):
#    listki = np.zeros((Nint+1)*Nhilbert,dtype=np.int64)
    loc = np.zeros((Nint+1)*Nhilbert,dtype=np.int64)
    elemnt = np.zeros((Nint+1)*Nhilbert,dtype=np.float64)
#    listki = [i for k in range(Nint+1) for i in range(Nhilbert)]
    listki = np.array([i for k in range(Nint+1) for i in range(Nhilbert)],dtype=np.int64)
    for k in range(Nint): # loop for all interactions
        isite1 = list_isite1[k]
        isite2 = list_isite2[k]
        is1 = 1<<isite1
        is2 = 1<<isite2
        is12 = is1 + is2
        wght = 2.0*Jxx[k]
        diag = Jzz[k]
## calculate elements of
## H_loc = Jzz sgmz.sgmz + Jxx (sgmx.sgmx + sgmy.sgmy)
##       = Jzz sgmz.sgmz + 2*Jxx (S+.S- + S-.S+)
        for i in range(Nhilbert): # loop for all spin configurations with fixed Sz
            ii = list_1[i]
            ibit = ii & is12 # find sgmz.sgmz|uu> = |uu> or sgmz.sgmz|dd> = |dd>
            loc[Nint*Nhilbert+i] = i # store diag index
            if (ibit==0 or ibit==is12): # if (spin1,spin2) = (00) or (11): sgmz.sgmz only
                elemnt[Nint*Nhilbert+i] += diag # store +Jzz
#                print("# diag k(interactions) i(Hilbert)",k,i)
#                print("# diag ii  ",np.binary_repr(ii,width=N))
#                print("# diag is12",np.binary_repr(is12,width=N))
#                print("# diag ibit",np.binary_repr(ibit,width=N))
            else: # if (spin1,spin2) = (01) or (10): sgmz.sgmz and (S+.S- or S-.S+)
                elemnt[Nint*Nhilbert+i] -= diag # store -Jzz
                iexchg = ii ^ is12 # find S+.S-|du> = |ud> or S-.S+|ud> = |du>
                newcfg = get_ja_plus_jb(iexchg,irght,ilft,ihfbit,list_ja,list_jb)
                elemnt[k*Nhilbert+i] = wght # store 2*Jxx
                loc[k*Nhilbert+i] = newcfg # store offdiag index
#                print("# offdiag k(interactions) i(Hilbert)",k,i)
#                print("# offdiag ii  ",np.binary_repr(ii,width=N))
#                print("# offdiag is12",np.binary_repr(is12,width=N))
#                print("# offdiag iexc",np.binary_repr(iexchg,width=N))
#    HamCSR = scipy.sparse.csr_matrix((elemnt,(listki,loc)),shape=(Nhilbert,Nhilbert))
#    return HamCSR
    return elemnt, listki, loc

def make_hamiltonian(Nhilbert,elemnt,listki,loc):
    return scipy.sparse.csr_matrix((elemnt,(listki,loc)),shape=(Nhilbert,Nhilbert),dtype=np.float64)

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
def make_lattice(Lx,Ly,Jleg,Jrung):
    Jxx = []
    Jzz = []
    list_isite1 = []
    list_isite2 = []
    Nint = 0
    for ix in range(Lx):
        site0 = ix+Lx
        site1 = ix
        site2 = (ix+1)%Lx+Lx
        site3 = (ix+1)%Lx
#
        list_isite1.append(site0)
        list_isite2.append(site2)
        Jxx.append(Jleg)
        Jzz.append(Jleg)
        Nint += 1
#
        list_isite1.append(site1)
        list_isite2.append(site3)
        Jxx.append(Jleg)
        Jzz.append(Jleg)
        Nint += 1
#
        list_isite1.append(site0)
        list_isite2.append(site1)
        Jxx.append(Jrung)
        Jzz.append(Jrung)
        Nint += 1
    return Jxx, Jzz, list_isite1, list_isite2, Nint


def main():
    args = parse_args()
    Lx = args.Lx
    Ly = args.Ly
    N = Lx*Ly
    Sz = args.Sz
    Jleg = args.Jleg
    Jrung = args.Jrung

    print("# make list")
    start = time.time()
    Nup, Nhilbert, ihfbit, irght, ilft, iup = init_parameters(N,Sz)
    binirght = np.binary_repr(irght,width=N)
    binilft = np.binary_repr(ilft,width=N)
    biniup = np.binary_repr(iup,width=N)
    print("Lx=",Lx)
    print("Ly=",Ly)
    print("Jleg=",Jleg)
    print("Jrung=",Jrung)
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
    Jxx, Jzz, list_isite1, list_isite2, Nint = make_lattice(Lx,Ly,Jleg,Jrung)
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
    elemnt, listki, loc = make_hamiltonian_child(Jxx,Jzz,list_isite1,list_isite2,N,Nint,Nhilbert,irght,ilft,ihfbit,list_1,list_ja,list_jb)
    HamCSR = make_hamiltonian(Nhilbert,elemnt,listki,loc)
#    print(HamCSR)
    end = time.time()
    print("# time=",end - start)
    print("")

    print("# diagonalization")
    start = time.time()
#    ene,vec = scipy.sparse.linalg.eigsh(HamCSR,k=5)
    ene,vec = scipy.sparse.linalg.eigsh(HamCSR,which='SA',k=5)
    ene = ene/N/4
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
    Ncorr = N # number of total correlations
    list_corr_isite1 = np.array([0 for k in range(Ncorr)],dtype=np.int64) # site 1
    list_corr_isite2 = np.array([k for k in range(Ncorr)],dtype=np.int64) # site 2
    print(list_corr_isite1)
    print(list_corr_isite2)
    psi = vec[:,0] # choose the ground state
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
