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
    return parser.parse_args()

def make_spin():
    S0 = scipy.sparse.csr_matrix(np.array([[1,0],[0,1]]))
    Sx = scipy.sparse.csr_matrix(np.array([[0,1],[1,0]]))
    Sy = scipy.sparse.csr_matrix(np.array([[0,-1j],[1j,0]]))
    Sz = scipy.sparse.csr_matrix(np.array([[1,0],[0,-1]]))
    return S0,Sx,Sy,Sz

def make_interaction_list(N,Nbond):
    list_site1 = [i for i in range(N)]
    list_site2 = [(i+1)%N for i in range(N)]
    list_Jxx = np.ones(N)
    list_Jyy = np.ones(N)
    list_Jzz = np.ones(N)
    return list_site1, list_site2, list_Jxx, list_Jyy, list_Jzz

def make_hamiltonian(S0,Sx,Sy,Sz,N,Nbond,\
    list_site1,list_site2,list_Jxx,list_Jyy,list_Jzz):
#    Ham = scipy.sparse.csr_matrix((2**N,2**N),dtype=complex)
    Ham = scipy.sparse.csr_matrix((2**N,2**N),dtype=float)
    for bond in range(Nbond):
        i1 = list_site1[bond]
        i2 = list_site2[bond]
        Jxx = list_Jxx[bond]
        Jyy = list_Jyy[bond]
        Jzz = list_Jzz[bond]
        SxSx = 1
        SySy = 1
        SzSz = 1
        for site in range(N):
            if site==i1 or site==i2:
                SxSx = scipy.sparse.kron(SxSx,Sx,format='csr')
                SySy = scipy.sparse.kron(SySy,Sy,format='csr')
                SzSz = scipy.sparse.kron(SzSz,Sz,format='csr')
            else:
                SxSx = scipy.sparse.kron(SxSx,S0,format='csr')
                SySy = scipy.sparse.kron(SySy,S0,format='csr')
                SzSz = scipy.sparse.kron(SzSz,S0,format='csr')
#        print(bond,site,SxSx)
#        print(bond,site,SySy)
#        print(bond,site,SzSz)
#        Ham += Jxx * SxSx + Jyy * SySy + Jzz * SzSz
        Ham += np.real(Jxx * SxSx + Jyy * SySy + Jzz * SzSz)
#    print(Ham)
    return Ham

def main():
    np.set_printoptions(threshold=10000)

    args = parse_args()
    N = args.N
    Nbond = N
    print("N=",N)

    start = time.time()
    S0, Sx, Sy, Sz = make_spin()
#    print(S0, Sx, Sy, Sz)
    list_site1, list_site2, list_Jxx, list_Jyy, list_Jzz \
        = make_interaction_list(N,Nbond)
#    print(list_site1, list_site2, list_Jxx, list_Jyy, list_Jzz)
    end = time.time()
    print (end - start)

    start = time.time()
    HamCSR = make_hamiltonian(S0,Sx,Sy,Sz,N,Nbond,\
        list_site1,list_site2,list_Jxx,list_Jyy,list_Jzz)
    end = time.time()
    print (end - start)

    start = time.time()
#    ene,vec = scipy.sparse.linalg.eigs(HamCSR,k=5) # complex Ham
    ene,vec = scipy.sparse.linalg.eigsh(HamCSR,k=5) # real Ham
    end = time.time()
    print (end - start)
    print ("# energy:",ene[0],ene[1],ene[2],ene[3],ene[4])

if __name__ == "__main__":
    main()
