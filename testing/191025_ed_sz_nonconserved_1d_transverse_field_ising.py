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

def make_spin():
    S0 = scipy.sparse.csr_matrix(np.array([[1,0],[0,1]]))
    Sx = scipy.sparse.csr_matrix(np.array([[0,1],[1,0]]))
    Sy = scipy.sparse.csr_matrix(np.array([[0,-1j],[1j,0]]))
    Sz = scipy.sparse.csr_matrix(np.array([[1,0],[0,-1]]))
    return S0,Sx,Sy,Sz

def make_interaction_list(N,N1bond,N2bond,Hx,Jzz):
    list_site0 = np.array([i for i in range(N1bond)])
    list_site1 = np.array([i for i in range(N2bond)])
    list_site2 = np.array([(i+1)%N for i in range(N2bond)])
    list_Hx = np.ones(N1bond,dtype=float) * Hx
    list_Jzz = np.ones(N2bond,dtype=float)
    return list_site0, list_site1, list_site2, list_Hx, list_Jzz

def make_hamiltonian(S0,Sx,Sy,Sz,N,N1bond,N2bond,\
    list_site0,list_site1,list_site2,list_Hx,list_Jzz):
#    Ham = scipy.sparse.csr_matrix((2**N,2**N),dtype=complex)
    Ham = scipy.sparse.csr_matrix((2**N,2**N),dtype=float)
    for bond in range(N1bond):
        i0 = list_site0[bond]
        par_Hx = list_Hx[bond]
        ham_Sx = 1
        for site in range(N):
            if site==i0:
                ham_Sx = scipy.sparse.kron(ham_Sx,Sx,format='csr')
            else:
                ham_Sx = scipy.sparse.kron(ham_Sx,S0,format='csr')
#        print(bond,site,ham_Sx)
        Ham += par_Hx * ham_Sx
    for bond in range(N2bond):
        i1 = list_site1[bond]
        i2 = list_site2[bond]
        par_Jzz = list_Jzz[bond]
        ham_SzSz = 1
        for site in range(N):
            if site==i1 or site==i2:
                ham_SzSz = scipy.sparse.kron(ham_SzSz,Sz,format='csr')
            else:
                ham_SzSz = scipy.sparse.kron(ham_SzSz,S0,format='csr')
#        print(bond,site,ham_SzSz)
        Ham += par_Jzz * ham_SzSz
#    print(Ham)
    return Ham

def main():
    np.set_printoptions(threshold=10000)

    args = parse_args()
    N = args.N
    Hx = args.Hx
    Jzz = args.Jzz
    N1bond = N
    N2bond = N
    print("N=",N)
    print("N1bond=",N1bond)
    print("N2bond=",N2bond)
    print("Jzz=",Jzz)
    print("Hx=",Hx)

    start = time.time()
    S0, Sx, Sy, Sz = make_spin()
#    print(S0, Sx, Sy, Sz)
    list_site0, list_site1, list_site2, list_Hx, list_Jzz \
        = make_interaction_list(N,N1bond,N2bond,Hx,Jzz)
    print("list_site0=",list_site0)
    print("list_site1=",list_site1)
    print("list_site2=",list_site2)
    print("list_Hx=",list_Hx)
    print("list_Jzz=",list_Jzz)
    end = time.time()
    print(end - start)

    start = time.time()
    HamCSR = make_hamiltonian(S0,Sx,Sy,Sz,N,N1bond,N2bond,\
        list_site0,list_site1,list_site2,list_Hx,list_Jzz)
    end = time.time()
    print(end - start)

    start = time.time()
    ene,vec = scipy.sparse.linalg.eigsh(HamCSR,which='SA',k=5)
    end = time.time()
    print(end - start)
    print("# energy:",ene[0],ene[1],ene[2],ene[3],ene[4])

if __name__ == "__main__":
    main()
