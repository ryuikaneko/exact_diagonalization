#!/usr/bin/env python

# coding:utf-8
from __future__ import print_function
import math
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import argparse
import time
from numba import jit

def parse_args():
    parser = argparse.ArgumentParser(description='transverse field Ising')
    parser.add_argument('-L',metavar='L',dest='L',type=int,default=4,help='set L')
    return parser.parse_args()

def num2bit(state,L):
    return np.binary_repr(state,L)

def bit2num(bit):
    return int(bit,2)

@jit(nopython=True)
def get_spin(state,site):
    return (state>>site)&1

@jit(nopython=True)
def find_state(state,list_1,maxind):
    imin = 0
    imax = maxind-1
    i = imin
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

@jit(nopython=True)
def flip_1spin(state,i1):
    return state^(1<<i1)

@jit(nopython=True)
def flip_2spins(state,i1,i2):
    return state^((1<<i1)+(1<<i2))

@jit(nopython=True)
def make_basis(L):
    Nrep = 1<<L
    list_state = np.array([state for state in range(0,Nrep+1)],dtype=np.int64)
    return list_state, Nrep

@jit(nopython=True)
def make_hamiltonian_child(Nrep,Nbond,Ns,list_site1,list_site2,list_state):
#
#---- FM TFIsing model (spin: \sigma)
## sz.sz:         #elements = Nrep*1 (diagonal elements)
## sx:            #elements = Nrep*Ns
##
    listki = np.array([i for k in range(1+Ns) for i in range(Nrep)],dtype=np.int64)
    loc = np.zeros((1+Ns)*Nrep,dtype=np.int64)
    elemnt = np.zeros((1+Ns)*Nrep,dtype=np.float64)
#    Ham = np.zeros((Nrep,Nrep),dtype=np.float64)
    for a in range(Nrep):
        sa = list_state[a]
        for i in range(Nbond): ## Ising (- \sigma^z \sigma^z)
            i1 = list_site1[i]
            i2 = list_site2[i]
            loc[a] = a
            if get_spin(sa,i1) == get_spin(sa,i2):
#                Ham[a,a] -= 1.0
                elemnt[a] -= 1.0
            else:
#                Ham[a,a] += 1.0
                elemnt[a] += 1.0
        for i in range(Ns): ## Transverse field (- \sigma^x = -2 S^x = - S^+ - S^-)
            bb = flip_1spin(sa,i)
            b = find_state(bb,list_state,Nrep)
#            Ham[a,b] -= 1.0
            elemnt[(1+i)*Nrep+a] -= 1.0
            loc[(1+i)*Nrep+a] = b
#---- end of FM TFIsing model (spin: \sigma)
#
    return elemnt, listki, loc

def make_hamiltonian(Nrep,elemnt,listki,loc):
    return scipy.sparse.csr_matrix((elemnt,(listki,loc)),shape=(Nrep,Nrep),dtype=np.float64)


def main():
    args = parse_args()
    L = args.L
    Ns = L

    start = time.time()
    print("## make basis: sz not conserved")
    list_state, Nrep = make_basis(L)
    print("# L",L)
    print("# Ns",Ns)
    print("# Nrep",Nrep)
#    for i in range(Nrep):## show all states
    for i in range(0,Nrep,Nrep-1):## show first and last states
        print("# i list_state bit",i,list_state[i],num2bit(list_state[i],L))
    end = time.time()
    print("## time:",end-start)
    print()

    start = time.time()
    print("## make interactions")
    Nbond = L
    list_site1 = np.array([i for i in range(Nbond)],dtype=np.int64)
    list_site2 = np.array([(i+1)%L for i in range(Nbond)],dtype=np.int64)
    print("# list_site1=",list_site1)
    print("# list_site2=",list_site2)
    end = time.time()
    print("## time:",end-start)
    print()

    start = time.time()
    print("## make Hamiltonian")
    elemnt, listki, loc = make_hamiltonian_child(Nrep,Nbond,Ns,list_site1,list_site2,list_state)
    Ham = make_hamiltonian(Nrep,elemnt,listki,loc)
    print("# Ham",Ham)
    end = time.time()
    print("## time:",end-start)
    print()

    start = time.time()
    print("## diag Hamiltonian")
    Neig = 1
    if Nrep < 100:
        print("# Nrep < 100: use scipy.linalg.eigh")
        ene,vec = scipy.linalg.eigh(Ham.todense(),eigvals=(0,min(Neig-1,Nrep-1)))
    else:
        print("# Nrep >= 100: use scipy.sparse.linalg.eigsh")
        ene,vec = scipy.sparse.linalg.eigsh(Ham,which='SA',k=min(Neig,Nrep-1))
#        ene = scipy.sparse.linalg.eigsh(Ham,which='SA',k=min(Neig,Nrep-1),return_eigenvectors=False)
    idx = ene.argsort()
    ene = ene[idx]
    vec = vec[:,idx]
    print("# ene",ene[0])
    print("# vec",vec[:,0])
    end = time.time()
    print("## time:",end-start)
    print()

if __name__ == "__main__":
    main()
