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
    parser = argparse.ArgumentParser(description='Calculate the ground state of S=1/2 Kitaev honeycomb')
    parser.add_argument('-Lx', metavar='Lx',dest='Lx', type=int, default=2, help='set Lx (should be >=2)')
    parser.add_argument('-Ly', metavar='Ly',dest='Ly', type=int, default=2, help='set Lx (should be >=2)')
    parser.add_argument('-Hx', metavar='Hx',dest='Hx', type=float, default=0.0, help='set field Hx')
    parser.add_argument('-Hy', metavar='Hy',dest='Hy', type=float, default=0.0, help='set field Hy')
    parser.add_argument('-Hz', metavar='Hz',dest='Hz', type=float, default=0.0, help='set field Hz')
    parser.add_argument('-Jxx', metavar='Jxx',dest='Jxx', type=float, default=1.0, help='set Ising interaction Jxx')
    parser.add_argument('-Jyy', metavar='Jyy',dest='Jyy', type=float, default=1.0, help='set Ising interaction Jyy')
    parser.add_argument('-Jzz', metavar='Jzz',dest='Jzz', type=float, default=1.0, help='set Ising interaction Jzz')
    return parser.parse_args()

def make_spin():
    S0 = scipy.sparse.csr_matrix(np.array([[1.0,0],[0,1.0]]))
    Sx = scipy.sparse.csr_matrix(np.array([[0,1],[1,0]])*0.5)
    Sy = scipy.sparse.csr_matrix(np.array([[0,-1j],[1j,0]])*0.5)
    Sz = scipy.sparse.csr_matrix(np.array([[1,0],[0,-1]])*0.5)
    return S0,Sx,Sy,Sz

def make_lattice(Lx,Ly,Lorb,Hx,Hy,Hz,Jxx,Jyy,Jzz):
    list_site0 = []
    list_site1 = []
    list_site2 = []
    list_Hx = []
    list_Hy = []
    list_Hz = []
    list_Jxx = []
    list_Jyy = []
    list_Jzz = []
    list_siteflux = []
    N1bond = 0
    N2bond = 0
    Nflux = 0
    site0 = np.zeros(Lorb,dtype=np.int)
    sitex = np.zeros(Lorb,dtype=np.int)
    sitey = np.zeros(Lorb,dtype=np.int)
    sitexy = np.zeros(Lorb,dtype=np.int)
    for iy in range(Ly):
        for ix in range(Lx):
            for iorb in range(Lorb):
                site0[iorb] = iorb + (ix + Lx*iy)*Lorb
                sitex[iorb] = iorb + ((ix+1)%Lx + Lx*iy)*Lorb
                sitex[iorb] = iorb + ((ix+1)%Lx + Lx*iy)*Lorb
                sitey[iorb] = iorb + (ix + Lx*((iy+1)%Ly))*Lorb
                sitexy[iorb] = iorb + ((ix+1)%Lx + Lx*((iy+1)%Ly))*Lorb
#
                list_site0.append(site0[iorb])
                list_Hx.append(Hx)
                list_Hy.append(Hy)
                list_Hz.append(Hz)
                N1bond += 1
#
            list_site1.append(site0[0])
            list_site2.append(site0[1])
            list_Jxx.append(0.0)
            list_Jyy.append(0.0)
            list_Jzz.append(Jzz)
            N2bond += 1
#
            list_site1.append(site0[1])
            list_site2.append(sitex[0])
            list_Jxx.append(Jxx)
            list_Jyy.append(0.0)
            list_Jzz.append(0.0)
            N2bond += 1
#
            list_site1.append(site0[1])
            list_site2.append(sitey[0])
            list_Jxx.append(0.0)
            list_Jyy.append(Jyy)
            list_Jzz.append(0.0)
            N2bond += 1
#
            ## z-y-x-z-y-x order
            list_siteflux.append([\
                site0[1],sitex[0],sitex[1],sitexy[0],sitey[1],sitey[0]])
            Nflux += 1
#
    return N1bond, N2bond, \
        np.array(list_site0), np.array(list_site1), np.array(list_site2), \
        np.array(list_Hx), np.array(list_Hy), np.array(list_Hz), \
        np.array(list_Jxx), np.array(list_Jyy), np.array(list_Jzz), \
        Nflux, np.array(list_siteflux)

def make_hamiltonian(S0,Sx,Sy,Sz,N,N1bond,N2bond,\
    list_site0,list_site1,list_site2,list_Hx,list_Hy,list_Hz,\
    list_Jxx,list_Jyy,list_Jzz):
    Ham = scipy.sparse.csr_matrix((2**N,2**N),dtype=complex)
#    Ham = scipy.sparse.csr_matrix((2**N,2**N),dtype=float)
    for bond in range(N1bond):
        i0 = list_site0[bond]
        par_Hx = list_Hx[bond]
        par_Hy = list_Hy[bond]
        par_Hz = list_Hz[bond]
        ham_Sx = 1
        ham_Sy = 1
        ham_Sz = 1
        for site in range(N):
            if site==i0:
                ham_Sx = scipy.sparse.kron(ham_Sx,Sx,format='csr')
                ham_Sy = scipy.sparse.kron(ham_Sy,Sy,format='csr')
                ham_Sz = scipy.sparse.kron(ham_Sz,Sz,format='csr')
            else:
                ham_Sx = scipy.sparse.kron(ham_Sx,S0,format='csr')
                ham_Sy = scipy.sparse.kron(ham_Sy,S0,format='csr')
                ham_Sz = scipy.sparse.kron(ham_Sz,S0,format='csr')
#        print(bond,site,ham_Sx)
        Ham += par_Hx * ham_Sx
        Ham += par_Hy * ham_Sy
        Ham += par_Hz * ham_Sz
    for bond in range(N2bond):
        i1 = list_site1[bond]
        i2 = list_site2[bond]
        par_Jxx = list_Jxx[bond]
        par_Jyy = list_Jyy[bond]
        par_Jzz = list_Jzz[bond]
        ham_SxSx = 1
        ham_SySy = 1
        ham_SzSz = 1
        for site in range(N):
            if site==i1 or site==i2:
                ham_SxSx = scipy.sparse.kron(ham_SxSx,Sx,format='csr')
                ham_SySy = scipy.sparse.kron(ham_SySy,Sy,format='csr')
                ham_SzSz = scipy.sparse.kron(ham_SzSz,Sz,format='csr')
            else:
                ham_SxSx = scipy.sparse.kron(ham_SxSx,S0,format='csr')
                ham_SySy = scipy.sparse.kron(ham_SySy,S0,format='csr')
                ham_SzSz = scipy.sparse.kron(ham_SzSz,S0,format='csr')
#        print(bond,site,ham_SzSz)
        Ham += par_Jxx * ham_SxSx
        Ham += par_Jyy * ham_SySy
        Ham += par_Jzz * ham_SzSz
#    print(Ham)
    return Ham

def make_op_flux(S0,Sx,Sy,Sz,N,list_siteflux,i):
    op_flux = 1
    for site in range(N):
        if site==list_siteflux[i,0]:
            op_flux = scipy.sparse.kron(op_flux,2*Sz,format='csr')
        elif site==list_siteflux[i,1]:
            op_flux = scipy.sparse.kron(op_flux,2*Sy,format='csr')
        elif site==list_siteflux[i,2]:
            op_flux = scipy.sparse.kron(op_flux,2*Sx,format='csr')
        elif site==list_siteflux[i,3]:
            op_flux = scipy.sparse.kron(op_flux,2*Sz,format='csr')
        elif site==list_siteflux[i,4]:
            op_flux = scipy.sparse.kron(op_flux,2*Sy,format='csr')
        elif site==list_siteflux[i,5]:
            op_flux = scipy.sparse.kron(op_flux,2*Sx,format='csr')
        else:
            op_flux = scipy.sparse.kron(op_flux,S0,format='csr')
    return op_flux

def main():
    np.set_printoptions(threshold=10000)

    args = parse_args()
    Lx = args.Lx
    Ly = args.Ly
    Hx = args.Hx
    Hy = args.Hy
    Hz = args.Hz
    Jxx = args.Jxx
    Jyy = args.Jyy
    Jzz = args.Jzz
    Lorb = 2
    N = Lx*Ly*Lorb
    print("N=",N)
    print("Hx=",Hx)
    print("Hy=",Hy)
    print("Hz=",Hz)
    print("|H|=",np.sqrt(Hx**2+Hy**2+Hz**2))
    print("Jxx=",Jxx)
    print("Jyy=",Jyy)
    print("Jzz=",Jzz)

    start = time.time()
    S0, Sx, Sy, Sz = make_spin()
#    print(S0, Sx, Sy, Sz)
    N1bond, N2bond, \
        list_site0, list_site1, list_site2, \
        list_Hx, list_Hy, list_Hz, \
        list_Jxx, list_Jyy, list_Jzz, \
        Nflux, list_siteflux \
        = make_lattice(Lx,Ly,Lorb,Hx,Hy,Hz,Jxx,Jyy,Jzz)
    print("N1bond=",N1bond)
    print("N2bond=",N2bond)
    print("list_site0=",list_site0)
    print("list_site1=",list_site1)
    print("list_site2=",list_site2)
    print("list_Hx=",list_Hx)
    print("list_Hy=",list_Hy)
    print("list_Hz=",list_Hz)
    print("list_Jxx=",list_Jxx)
    print("list_Jyy=",list_Jyy)
    print("list_Jzz=",list_Jzz)
    print("Nflux=",Nflux)
    print("list_siteflux=",list_siteflux)
    end = time.time()
    print(end - start)

    start = time.time()
    HamCSR = make_hamiltonian(S0,Sx,Sy,Sz,N,N1bond,N2bond,\
        list_site0,list_site1,list_site2,list_Hx,list_Hy,list_Hz,\
        list_Jxx,list_Jyy,list_Jzz)
    end = time.time()
    print(end - start)

    start = time.time()
#    ene,vec = np.linalg.eigh(HamCSR.todense())
#    ene,vec = scipy.linalg.eigh(HamCSR.todense())
    kmax=5
    ene,vec = scipy.sparse.linalg.eigsh(HamCSR,which='SA',k=kmax)
#    ene,vec = scipy.sparse.linalg.eigsh(HamCSR,which='SA',k=5)
#    ene,vec = scipy.sparse.linalg.eigsh(HamCSR,which='SA',k=10)
#    ene,vec = scipy.sparse.linalg.eigsh(HamCSR,which='SA',k=10,sigma=0.001)
#    ene,vec = scipy.sparse.linalg.eigsh(HamCSR,which='LM',k=10,sigma=0.001)
#    ene,vec = scipy.sparse.linalg.eigsh(HamCSR,which='LM',k=10)
#
## sort: https://stackoverflow.com/questions/8092920/sort-eigenvalues-and-associated-eigenvectors-after-using-numpy-linalg-eig-in-pyt
#    idx = ene.argsort()[::-1] # reverse order
    idx = ene.argsort()
    ene = ene[idx]
    vec = vec[:,idx]
#
    end = time.time()
    print(end - start)
#    print("# energy:",ene[0],ene[1],ene[2],ene[3],ene[4])
    print("# energy:",ene)
    enedens = ene/N
#    print("# energy density:",enedens[0],enedens[1],enedens[2],enedens[3],enedens[4])
    print("# energy density:",enedens)

    start = time.time()
    for k in range(kmax):
#    for k in range(5):
#    for k in range(10):
        flux = []
        for i in range(Lx*Ly):
            op_flux = make_op_flux(S0,Sx,Sy,Sz,N,list_siteflux,i)
#            print(i,op_flux)
#            flux.append(np.real(np.dot(vec[:,0].conj(),op_flux.dot(vec[:,0]))))
            flux.append(np.real(np.dot(vec[:,k].conj(),op_flux.dot(vec[:,k]))))
        print(k,flux)
    end = time.time()
    print(end - start)

if __name__ == "__main__":
    main()
