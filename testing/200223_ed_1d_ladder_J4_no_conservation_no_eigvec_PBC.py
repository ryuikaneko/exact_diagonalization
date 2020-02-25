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
    parser = argparse.ArgumentParser(description='Calculate the ground state of S=1/2 Heisenberg ladder')
    parser.add_argument('-Lx', metavar='Lx',dest='Lx', type=int, default=8, help='set Lx (should be >=2)')
    parser.add_argument('-Ly', metavar='Ly',dest='Ly', type=int, default=2, help='set Ly=2')
#    parser.add_argument('-N', metavar='N',dest='N', type=int, default=12, help='set Nsize (should be >=4)')
#    parser.add_argument('-Jleg_xx', metavar='Jleg_xx',dest='Jleg_xx', type=float, default=1.0, help='set Jleg_xx')
#    parser.add_argument('-Jleg_zz', metavar='Jleg_zz',dest='Jleg_zz', type=float, default=1.0, help='set Jleg_zz')
#    parser.add_argument('-Jrung_xx', metavar='Jrung_xx',dest='Jrung_xx', type=float, default=1.0, help='set Jrung_xx')
#    parser.add_argument('-Jrung_zz', metavar='Jrung_zz',dest='Jrung_zz', type=float, default=1.0, help='set Jrung_zz')
    parser.add_argument('-Jleg', metavar='Jleg',dest='Jleg', type=float, default=1.0, help='set Jleg')
    parser.add_argument('-Jrung', metavar='Jrung',dest='Jrung', type=float, default=1.0, help='set Jrung')
    parser.add_argument('-Jising', metavar='Jising',dest='Jising', type=float, default=1.0, help='set Jising')
    parser.add_argument('-J4', metavar='J4',dest='J4', type=float, default=0.0, help='set J4')
#    parser.add_argument('-J4', metavar='J4',dest='J4', type=float, default=1.0, help='set J4')
    parser.add_argument('-OBC', metavar='OBC',dest='OBC', type=int, default=0, help='set open BC (OBC=1)')
#    parser.add_argument('-OBC', metavar='OBC',dest='OBC', type=int, default=1, help='set open BC (OBC=1)')
    return parser.parse_args()

def make_spin():
    S0 = scipy.sparse.csr_matrix(np.array([[1,0],[0,1]]))
    Sx = scipy.sparse.csr_matrix(0.5*np.array([[0,1],[1,0]]))
    Sy = scipy.sparse.csr_matrix(0.5*np.array([[0,-1j],[1j,0]]))
    Sz = scipy.sparse.csr_matrix(0.5*np.array([[1,0],[0,-1]]))
    return S0,Sx,Sy,Sz

def make_interaction_list(N,Jleg_xx,Jleg_zz,Jrung_xx,Jrung_zz,J4_xx,J4_zz,J4_xz,OBC):
#
# 0-2-4- ... -  2*i  - 2*i+2 - ...
# | | |          |       | 
# 1-3-5- ... - 2*i+1 - 2*i+3 - ...
#
    Lx = N//2
    Ly = 2
    Nbond1 = 0
    Nbond2 = 0
#
    list_site1 = []
    list_site2 = []
    list_site3 = []
    list_site4 = []
    list_site5 = []
    list_site6 = []
    list_Jxx = []
    list_Jzz = []
    list_J4xx = []
    list_J4zz = []
    list_J4xz = []
    for i in range(Lx):
        site0 = 2*i
        site1 = 2*i+1
        site2 = (2*i+2)%N
        site3 = (2*i+3)%N
        list_site1.append(site0)
        list_site2.append(site1)
        list_Jxx.append(Jrung_xx)
        list_Jzz.append(Jrung_zz)
        Nbond1 += 1
        list_site1.append(site0)
        list_site2.append(site2)
        if OBC == 1 and i==Lx-1:
            list_Jxx.append(0.0)
            list_Jzz.append(0.0)
        else:
            list_Jxx.append(Jleg_xx)
            list_Jzz.append(Jleg_zz)
        Nbond1 += 1
        list_site1.append(site1)
        list_site2.append(site3)
        if OBC == 1 and i==Lx-1:
            list_Jxx.append(0.0)
            list_Jzz.append(0.0)
        else:
            list_Jxx.append(Jleg_xx)
            list_Jzz.append(Jleg_zz)
        Nbond1 += 1
        list_site3.append(site0)
        list_site4.append(site2)
        list_site5.append(site1) 
        list_site6.append(site3)
        if OBC == 1 and i==Lx-1:
            list_J4xx.append(0.0)
            list_J4zz.append(0.0)
            list_J4xz.append(0.0)
        else:
            list_J4xx.append(J4_xx)
            list_J4zz.append(J4_zz)
            list_J4xz.append(J4_xz)
        Nbond2 += 1
    return Nbond1, Nbond2, \
        list_site1, list_site2, \
        list_site3, list_site4, list_site5, list_site6, \
        list_Jxx, list_Jzz, \
        list_J4xx, list_J4zz, list_J4xz

def make_hamiltonian(S0,Sx,Sy,Sz,N,\
    Nbond1, Nbond2, \
    list_site1, list_site2, \
    list_site3, list_site4, list_site5, list_site6, \
    list_Jxx, list_Jzz, \
    list_J4xx, list_J4zz, list_J4xz \
    ):
    Ham = scipy.sparse.csr_matrix((2**N,2**N),dtype=float)
#
    for bond in range(Nbond1):
        i1 = list_site1[bond]
        i2 = list_site2[bond]
        Jxx = list_Jxx[bond]
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
        Ham += np.real(Jxx * (SxSx + SySy) + Jzz * SzSz)
#
    for bond in range(Nbond2):
        i3 = list_site3[bond]
        i4 = list_site4[bond]
        i5 = list_site5[bond]
        i6 = list_site6[bond]
        J4xx = list_J4xx[bond]
        J4zz = list_J4zz[bond]
        J4xz = list_J4xz[bond]
        SxSxSxSx = 1
        SxSxSySy = 1
        SxSxSzSz = 1
        SySySxSx = 1
        SySySySy = 1
        SySySzSz = 1
        SzSzSxSx = 1
        SzSzSySy = 1
        SzSzSzSz = 1
        for site in range(N):
            if site==i3 or site==i4:
                SxSxSxSx = scipy.sparse.kron(SxSxSxSx,Sx,format='csr')
                SxSxSySy = scipy.sparse.kron(SxSxSySy,Sx,format='csr')
                SxSxSzSz = scipy.sparse.kron(SxSxSzSz,Sx,format='csr')
                SySySxSx = scipy.sparse.kron(SySySxSx,Sy,format='csr')
                SySySySy = scipy.sparse.kron(SySySySy,Sy,format='csr')
                SySySzSz = scipy.sparse.kron(SySySzSz,Sy,format='csr')
                SzSzSxSx = scipy.sparse.kron(SzSzSxSx,Sz,format='csr')
                SzSzSySy = scipy.sparse.kron(SzSzSySy,Sz,format='csr')
                SzSzSzSz = scipy.sparse.kron(SzSzSzSz,Sz,format='csr')
            elif site==i5 or site==i6:
                SxSxSxSx = scipy.sparse.kron(SxSxSxSx,Sx,format='csr')
                SxSxSySy = scipy.sparse.kron(SxSxSySy,Sy,format='csr')
                SxSxSzSz = scipy.sparse.kron(SxSxSzSz,Sz,format='csr')
                SySySxSx = scipy.sparse.kron(SySySxSx,Sx,format='csr')
                SySySySy = scipy.sparse.kron(SySySySy,Sy,format='csr')
                SySySzSz = scipy.sparse.kron(SySySzSz,Sz,format='csr')
                SzSzSxSx = scipy.sparse.kron(SzSzSxSx,Sx,format='csr')
                SzSzSySy = scipy.sparse.kron(SzSzSySy,Sy,format='csr')
                SzSzSzSz = scipy.sparse.kron(SzSzSzSz,Sz,format='csr')
            else:
                SxSxSxSx = scipy.sparse.kron(SxSxSxSx,S0,format='csr')
                SxSxSySy = scipy.sparse.kron(SxSxSySy,S0,format='csr')
                SxSxSzSz = scipy.sparse.kron(SxSxSzSz,S0,format='csr')
                SySySxSx = scipy.sparse.kron(SySySxSx,S0,format='csr')
                SySySySy = scipy.sparse.kron(SySySySy,S0,format='csr')
                SySySzSz = scipy.sparse.kron(SySySzSz,S0,format='csr')
                SzSzSxSx = scipy.sparse.kron(SzSzSxSx,S0,format='csr')
                SzSzSySy = scipy.sparse.kron(SzSzSySy,S0,format='csr')
                SzSzSzSz = scipy.sparse.kron(SzSzSzSz,S0,format='csr')
        Ham += np.real( \
            J4xx * ( SxSxSxSx + SxSxSySy + SySySxSx + SySySySy) + \
            J4zz * SzSzSzSz + \
            J4xz * ( SxSxSzSz + SySySzSz + SzSzSxSx + SzSzSySy) \
            )
#
    return Ham

def main():
#    np.set_printoptions(threshold=10000)
    np.set_printoptions(threshold=100,linewidth=1000,\
        formatter={'float': '{: 0.10f}'.format})

    args = parse_args()
#    N = args.N
    Lx = args.Lx
    Ly = args.Ly
    N = Lx*Ly
#    Jleg_xx = args.Jleg_xx
#    Jleg_zz = args.Jleg_zz
#    Jrung_xx = args.Jrung_xx
#    Jrung_zz = args.Jrung_zz
    Jleg = args.Jleg
    Jrung = args.Jrung
    Jising = args.Jising
    OBC = args.OBC
    J4 = args.J4
    Jleg_xx = Jleg
    Jleg_zz = Jleg*Jising
    Jrung_xx = Jrung
    Jrung_zz = Jrung*Jising
    J4_xx = J4
    J4_zz = J4
    J4_xz = J4
    print("N=",N)
    print("Lx=",Lx)
    print("Ly=",Ly)
    print("Jleg=",Jleg)
    print("Jrung=",Jrung)
    print("Jising=",Jising)
    print("J4=",J4)
    print("Jleg_xx=",Jleg_xx)
    print("Jleg_zz=",Jleg_zz)
    print("Jrung_xx=",Jrung_xx)
    print("Jrung_zz=",Jrung_zz)
    print("J4_xx=",J4_xx)
    print("J4_zz=",J4_zz)
    print("J4_xz=",J4_xz)
    print("OBC=",OBC)
    print("")

#    Lx = N//2
    print("lattice structure")
    print([2*i for i in range(Lx)])
    print([2*i+1 for i in range(Lx)])
    print("")

    start = time.time()
    S0, Sx, Sy, Sz = make_spin()
    Nbond1, Nbond2, \
        list_site1, list_site2, \
        list_site3, list_site4, list_site5, list_site6, \
        list_Jxx, list_Jzz, \
        list_J4xx, list_J4zz, list_J4xz \
        = make_interaction_list(N,Jleg_xx,Jleg_zz,Jrung_xx,Jrung_zz,J4_xx,J4_zz,J4_xz,OBC)
    print("N_2body=",Nbond1)
    print("site1=",list_site1)
    print("site2=",list_site2)
    print("Jxx=",list_Jxx)
    print("Jzz=",list_Jzz)
    print("N_4body=",Nbond2)
    print("site1=",list_site3)
    print("site2=",list_site4)
    print("site3=",list_site5)
    print("site4=",list_site6)
    print("J4xx=",list_J4xx)
    print("J4zz=",list_J4zz)
    print("J4xz=",list_J4xz)
    print("")
    end = time.time()
    print("time=",end - start)

    start = time.time()
    HamCSR = make_hamiltonian(S0,Sx,Sy,Sz,N,\
        Nbond1, Nbond2, \
        list_site1, list_site2, \
        list_site3, list_site4, list_site5, list_site6, \
        list_Jxx, list_Jzz, \
        list_J4xx, list_J4zz, list_J4xz)
#    print(HamCSR)
#    for i,j in zip(*HamCSR.nonzero()):
#        print((i,j), HamCSR[i,j])
#    print("Nhilbert:",2**N)
#    print("all elements in Hamitonian:",2**N*2**N)
#    print("nonzero elements in Hamiltonian:",np.count_nonzero(HamCSR.toarray()))
#    print("")
    end = time.time()
    print("time=",end - start)

    start = time.time()
#    Nene = 2**N-1
#    Nene = 5
    Nene = 20
#    Nene = 2
    Nstate = Nene
#    Nstate = 1
#    ene,vec = np.linalg.eigh(HamCSR.todense())
#    ene,vec = scipy.sparse.linalg.eigsh(HamCSR,which='SA',k=Nene)
    enetmp = scipy.sparse.linalg.eigsh(HamCSR,which='SA',k=min(Nene,2**N-1),return_eigenvectors=False)
    ene = np.sort(enetmp)
    end = time.time()
    print("time=",end - start)

    print("")
#    print("energy:",ene)
#    print("Nsite, energy per site:",N,ene/N)
#    ene = ene/N
#    print("# Nsite, energy per site:",N,ene[0],ene[1],ene[2],ene[3],ene[4])
#    print("# Nsite, energy per site:",N,' '.join(map(str,ene)))
    print("Nsite, Jleg, Jrung, Jising, J4, BC, energy:",\
        N,Jleg,Jrung,Jising,J4,OBC,str(ene).lstrip('[').rstrip(']'))
    print("Nsite, Jleg, Jrung, Jising, J4, BC, gap:",\
        N,Jleg,Jrung,Jising,J4,OBC,str(ene-ene[0]).lstrip('[').rstrip(']'))
#    print("Nsite, energy per site:",N,str(ene/N).lstrip('[').rstrip(']'))
#    print("Nsite, gap:",N,ene[1]-ene[0])
#    for i in range(Nstate):
##        print("eigenstate:",i,ene[i],vec[:,i].flatten())
#        print("state: {:2d}".format(i),end=" ")
#        print("{:+.4f}".format(ene[i]),end=" ")
#        print(vec[:,i].flatten())

if __name__ == "__main__":
    main()
