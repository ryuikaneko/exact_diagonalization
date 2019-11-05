#!/usr/bin/env python

# coding:utf-8
from __future__ import print_function
import math
import numpy as np
#
#import sys

def num2bit(state,L):
    return np.binary_repr(state,L)

def bit2num(bit):
    return int(bit,2)

def test_shift_child_BA2AB(s,a,b):
    left = s>>a
    print("# left",num2bit(left,a+b))
    mask = (1<<b)-1
    print("# mask",num2bit(mask,a+b))
    left = left&mask
    print("# left",num2bit(left,a+b))
    right = s<<b
    print("# right",num2bit(right,a+b))
    mask = ((1<<a)-1)<<b
    print("# mask",num2bit(mask,a+b))
    right = right&mask
    print("# right",num2bit(right,a+b))
    new = left|right
    print("# new",num2bit(new,a+b))
    return new

def shift_child_BA2AB(s,a,b):
    return ((s<<b)&(((1<<a)-1)<<b))|((s>>a)&((1<<b)-1))

def test_shift_child_CBA2CAB(s,a,b,c):
    mask = ((1<<c)-1)<<(a+b)
    print("# mask",num2bit(mask,a+b+c))
    left = s&mask
    print("# left",num2bit(left,a+b+c))
    mask = (1<<(a+b))-1
    print("# mask",num2bit(mask,a+b+c))
    right = s&mask
    print("# right",num2bit(right,a+b+c))
    right = shift_child_BA2AB(right,a,b)
    print("# right",num2bit(right,a+b+c))
    new = left|right
    print("# new",num2bit(new,a+b+c))
    return new

def shift_child_CBA2CAB(s,a,b,c):
    left = s&(((1<<c)-1)<<(a+b))
    right = s&((1<<(a+b))-1)
    right = shift_child_BA2AB(right,a,b)
    return left|right

def test_shift_child_DCBA2DBCA(s,a,b,c,d):
    mask = ((1<<d)-1)<<(a+b+c)
    print("# mask",num2bit(mask,a+b+c+d))
    left = s&mask
    print("# left",num2bit(left,a+b+c+d))
    mask = (1<<a)-1
    print("# mask",num2bit(mask,a+b+c+d))
    right = s&mask
    print("# right",num2bit(right,a+b+c+d))
    mid = s>>a
    print("# mid",num2bit(mid,a+b+c+d))
    mask = (1<<(b+c))-1
    print("# mask",num2bit(mask,a+b+c+d))
    mid = mid&mask
    print("# mid",num2bit(mid,a+b+c+d))
    mid = shift_child_BA2AB(mid,b,c)
    print("# mid",num2bit(mid,a+b+c+d))
    mid = mid<<a
    print("# mid",num2bit(mid,a+b+c+d))
    new = left|(mid|right)
    print("# new",num2bit(new,a+b+c+d))
    return new

def shift_child_DCBA2DBCA(s,a,b,c,d):
    left = s&(((1<<d)-1)<<(a+b+c))
    right = s&((1<<a)-1)
    mid = (s>>a)&((1<<(b+c))-1)
    mid = shift_child_BA2AB(mid,b,c)
    mid = mid<<a
    return left|(mid|right)

def shift_y_1spin(state,Lx,Ly,Ns):
    return shift_child_BA2AB(state,Ns-Lx,Lx)

#def test_shift_x_1spin(state,Lx,Ly,Ns):
#    s = shift_child_CBA2CAB(state,Lx-1,1,Ns-Lx)
#    for n in range(1,Ly):
#        s = shift_child_DCBA2DBCA(s,n*Lx,Lx-1,1,Ns-(n+1)*Lx)
#    return s

#def test_shift_x_1spin(state,Lx,Ly,Ns):
#    for n in range(Ly):
#        state = shift_child_DCBA2DBCA(state,n*Lx,Lx-1,1,Ns-(n+1)*Lx)
#    return state

def shift_x_1spin(state,Lx,Ly,Ns):
    s = shift_child_CBA2CAB(state,Lx-1,1,Ns-Lx)
    for n in range(1,Ly):
        s = shift_child_DCBA2DBCA(s,n*Lx,Lx-1,1,Ns-(n+1)*Lx)
    return s

#def shift_x_1spin(state,Lx,Ly,Ns):
#    for n in range(Ly):
#        state = shift_child_DCBA2DBCA(state,n*Lx,Lx-1,1,Ns-(n+1)*Lx)
#    return state


def main():
    Lx = 4
    Ly = 3
    Ns = Lx*Ly

    print("# shift_y_1spin")
#    nup = 4
    nup = 5
    first = (1<<(Ns-nup))-1
    t = first
    for i in range(Ly+1):
        print(i,t,num2bit(t,Ns))
        t = shift_y_1spin(t,Lx,Ly,Ns)
    print()

##    nup = 4
#    nup = Ns-1
#    first = (1<<(Ns-nup))-1
#    t = first
#    print(t,num2bit(t,Ns))
##    t = test_shift_child_CBA2CAB(t,Lx-1,1,Ns-Lx)
#    t = shift_child_CBA2CAB(t,Lx-1,1,Ns-Lx)
#    print(t,num2bit(t,Ns))
#    print()

##    nup = Ns-1
#    nup = 5
#    first = (1<<(Ns-nup))-1
#    t = first
#    print(t,num2bit(t,Ns))
##    t = test_shift_child_DCBA2DBCA(t,Lx,Lx-1,1,Ns-2*Lx)
#    t = shift_child_DCBA2DBCA(t,Lx,Lx-1,1,Ns-2*Lx)
#    print(t,num2bit(t,Ns))
#    print()

##    nup = Ns-1
#    nup = 5
#    first = (1<<(Ns-nup))-1
#    t = first
#    print(t,num2bit(t,Ns))
##    t = test_shift_x_1spin(t,Lx,Ly,Ns)
#    t = shift_x_1spin(t,Lx,Ly,Ns)
#    print(t,num2bit(t,Ns))
#    print()

    print("# shift_x_1spin")
#    nup = 9
    nup = 5
#    nup = 1
    first = (1<<(Ns-nup))-1
    t = first
    for i in range(Lx+1):
        print(i,t,num2bit(t,Ns))
        t = shift_x_1spin(t,Lx,Ly,Ns)
    print()

if __name__ == "__main__":
    main()
