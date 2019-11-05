#!/usr/bin/env python

# coding:utf-8
from __future__ import print_function
import math
import numpy as np
#
import sys

def num2bit(state,L):
    return np.binary_repr(state,L)

def bit2num(bit):
    return int(bit,2)

def get_spin(state,site):
    return (state>>site)&1

def shift_1spin_inv(state,L):
    first = get_spin(state,0)
    return (state>>1)|(first<<(L-1))

def test_1_shift_1spin(state,L):
    left = state<<1
    print("# left",num2bit(left,L))
    mask = (1<<L)-2
    print("# mask",num2bit(mask,L))
    left = left&mask
    print("# left&mask",num2bit(left,L))
    right = state>>(L-1)
    print("# right",num2bit(right,L))
    mask = 1
    print("# mask",num2bit(mask,L))
    right = right&mask
    print("# right&mask",num2bit(right,L))
    new = left|right
    print("# (left&mask)|(right&mask)",num2bit(new,L))
    return new

def test_2_shift_1spin(state,L):
    return ((state<<1)&(1<<L)-2)|((state>>(L-1))&1)

def test_1_shift_1spin_inv(state,L):
    left = state<<(L-1)
    print("# left",num2bit(left,L))
    mask = 1<<(L-1)
    print("# mask",num2bit(mask,L))
    left = left&mask
    print("# left&mask",num2bit(left,L))
    right = state>>1
    print("# right",num2bit(right,L))
    mask = (1<<(L-1))-1
    print("# mask",num2bit(mask,L))
    right = right&mask
    print("# right&mask",num2bit(right,L))
    new = left|right
    print("# (left&mask)|(right&mask)",num2bit(new,L))
    return new

def test_2_shift_1spin_inv(state,L):
    return ((state<<(L-1))&(1<<(L-1)))|((state>>1)&((1<<(L-1))-1))

def main():
    L = 6
    print("2**32",2**32)
    print("2**64",2**64)
    print("sys.maxint",sys.maxint)
    print("sys.maxsize",sys.maxsize)
    print()

    print("# shift_1spin_inv")
    nup = L//2
    first = (1<<(L-nup))-1
    t = first
    for i in range(L+1):
        print(i,t,num2bit(t,L))
        t = shift_1spin_inv(t,L)
    print()

    print("# test_1_shift_1spin")
    nup = L//2
    first = (1<<(L-nup))-1
    t = first
    for i in range(L+1):
        print(i,t,num2bit(t,L))
        t = test_1_shift_1spin(t,L)
    print()

    print("# test_2_shift_1spin")
    nup = L//2
    first = (1<<(L-nup))-1
    t = first
    for i in range(L+1):
        print(i,t,num2bit(t,L))
        t = test_2_shift_1spin(t,L)
    print()

    print("# test_1_shift_1spin_inv")
    nup = L//2
    first = (1<<(L-nup))-1
    t = first
    for i in range(L+1):
        print(i,t,num2bit(t,L))
        t = test_1_shift_1spin_inv(t,L)
    print()

    print("# test_2_shift_1spin_inv")
    nup = L//2
    first = (1<<(L-nup))-1
    t = first
    for i in range(L+1):
        print(i,t,num2bit(t,L))
        t = test_2_shift_1spin_inv(t,L)
    print()

if __name__ == "__main__":
    main()
