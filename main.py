#!/usr/bin/env python
#coding=utf-8

import os
import numpy as np
import imageio
import nplie


def main(name):
    print(name)
    rim = imageio.imread(name)
    start = time.time()
    out = nplie.NPLIE(rim)
    end = time.time()
    cost = (end-start)
    print('fusion cost: {}'.format(cost))
    return out


if __name__ == "__main__":
    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--indir',help='input image sequence list')
    parser.add_argument('-o','--odir',help='result image name',\
            default='res')
    parser.add_argument('-f','--flist',help='specific test sample',\
            default=0)
    args = parser.parse_args()
    indir = args.indir
    odir = args.odir
    flist = args.flist
    names = os.listdir(indir)
    names.sort()
    if flist:
        with open(flist,'r') as f:
            flines = f.readlines()
            flines = [ele.strip()\
                    for ele in flines]
    for name in names:
        if flist and (os.path.splitext(name)[0] not in flines):
            continue
        oname = os.path.join(odir,name)
        name = os.path.join(indir,name)
        start = time.time()
        res = main(name)
        end = time.time()
        cost = (end-start)
        print('total cost: {}'.format(cost))
        imageio.imwrite(oname.replace('bmp','jpg'),res)
