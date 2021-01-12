#!/usr/bin/env python 
#coding=utf-8


import cv2
import imageio
import numpy as np
import fastGuideFilter as fgf


def normValue(im):
    maxV = np.max(im)
    minV = np.min(im)
    im = (im-minV)/(maxV-minV+1e-6)
    return im


def imgrad(im):
    gv = cv2.Sobel(im,cv2.CV_64F,1,0)
    gh = cv2.Sobel(im,cv2.CV_64F,0,1)
    g = 0.5*abs(gv)+0.5*abs(gh)
    return g



def psf2otf(psf,outSize):
    '''
    code is from https://blog.csdn.net/
    weixin_43890288/article/details/
    105676416
    '''
    psfSize = np.array(psf.shape)
    outSize = np.array(outSize)
    padSize = outSize - psfSize
    psf = np.pad(psf,((0,padSize[0]),\
            (0,padSize[1])),'constant')
    for i in range(len(psfSize)):
        psf = np.roll(psf,-int(psfSize[i]/2),i)
    otf = np.fft.fftn(psf)
    nElem = np.prod(psfSize)
    nOps = 0
    for k in range(len(psfSize)):
        nffts = nElem / psfSize[k]
        nOps = nOps + psfSize[k]*np.log2(psfSize[k])*nffts
    if np.max(np.abs(np.imag(otf))) / np.max(np.abs(otf))\
            <= nOps * np.finfo(np.float32).eps:
        otf = np.real(otf)
    return otf


def sign(x):
    xs = x.copy()
    xs[x>0]=1
    xs[x<0]=-1
    return xs


def shrink(x,lam):
    eps = 1e-10
    sig = sign(x)
    abs_x = abs(x)
    f = sig*np.maximum(abs_x-lam,0)
    return f


def updateI(I0,K,L,w,dx,dy):
    tmp = w*K-L
    fn0 = np.fft.fft2(2*I0)
    fn1 = np.conjugate(dx*tmp)\
        +np.conjugate(dy*tmp)
    fn = fn0+fn1
    fd = 2+w*(np.abs(dx)**2+np.abs(dy)**2)
    I = np.fft.ifft2(fn/fd)
    I = abs(I).astype(np.float64)
    return I


def getG(I):
    G = -np.log(imgrad(I)+1e-6)
    return G


def updateK(I,L,Jm,w,G,a,b):
    Fi = b*G/(2*a+w)
    x = 2*a*Jm+w*imgrad(I)+L
    K = shrink(x,Fi)
    return K


def updateLw(L,I,K,w,sigma):
    tmp = imgrad(I)-K
    L += w*tmp
    w = w*sigma
    return L,w


def initJm(im):
    gim = np.zeros(im.shape)
    for i in range(3):
        gim[...,i] = imgrad(im[...,i])
    return np.max(gim,2)


def optimizAlgo(I0,Jm,alpha=0.5,beta=0.1,ite=1):
    K = 0.
    L = 0.
    w = 1.
    sigma = 1.5
    G = getG(I0)

    H,W = I0.shape
    ky = np.array([[1,-1]])
    kx = np.array([[1],[-1]])
    dx = psf2otf(kx,(H+1,W))
    dy = psf2otf(ky,(H,W+1))
    dy = dy[:,1:]
    dx = dx[1:,:]

    for i in range(ite):
        I = updateI(I0,K,L,w,dx,dy)
        K = updateK(I,L,Jm,w,G,alpha,beta)
        L,w = updateLw(L,I,K,w,sigma)
    return I


def initIlluminunce(im):
    epsi = 0.025
    gs = 2
    H,W,C = im.shape
    bmin = 3
    bmax = int(np.round(np.minimum(H,W)/2.))
    bmean = int(np.round((bmin+bmax)/2.))
    ws = [bmin,bmean,bmax]
    ttI = np.zeros(im.shape)
    for i in range(3):
        I = np.zeros(im.shape)
        r = im[:,:,0]
        g = im[:,:,1]
        b = im[:,:,2]
        I[:,:,0] = fgf.guideFilter(r,r,[ws[i],ws[i]],epsi,gs)
        I[:,:,1] = fgf.guideFilter(g,g,[ws[i],ws[i]],epsi,gs)
        I[:,:,2] = fgf.guideFilter(b,b,[ws[i],ws[i]],epsi,gs)
        ttI[:,:,i] = np.max(I,2)
    return np.max(ttI,2)


def NPLIE(src):
    im = src.copy().astype(np.float32)
    im /= 255.
    I0 = initIlluminunce(im)
    Jm = initJm(im)
    I = optimizAlgo(I0,Jm,0.5,0.1,8)
    Ie = np.power(I,0.45)
    for i in range(3):
        R = im[:,:,i]/I
        im[:,:,i] = R*Ie
    im = np.minimum(1,np.maximum(0,im))
    im *= 255
    return im
