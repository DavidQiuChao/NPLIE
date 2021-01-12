#!/usr/bin/env python
#coding=utf-8

import cv2

def guideFilter(I,p,winsize,eps,s):
    # 输入图像高，宽
    h,w = I.shape[:2]

    # 缩小图像
    size = (int(round(w*1.0/s)),int(round(h*1.0/s)))
    small_I = cv2.resize(I,size,interpolation=cv2.INTER_NEAREST)
    small_p=  cv2.resize(p,size,interpolation=cv2.INTER_NEAREST)
    # 缩小滑动窗口
    #import pdb;pdb.set_trace()
    X = winsize[0]
    #small_winsize = (int(round(X*s)),int(round(X*s)))
    r = round((2*X+1)*1.0/s+1)
    small_winsize = (int(r),int(r))

    #I的均值平滑
    mean_small_I = cv2.blur(small_I,small_winsize)

    #p的均值平滑
    mean_small_p = cv2.blur(small_p,small_winsize)

    #I*I和p*p的均值平滑
    mean_small_II = cv2.blur(small_I*small_I,small_winsize)
    mean_small_Ip = cv2.blur(small_I*small_p,small_winsize)
    # 方差
    var_small_I = mean_small_II - mean_small_I*mean_small_I
    # 协方差
    cov_small_Ip = mean_small_Ip - mean_small_I*mean_small_p

    small_a = cov_small_Ip / (var_small_I + eps)
    small_b = mean_small_p - small_a*mean_small_I

    # 对a,b进行均值平滑
    mean_small_a = cv2.blur(small_a,small_winsize)
    mean_small_b = cv2.blur(small_b,small_winsize)

    # 放大
    size1 = (w,h)
    mean_a = cv2.resize(mean_small_a,size1,interpolation=cv2.INTER_LINEAR)
    mean_b = cv2.resize(mean_small_b,size1,interpolation=cv2.INTER_LINEAR)
    q = mean_a*I + mean_b
    return q
