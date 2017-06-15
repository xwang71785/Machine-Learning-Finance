# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 11:42:36 2017
Vision with Python
@author: wangx3
"""
import PIL.Image as img
import pylab
import scipy.ndimage.filters as flt
import scipy.ndimage.morphology as mpl
import scipy.ndimage.measurements as msm

source = img.open('girls.jpg')
gray = source.convert('L')

im = pylab.array(gray)
pylab.figure()
pylab.hist(im.flatten(), 128)
pylab.show()

im2 = flt.gaussian_filter(im, 10)

imx = pylab.zeros(im.shape)
flt.sobel(im, 1, imx)

imy = pylab.zeros(im.shape)
flt.sobel(im, 0, imy)

magnitude = pylab.sqrt(imx**2 + imy**2)


im = 1 * (im < 128)
im_open = mpl.binary_opening(im, pylab.ones((64, 4)), iterations=2)
labels, nbr = msm.label(im_open)
print('Number of objects: ', nbr)
pylab.imshow(im)






