import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

from numpy.lib.stride_tricks import as_strided as ast

# def block_view(A, block=(3, 3)):
#     """Provide a 2D block view to 2D array. No error checking made.
#     Therefore meaningful (as implemented) only for blocks strictly
#     compatible with the shape of A."""
#     # simple shape and strides computations may seem at first strange
#     # unless one is able to recognize the 'tuple additions' involved ;-)
#     shape = (A.shape[0]/ block[0], A.shape[1]/ block[1])+ block
#     strides = (block[0]* A.strides[0], block[1]* A.strides[1])+ A.strides
#     return ast(A, shape= shape, strides= strides)
#
#
# def ssim(img1, img2, C1=0.01**2, C2=0.03**2):
#
#     bimg1 = block_view(img1, (4,4))
#     bimg2 = block_view(img2, (4,4))
#     s1  = np.sum(bimg1, (-1, -2))
#     s2  = np.sum(bimg2, (-1, -2))
#     ss  = np.sum(bimg1*bimg1, (-1, -2)) + np.sum(bimg2*bimg2, (-1, -2))
#     s12 = np.sum(bimg1*bimg2, (-1, -2))
#
#     vari = ss - s1*s1 - s2*s2
#     covar = s12 - s1*s2
#
#     ssim_map =  (2*s1*s2 + C1) * (2*covar + C2) / ((s1*s1 + s2*s2 + C1) * (vari + C2))
#     return np.mean(ssim_map)
#
# def cal_ssim(im1, im2):
#     assert len(im1.shape) == 2 and len(im2.shape) == 2
#     assert im1.shape == im2.shape
#     mu1 = im1.mean()
#     mu2 = im2.mean()
#     sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
#     sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
#     sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
#     k1, k2, L = 0.01, 0.03, 255
#     C1 = (k1 * L) ** 2
#     C2 = (k2 * L) ** 2
#     C3 = C2 / 2
#     l12 = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
#     c12 = (2 * sigma1 * sigma2 + C2) / (sigma1 ** 2 + sigma2 ** 2 + C2)
#     s12 = (sigma12 + C3) / (sigma1 * sigma2 + C3)
#     ssim = l12 * c12 * s12
#     return ssim

def cal_mse(im1, im2):
    mse = (np.abs(im1 - im2) ** 2).mean()
    return mse

def cal_psnr(im1, im2):
    mse = (np.abs(im1 - im2) ** 2).mean()
    psnr = 10 * np.log10(255 * 255 / mse)
    return psnr

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calc_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
