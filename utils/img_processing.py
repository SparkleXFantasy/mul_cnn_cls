import numpy as np
import torch
import torch.fft as fft
import torch.nn.functional as F
from io import BytesIO
from PIL import Image
from random import random, choice
from scipy.ndimage.filters import gaussian_filter

def img_post_processing(img, opt):
    img = np.array(img)
    if opt.post_processing.jpeg_enabled:
        if random() < opt.post_processing.jpeg.prob:
            quality = choice(opt.post_processing.jpeg.quality)
            img = pil_jpeg(img, quality)
    if opt.post_processing.gaussian_enabled:
        if random() < opt.post_processing.gaussian.prob:
            for i in range(3):
                gaussian_filter(img[:, :, i], output=img[:, :, i], sigma=opt.post_processing.gaussian.sigma)
    return Image.fromarray(img)


def pil_jpeg(img, quality):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=quality)
    img = Image.open(out)
    img = np.array(img)
    out.close()
    return img


def denormalize_t(img, mean, std):
    ''' Denormalizes a tensor of images.'''
    for t in range(img.shape[0]):
        img[t, :, :] = (img[t, :, :] * std[t]) + mean[t]
    return img


def denormalize_batch_t(img, mean, std):
    ''' Denormalizes a tensor of images.'''
    img_denorm = torch.empty_like(img)
    img_denorm[:, 0, :, :] = (img[:, 0, :, :].clone() * std[0]) + mean[0]
    img_denorm[:, 1, :, :] = (img[:, 1, :, :].clone() * std[1]) + mean[1]
    img_denorm[:, 2, :, :] = (img[:, 2, :, :].clone() * std[2]) + mean[2]
    return img_denorm


class FrequencyDecomposer:
    def generate_pass_filter(self, shape, radius):  # shape: [B, C, H, W]
        lpf = torch.zeros(shape)
        r = min(shape[2], shape[3]) / 2. * radius
        for i in range(shape[2]):
            for j in range(shape[3]):
                if ((i - ((shape[2] - 1) / 2)) ** 2 + (j - (shape[3] - 1) / 2) ** 2 < r ** 2):
                    lpf[:, :, i, j] = 1
        hpf = 1 - lpf
        return lpf, hpf

    def imgt2fshift(self, img_t, method='fft'):
        fshift = None
        if method == 'fft':
            f = fft.fft2(img_t, dim=(2, 3))
            fshift = fft.fftshift(f)
        return fshift

    def fshift2imgt(self, fshift, method='fft'):
        iimg = None
        if method == 'fft':
            freq = fft.ifftshift(fshift)
            spatial = fft.ifft2(freq, dim=(2, 3))
            iimg = torch.abs(spatial)
        return iimg

    def frequency_decomposition(self, img_t, radius, method='fft'):
        '''
        :param img: torch.tensor, [B, C, H, W]
        :param radius: radius threshold for lpf and hpf in frequency map
        :return: img after lpf, img after hpf
        '''
        fshift = self.imgt2fshift(img_t)
        lpf, hpf = self.generate_pass_filter(img_t.shape, radius)
        lpf, hpf = lpf.to(img_t.device), hpf.to(img_t.device)
        fshift_l = fshift * lpf
        fshift_h = fshift * hpf
        iimg_l = self.fshift2imgt(fshift_l)
        iimg_h = self.fshift2imgt(fshift_h)
        return iimg_l, iimg_h


class NoiseGenerator:
    def srm_generation(self, img_t):
        """
        :param image: [B, C, H, W]
        :return: noises
        """

        # srm kernel 1
        srm1 = np.zeros([5, 5]).astype('float32')
        srm1[1:-1, 1:-1] = np.array([[-1, 2, -1],
                                     [2, -4, 2],
                                     [-1, 2, -1]])
        srm1 /= 4.
        # srm kernel 2
        srm2 = np.array([[-1, 2, -2, 2, -1],
                         [2, -6, 8, -6, 2],
                         [-2, 8, -12, 8, -2],
                         [2, -6, 8, -6, 2],
                         [-1, 2, -2, 2, -1]]).astype('float32')
        srm2 /= 12.
        # srm kernel 3
        srm3 = np.zeros([5, 5]).astype('float32')
        srm3[2, 1:-1] = np.array([1, -2, 1])
        srm3 /= 2.

        srm = np.stack([srm1, srm2, srm3], axis=0)

        W_srm = np.zeros([3, 3, 5, 5]).astype('float32')

        for i in range(3):
            W_srm[i, 0, :, :] = srm[i, :, :]
            W_srm[i, 1, :, :] = srm[i, :, :]
            W_srm[i, 2, :, :] = srm[i, :, :]

        W_srm = torch.from_numpy(W_srm).to(img_t.device)

        srm_noise = F.conv2d(img_t, W_srm, padding=2)

        return srm_noise