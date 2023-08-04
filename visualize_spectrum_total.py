import argparse
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--file', type=str, help='the file name of the targeted image')
    parser.add_argument('-d', '--directory', type=str, help='the directory name of targeted images ')
    parser.add_argument('-m', '--method', type=str, default='fft', help='the method for visualization')
    parser.add_argument('-o', '--output', type=str, default='output', help='the output directory')
    parser.add_argument('-r', '--radius', nargs='*', type=float, help='radius threshold for low and high pass filter')
    return parser.parse_args()


def generate_pass_filter(size, radius):    # size: (h, w)
    lpf = np.zeros(size)
    r = min(size) / 2. * radius
    for i in range(size[0]):
        for j in range(size[1]):
            if ((i - ((size[0] - 1) / 2)) ** 2 + (j - (size[1] - 1) / 2) ** 2 < r ** 2):
                lpf[i, j] = 1
    hpf = 1 - lpf
    return lpf, hpf


def img2fshift(img, method):
    fshift = None
    if method == 'fft':
        f = np.fft.fft2(img, axes=(0, 1))
        fshift = np.fft.fftshift(f)
    return fshift


def visualize_fshift(fshift, method):
    freq_img = None
    if method == 'fft':
        freq_img = np.clip(20 * np.log(np.abs(fshift) + 1), 0, 255)
    return freq_img


def fshift2img(fshift, method):
    iimg = None
    if method == 'fft':
        freq = np.fft.ifftshift(fshift)
        spatial = np.fft.ifft2(freq)
        iimg = np.abs(spatial)
    return iimg


def check_option(opt):
    try:
        assert(opt.file or opt.directory)
    except:
        print('Please specify the target file or directory.')
        sys.exit(-1)
    if opt.radius:
        try:
            for r in opt.radius:
                assert(r > 0 and r < 1)
        except:
            print('pass filter radius should between 0 and 1.')
            sys.exit(-1)


def check_output_folder(opt):
    if not os.path.exists(opt.output):
        os.makedirs(opt.output)


def grayscale_to_color(gray_img, colormap='jet'):
    # Normalize grayscale image to [0, 1]
    norm_img = (gray_img - np.min(gray_img)) / (np.max(gray_img) - np.min(gray_img))

    # Use colormap to create color image
    cmap = plt.get_cmap(colormap)
    color_img = cmap(norm_img)[:, :, :3]  # Keep only RGB channels, ignore alpha channel if present

    return (color_img * 255).astype(np.uint8)


def img_processing(img, opt, filename):
    result = dict()
    fshift = img2fshift(img, opt.method)
    result['freq'] = visualize_fshift(fshift, opt.method).astype(np.uint8)

    if opt.radius:
        for r in opt.radius:
            lpf, hpf = generate_pass_filter((img.shape[0], img.shape[1]), r)
            fshift_l = fshift * lpf
            fshift_h = fshift * hpf
            result['freq_lpf'] = visualize_fshift(fshift_l, opt.method).astype(np.uint8)
            result['freq_hpf'] = visualize_fshift(fshift_h, opt.method).astype(np.uint8)

    return result

if __name__ == '__main__':
    opt = parse_args()
    check_option(opt)
    check_output_folder(opt)

    if opt.file:
        img = np.array(Image.open(opt.file).convert('L'))
        img_processing(img, opt, opt.file)

    if opt.directory:
        work_directory = opt.directory
        files = os.listdir(work_directory)
        freq_imgs = None
        id = 0
        for file in tqdm(files):
            img = np.array(Image.open(os.path.join(work_directory, file)).convert('L'))
            result = img_processing(img, opt, file)
            if freq_imgs is None:
                freq_imgs = np.expand_dims(result['freq'], axis=0)
            else:
                try:
                    assert(result['freq'].shape[0] == freq_imgs.shape[1] and result['freq'].shape[1] == freq_imgs.shape[2])
                    freq_imgs = np.concatenate([freq_imgs, np.expand_dims(result['freq'], axis=0)], axis=0)
                except:
                    print('Warning: Unmatched Shapes Found. Expected ({0}, {1}). Got ({2}, {3}). Ignore the Exception.'.format(
                        freq_imgs.shape[1], freq_imgs.shape[2], result['freq'].shape[0], result['freq'].shape[1]
                    ))

        avg_freq = np.mean(freq_imgs, axis=0).astype(np.uint8)
        avg_freq_cmap = grayscale_to_color(avg_freq, colormap='jet')

        Image.fromarray(avg_freq).save(os.path.join(opt.output, 'average_spectrum.png'))
        Image.fromarray(avg_freq_cmap).save(os.path.join(opt.output, 'average_spectrum_colormap.png'))