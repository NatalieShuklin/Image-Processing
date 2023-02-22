import numpy as np
from imageio import imread
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy.signal import convolve
from imageio import imread
from skimage.color import rgb2gray
import os

GRAYSCALE = 1

def read_image(filename, representation):
    """
    Reads an image and converts it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    :return: Returns the image as an np.float64 matrix normalized to [0,1]
    """
    im = imread(filename)
    im_float = im.astype(np.float64)  # type change
    im_float /= 255
    if representation == GRAYSCALE:
        im_g = rgb2gray(im_float)
        return im_g
    return im_float


def reduce(im, blur_filter):
    """
    Reduces an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the downsampled image
    """
    blurr_im = ndimage.convolve(im, blur_filter, mode='constant', cval=0.0)
    blurr_im = ndimage.convolve(im, blur_filter.T, mode='constant', cval=0.0)
    sampled_im = blurr_im[::2, ::2]
    return sampled_im


def expand(im, blur_filter):
    """
    Expand an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the expanded image
    """
    r, c = im.shape
    pad_im = np.zeros((2 * r, 2 * c), np.float64)
    pad_im[::2, ::2] = im
    pad_im = ndimage.convolve(pad_im, blur_filter, mode='constant', cval=0.0)
    pad_im = ndimage.convolve(pad_im, blur_filter.T, mode='constant', cval=0.0)
    return pad_im


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Builds a gaussian pyramid for a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
            (an odd scalar that represents a squared filter)
            to be used in constructing the pyramid filter
    :return: pyr, filter_vec. Where pyr is the resulting pyramid as a
            standard python array with maximum length of max_levels,
            where each element of the array is a grayscale image.
            and filter_vec is a row vector of shape (1, filter_size)
            used for the pyramid construction.
    """
    pyramid = [im]
    gauss = im
    vec = np.array([1, 1])
    G_f = np.asarray([1])
    for i in range(filter_size - 1):
        G_f = np.convolve(G_f, vec)
    G_f = (1 / (2 ** (filter_size - 1))) * G_f.reshape((1, filter_size))

    for i in range(1, max_levels):
        if np.shape(gauss)[0] == 16 and np.shape(gauss)[1] == 16:
            break
        else:
            gauss = reduce(gauss, G_f)
            pyramid.append(gauss)
    return pyramid, G_f


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    Builds a laplacian pyramid for a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
            (an odd scalar that represents a squared filter)
            to be used in constructing the pyramid filter
    :return: pyr, filter_vec. Where pyr is the resulting pyramid as a
            standard python array with maximum length of max_levels,
            where each element of the array is a grayscale image.
            and filter_vec is a row vector of shape (1, filter_size)
            used for the pyramid construction.
    """
    pyr, filter = build_gaussian_pyramid(im, max_levels, filter_size)
    laplacian = []
    for i in range(len(pyr) - 1):
        laplacian.append(pyr[i] - (expand(pyr[i + 1], 2 * filter )))
    laplacian.append(pyr[-1])
    return laplacian, filter


def laplacian_to_image(lpyr, filter_vec, coeff):
    """

    :param lpyr: Laplacian pyramid
    :param filter_vec: Filter vector
    :param coeff: A python list in the same length as the number of levels in
            the pyramid lpyr.
    :return: Reconstructed image
    """
    l = lpyr[-1]
    for i in range(len(lpyr)):
        lpyr[i] *= coeff[i]
    for i in reversed(range(len(lpyr) - 1)):
        filt = 2 * filter_vec
        l = lpyr[i] + expand(l, filt)
    return l

def stretch_im(im):
        """
        stretch image to [0,1]
        :return: stretched image
        """
        return ((im - im.min() )/ (im.max() -  im.min()))

def render_pyramid(pyr, levels):
        """
        Render the pyramids as one large image with 'levels' smaller images
            from the pyramid
        :param pyr: The pyramid, either Gaussian or Laplacian
        :param levels: the number of levels to present
        :return: res a single black image in which the pyramid levels of the
                given pyramid pyr are stacked horizontally.
        """
        len_m = min(len(pyr),levels)
        w = (0,0)
        render = stretch_im(pyr[0])
        for i in range(1, len_m):
            stretch = stretch_im(pyr[i])
            h = render.shape[0] - pyr[i].shape[0]
            z = np.pad(stretch, ((0, h), w), mode='constant')
            render = np.hstack((render, z))
        return render

def display_pyramid(pyr, levels):
        """
        display the rendered pyramid
        """

        im_show = render_pyramid(pyr, levels)
        plt.imshow(im_show, cmap='gray')
        plt.show()

def pyramid_blending(im1, im2, mask, max_levels, filter_size_im,
                         filter_size_mask):
        """
         Pyramid blending implementation
        :param im1: input grayscale image
        :param im2: input grayscale image
        :param mask: a boolean mask
        :param max_levels: max_levels for the pyramids
        :param filter_size_im: is the size of the Gaussian filter (an odd
                scalar that represents a squared filter)
        :param filter_size_mask: size of the Gaussian filter(an odd scalar
                that represents a squared filter) which defining the filter used
                in the construction of the Gaussian pyramid of mask
        :return: the blended image
        """
        # step 1
        L1, f1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)
        L2, f2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)

        # step 2
        mask = mask.astype(np.float64)
        G_m, f_m = build_gaussian_pyramid(mask, max_levels, filter_size_mask)

        # step 3
        laplacian = []
        for k in range(max_levels):
            laplacian.append((G_m[k] * L1[k]) + ((1 - G_m[k]) * L2[k]))

        # step 4
        im_blended = laplacian_to_image(laplacian, f1, np.ones(max_levels))
        return np.clip(im_blended, a_min=0, a_max=1)

def relpath(filename):
        return os.path.join(os.path.dirname(__file__), filename)

def blending_example1():
        """
        Perform pyramid blending on two images RGB and a mask
        :return: image_1, image_2 the input images, mask the mask
            and out the blended image
        """
        im1 = read_image(relpath('externals/Q1.jpg'), 2)
        im2 = read_image(relpath('externals/Q2.jpg'), 2)
        mask = read_image(relpath('externals/QM.jpg'), 1)
        mask = np.round(mask).astype(np.bool)

        red1, green1, blue1 = im1[:, :, 0], im1[:, :, 1], im1[:, :, 2]
        red2, green2, blue2 = im2[:, :, 0], im2[:, :, 1], im2[:, :, 2]

        r = pyramid_blending(red2, red1, mask,3, 5, 5)
        g = pyramid_blending(green2, green1,mask, 3, 5, 5)
        b = pyramid_blending(blue2, blue1,mask, 3, 5,5)
        res = np.dstack((r, g, b))

        f, t = plt.subplots(nrows=2, ncols=2)
        t[0][0].imshow(im1)
        t[0][1].imshow(im2)
        t[1][0].imshow(mask, cmap='gray')
        t[1][1].imshow(res)

        plt.show()
        return im1, im2, mask, res

def blending_example2():
        """
        Perform pyramid blending on two images RGB and a mask
        :return: image_1, image_2 the input images, mask the mask
            and out the blended image
        """
        im1 = read_image(relpath('externals/jack.jpg'), 2)
        im2 = read_image(relpath('externals/m2.jpg'), 2)
        mask = read_image(relpath('externals/maskk.jpg'), 1)
        mask = np.round(mask).astype(np.bool)

        red1, green1, blue1 = im1[:, :, 0], im1[:, :, 1], im1[:, :, 2]
        red2, green2, blue2 = im2[:, :, 0], im2[:, :, 1], im2[:, :, 2]

        r = pyramid_blending(red2, red1, mask, 3, 5, 5)
        g = pyramid_blending(green2, green1, mask, 3, 5, 5)
        b = pyramid_blending(blue2, blue1, mask, 3, 5, 5)
        res = np.dstack((r, g, b))

        f, t = plt.subplots(nrows=2, ncols=2)
        t[0][0].imshow(im1)
        t[0][1].imshow(im2)
        t[1][0].imshow(mask, cmap='gray')
        t[1][1].imshow(res)

        plt.show()
        return im1, im2, mask, res

