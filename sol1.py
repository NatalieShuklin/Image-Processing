import matplotlib.pyplot as plt
import numpy as np
from imageio import imread, imwrite
from skimage.color import rgb2gray

GRAYSCALE = 1
RGB = 2
RGB_YIQ_TRANSFORMATION_MATRIX = np.array([[0.299, 0.587, 0.114],
                                          [0.596, -0.275, -0.321],
                                          [0.212, -0.523, 0.311]])


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


def imdisplay(filename, representation):
    """
    Reads an image and displays it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    """
    im = read_image(filename, representation)
    if representation == GRAYSCALE:
        plt.imshow(im, cmap=plt.cm.gray)
    else:
        plt.imshow(im)
    plt.show()


def rgb2yiq(imRGB):
    """
    Transform an RGB image into the YIQ color space
    :param imRGB: height X width X 3 np.float64 matrix in the [0,1] range
    :return: the image in the YIQ space
    """
    transpose = RGB_YIQ_TRANSFORMATION_MATRIX.transpose()
    img2yiq = imRGB.dot(transpose)
    return img2yiq


def yiq2rgb(imYIQ):
    """
    Transform a YIQ image into the RGB color space
    :param imYIQ: height X width X 3 np.float64 matrix in the [0,1] range for
        the Y channel and in the range of [-1,1] for the I,Q channels
    :return: the image in the RGB space
    """
    transpose = np.linalg.inv(RGB_YIQ_TRANSFORMATION_MATRIX.transpose())
    img2rgb = imYIQ.dot(transpose)
    return img2rgb

def linear_stretch(hist):
    factor = 255 / hist.max() - hist.min()
    return hist * factor


def histogram_rgb(im_orig):
    yiq = rgb2yiq(im_orig)
    y = (255 * yiq[:, :, 0])
    y = y.round().astype(np.int)
    # step 1
    hist_orig = np.histogram(y, bins=256, range= [0,255])[0]
    # step 2
    C_hist = np.cumsum(hist_orig)
    # step 3 - 7
    m = np.argmax(C_hist > 0)
    C_m = C_hist[m]
    C_255 = C_hist[255]
    norm = C_hist * 255 / C_255
    # check stretch
    if np.max(norm) != 255 or np.min(norm) != 0:
        norm_factor = 255 / ( C_255 - C_m)
        T = (norm_factor * (C_hist - C_m)).round().astype(np.int)
    else:
        T = norm.round().astype(np.int)

    map_y_channel = (T[y] / 255)
    yiq[:,:,0] = map_y_channel
    hist_eq = np.histogram((255 * yiq[:, :, 0]).round().astype(np.int),bins=256,range = [0,255])[0]
    # now after we mapped y channel, convert back to rgb
    img_eq = yiq2rgb(yiq)
    return [img_eq, hist_orig, hist_eq]


def histogram_equalize(im_orig):
    """
    Perform histogram equalization on the given image
    :param im_orig: Input float64 [0,1] image
    :return: [im_eq, hist_orig, hist_eq]
    """
    # check image colors
    if im_orig.ndim == 3: # rgb
        return histogram_rgb(im_orig)
    else:
        im = (255 * im_orig)
        im = im.round().astype(np.int)
        hist_orig = np.histogram(im, bins=256, range=[0, 255])[0]
        C_hist = np.cumsum(hist_orig)

        m = np.argmax(C_hist > 0)
        C_m = C_hist[m]
        C_255 = C_hist[255]
        norm = C_hist * 255 / C_255

        if np.max(norm) != 255 or np.min(norm) != 0:
            norm_factor = 255 / (C_255 - C_m)
            T = (norm_factor * (C_hist - C_m)).round().astype(np.int)
        else:
            T = norm.round().astype(np.int)

        img_eq = (T[im] / 255).astype(np.float64)
        hist_eq = np.histogram((255 * img_eq).round().astype(np.int), bins=256, range=[0, 255])
        return [img_eq, hist_orig, hist_eq]



def get_z(hist, n_quant):
    c_hist = np.cumsum(hist)
    per_q = (c_hist[255]//n_quant)
    list_z = [0]
    for i in range(1,n_quant):
        list_z.append(np.argmin(c_hist < i *per_q))
    list_z.append(255)
    return np.array(list_z).astype(int)

def get_error(n_quant,z,hist):
    err = []
    error = 0
    for i in range(n_quant):
        if i != n_quant-1:
            ar_n = np.arange(z[i], z[i + 1])
            ar_l = 2 * (z[i]-ar_n)
            error = error + np.sum(hist[z[i]:z[i + 1]]*ar_l)
    return error

def apply_vals_on_hist(hist,z,q):
    for i in range(z.size-1):
        if i + 2 != z.size:
            hist[z[i]:z[i + 1]] = q[i]
        else:
            hist[z[i]:z[i + 1] + 1] = q[i]
    return hist

def quantize(im_orig, n_quant, n_iter):
    """
    Performs optimal quantization of a given greyscale or RGB image
    :param im_orig: Input float64 [0,1] image
    :param n_quant: Number of intensities im_quant image will have
    :param n_iter: Maximum number of iterations of the optimization
    :return:  im_quant - is the quantized output image
              error - is an array with shape (n_iter,) (or less) of
                the total intensities error for each iteration of the
                quantization procedure
    """
    if im_orig.ndim == 3:
        yiq = rgb2yiq(im_orig)
        im = (255 * yiq[:,:,0]).round().astype(np.int)
    else:
        im = (255 * im_orig).round().astype(int)

    hist = np.histogram(im, bins=256, range=[0,255])[0]
    z = get_z(hist,n_quant)

    error = []
    q_l = []
    for i in range(z.size - 1):
        if i+2 != z.size:
            a = hist[z[i]:z[i + 1]]
            b = np.arange(z[i], z[i + 1])
        else:
            a = hist[z[i]:z[i + 1] + 1]
            b = np.arange(z[i], z[i + 1] + 1)
        q_1 = np.sum(a)
        q_2 = np.sum(a*b)
        q_l.append((q_2//q_1))
    q = np.array(q_l).astype(int)

    for i in range(0,n_iter):
        n_z = [0]
        for j in range(1,n_quant):
            n_z.append((q[j - 1] + q[j]) / 2)
        n_z.append(255)

        n_z2 = np.array(n_z).astype(int)
        if np.array_equal(n_z2, n_z):
            break
        z = n_z2

        n_q = []
        for i in range(z.size-1):
            if (i + 2) != z.size:
                a_1 = hist[z[i]:z[i + 1]]
                b_1 = np.arange(z[i], z[i + 1])
            else:
                a_1 = hist[z[i]:z[i + 1] + 1]
                b_1 = np.arange(z[i], z[i + 1] + 1)
            q_11 = np.sum(a)
            q_21 = np.sum(a * b)
            n_q.append((q_21 // q_11))
        q = np.array(n_q).astype(int)

        error.append(get_error(n_quant,z,hist))
    err =  np.array(error)

    hist = apply_vals_on_hist(hist,z,q)

    if im_orig.ndim == 3:
        yiq[:,:,0] = hist[im]/255
        return [yiq2rgb(yiq),err]
    else:
        return [hist[im]/255, err]




def quantize_rgb(im_orig, n_quant):  # Bonus - optional
    """
    Performs optimal quantization of a given greyscale or RGB image
    :param im_orig: Input RGB image of type float64 in the range [0,1]
    :param n_quant: Number of intensities im_quant image will have
    :return:  im_quant - the quantized output image
    """
    pass

