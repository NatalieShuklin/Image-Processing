import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
from scipy import signal
from scipy.signal import convolve2d
from scipy.ndimage.interpolation import map_coordinates
from imageio import imread
from skimage.color import rgb2gray


SIZE = 0
GRAYSCALE = 1

def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T

def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), )
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec

def phase_vocoder(spec, ratio):
    num_timesteps = int(spec.shape[1] / ratio)
    time_steps = np.arange(num_timesteps) * ratio

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec


def DFT(signal):
    """
    transform a 1D discrete signal to its Fourier representation
    :param signal:  is an array of dtype float64 with shape (N,) or (N,1)
    :return:  complex Fourier signal
    """
    N = signal.shape[SIZE]
    temp = ((-2*np.pi)*(1j))/N
    omega = np.exp(temp)
    i , j = np.meshgrid(np.arange(N),np.arange(N))
    fourier_base =np.power(omega,i *j)
    return fourier_base.dot(signal).astype(np.complex128)

def IDFT(fourier_signal):
    """
    Inverse transform fourier into signal
    :param fourier_signal:  is an array of
    dtype complex128 with the same shape
    :return: complex signal
    """
    N = fourier_signal.shape[SIZE]
    temp = ((2 * np.pi) * (1j)) / N
    omega = np.exp(temp)
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    inverse_fourier = np.power(omega, i * j)
    inverse_fourier = inverse_fourier.dot(fourier_signal)/N
    return inverse_fourier.astype(np.complex128)

def DFT2(image):
    """
    convert a 2D discrete signal to its Fourier representation
    :param image: a grayscale image of dtype float64, of shape (M,N) or (M,N,1)
    :return: Fourier representation of the image
    """
    M = image.shape[0]
    N = image.shape[1]
    im = image.astype(np.complex128)

    for j in range(N):
        im[:,j] = DFT(im[:,j])
    for i in range(M):
        im[i,:] = DFT(im[i,:])
    return im

def IDFT2(fourier_image):
    """
    convert a Fourier representation to 2D discrete signal
    :param fourier_image: is a 2D array of dtype complex128 of shape (M,N) or (M,N,1)
    :return:
    """
    M = fourier_image.shape[0]
    N = fourier_image.shape[1]
    im = fourier_image.astype(np.complex128)

    for j in range(N):
        im[:, j] = IDFT(im[:, j])
    for i in range(M):
        im[i, :] = IDFT(im[i, :])
    return im

def change_rate(filename, ratio):
    """
    changes the duration of an audio file by keeping the same samples, but changing the
    sample rate written in the file header
    :param filename:  a string representing the path to a WAV file
    :param ratio: a positive float64 representing the duration change
    """
    sample_r, data = wavfile.read(filename)
    new_r = ratio * sample_r
    wavfile.write('change_rate.wav',int(new_r), data)

def change_samples(filename, ratio):
    """
    a fast forward function that changes the duration of an audio file by reducing the number of samples
    using Fourier
    :param filename: the file
    :param ratio: the ratio
    :return: the changed file
    """
    sample_r, data = wavfile.read(filename)
    changed = resize(data.astype(np.float64), ratio)
    wavfile.write('change samples.wav',sample_r,changed)
    return changed

def resize(data, ratio):
    """
     change the number of samples by the given ratio
    :param data: sample data
    :param ratio: ration to change to
    :return:
    """
    size = len(data)
    dft = DFT(data)
    shift = np.fft.fftshift(dft)
    if ratio > 1:

        l = int((size / 2) - (size / (2 * ratio)))

        r = size - l

        if size % 2 != 0:

            r -= 1
        c = shift[l:r]
        if data.dtype == np.float64:
            return np.real(IDFT(np.fft.ifftshift(c))).astype(data.dtype)
        else:
            return IDFT(np.fft.ifftshift(c)).astype(data.dtype)
    if ratio < 1:
        pad = int((float(1 / ratio) * size) - size)
        zero = np.zeros(pad)
        id = 0
        if id % 2 != 0:
            id = int(size / 2) + 1
        else:
            id = int(size / 2)
        p = np.hstack((zeros[:id], shift, zero[id:]))
        if data.dtype == np.float64:
            return np.real(IDFT(np.fft.ifftshift(c))).astype(data.dtype)
        else:
            return IDFT(np.fft.ifftshift(c)).astype(data.dtype)
    else:
        return data


def resize_spectrogram(data, ratio):
    """
    speeds up a WAV file, without changing the pitch, using spectrogram scaling
    :param data: data
    :param ratio: ration
    :return: sped up file
    """
    spec = stft(data)
    resized = np.apply_along_axis(resize, 1,arr= spec,ratio= ratio)
    return istft(resized)

def resize_vocoder(data, ratio):
    """
    speedups a WAV file by phase vocoding its spectrogram.
    :param data: data file
    :param ratio: ration
    :return: the given data
        rescaled according to ratio with the same datatype as data
    """
    spec = stft(data)
    return  istft(phase_vocoder(spec,ratio)).astype(data.dtype)

def conv_der(im):
    """
    computes the magnitude of image derivatives.
    :param im: image
    :return: magnitude of image derivatives
    """
    dx_c = np.array([0.5,0,-0.5]).reshape((1,3))
    dy_c = dx_c.transpose()
    dx = signal.convolve2d(im,dx_c,'same').astype(np.float64)
    dy = signal.convolve2d(im,dy_c,'same').astype(np.float64)
    return np.sqrt(np.abs(dx)**2 + np.abs(dy)**2)


def fourier_der(im):
    """
     computes the magnitude of the image derivatives using Fourier transform
    :param im: image
    :return: magnitude of the image derivatives using Fourier transform
    """

    r = im.shape[0]
    c = im.shape[1]
    dft = DFT2(im)
    shifted = np.fft.ifftshift(dft)
    r_f = (2 *  np.pi *1j)/r
    c_f = 2*np.pi*1j/c

    u = c_f * np.arange(-r/2, r/2).reshape([r,1])
    v = np.arange(-c/2, c/2).reshape([c,1])
    x = np.multiply(shifted,u)
    y = np.multiply(shifted.T,v)

    idft_x = IDFT2(np.fft.ifftshift(x))
    idft_y = IDFT2((np.fft.ifftshift(y)).T)
    return np.sqrt(np.abs(idft_x)**2 + np.abs(idft_y)**2)


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



