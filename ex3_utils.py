import math
import sys
from typing import List, Tuple

import numpy as np
import cv2
from numpy.linalg import inv, linalg
from scipy import signal
import matplotlib.pyplot as plt
from alive_progress import alive_bar


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 314855099


# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10,
                win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """
    if im1.ndim > 2:  # this is RGB image
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    win_half = int(win_size / 2)
    im1 = cv2.copyMakeBorder(im1, win_half, win_half, win_half,  #
                             win_half, borderType=cv2.BORDER_REPLICATE)
    im2 = cv2.copyMakeBorder(im2, win_half, win_half, win_half,
                             win_half, borderType=cv2.BORDER_REPLICATE)
    x_drive = cv2.Sobel(im2, cv2.CV_64F, 1, 0, ksize=3,
                        borderType=cv2.BORDER_DEFAULT)
    y_drive = cv2.Sobel(im2, cv2.CV_64F, 0, 1, ksize=3,
                        borderType=cv2.BORDER_DEFAULT)
    points = []
    u_v = []
    for i in range(win_half, im1.shape[0] - win_half - 1, step_size):
        for j in range(win_half, im1.shape[1] - win_half - 1, step_size):
            Ix = x_drive[i - win_half:i + win_half + 1, j - win_half:j + win_half + 1]
            Iy = y_drive[i - win_half:i + win_half + 1, j - win_half:j + win_half + 1]
            It = im1[i - win_half:i + win_half + 1, j - win_half:j + win_half + 1] \
                 - im2[i - win_half:i + win_half + 1, j - win_half:j + win_half + 1]
            A: np.ndarray = np.concatenate(
                [np.atleast_2d(Ix.flatten()).T, np.atleast_2d(Iy.flatten()).T], axis=1)
            AtA = A.T @ A
            lamdas = np.linalg.eigvals(AtA)
            lamda2 = np.min(lamdas)
            lamda1 = np.max(lamdas)
            if lamda2 <= 1 or lamda1 / lamda2 >= 100:  # AtA is Non-well-Defined so we get not good points
                continue
            Atb = A.T.dot(np.atleast_2d(It.flatten()).T)
            uv = np.linalg.inv(AtA).dot(Atb)
            u_v.append(uv.T.reshape(-1))
            points.append([j - win_half, i - win_half])
    return np.array(points), np.array(u_v)


def opticalFlowPyrLK(img1: np.ndarray, img2: np.ndarray, k: int,
                     stepSize: int, winSize: int) -> np.ndarray:
    """
    :param img1: First image
    :param img2: Second image
    :param k: Pyramid depth
    :param stepSize: The image sample size
    :param winSize: The optical flow window size (odd number)
    :return: A 3d array, with a shape of (m, n, 2),
    where the first channel holds U, and the second V.
    Ui = Ui + 2 ∗ Ui−1, Vi = Vi + 2 ∗ Vi−1
    """
    im1_pyr = gaussianPyr(img1, k)  # gauss pyramid for img1 my implementation
    im2_pyr = gaussianPyr(img2, k)  # gauss pyramid for img2
    curr = np.zeros(
        (im1_pyr[k - 2].shape[0], im1_pyr[k - 2].shape[1], 2))  # (m,n,2) zero array to put in u,v for each pixel
    last = np.zeros((im1_pyr[k - 1].shape[0], im1_pyr[k - 1].shape[1], 2))
    points, uv = opticalFlow(im1_pyr[k - 1], im2_pyr[k - 1], stepSize, winSize)
    for j in range(len(points)):  # change pixels uv by formula
        y, x = points[j]
        u, v = uv[j]
        last[x, y, 0] = u
        last[x, y, 1] = v

    for i in range(k - 2, -1, -1):  # for each level of pyramids (small -> big)
        points, uv = opticalFlow(im1_pyr[i], im2_pyr[i], stepSize, winSize)  # uv for i'th img
        for j in range(len(points)):  # change pixels uv by formula
            y, x = points[j]
            u, v = uv[j]
            curr[x, y, 0] = u
            curr[x, y, 1] = v
        for z in range(last.shape[0]):
            for r in range(last.shape[1]):
                #  Ui = Ui + 2 ∗ Ui−1, Vi = Vi + 2 ∗ Vi−1
                curr[z * 2, r * 2, 0] += last[z, r, 0] * 2
                curr[z * 2, r * 2, 1] += last[z, r, 1] * 2

        last = curr.copy()
        if i - 1 >= 0:
            curr.fill(0)
            curr.resize((im1_pyr[i - 1].shape[0], im1_pyr[i - 1].shape[1], 2), refcheck=False)

    return curr


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------

def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """
    points, uv = opticalFlow(im1, im2, 20, 5)
    # print("uv: ", uv)
    min_mse = 1000  # arbitrary
    tran_mat = np.array(0)
    for i in range(uv.shape[0]):  # find the best uv by the minimum mse
        tx = np.around(uv[i, 0], decimals=2)
        ty = np.around(uv[i, 1], decimals=2)
        if tx != 0 and ty != 0:
            tmp_t = np.array([[1, 0, tx],
                              [0, 1, ty],
                              [0, 0, 1]], dtype=np.float64)

            img_by_t = cv2.warpPerspective(im1, tmp_t, im1.shape[::-1])
            mse = np.square(im2 - img_by_t).mean()
            if mse < min_mse:
                min_mse = mse
                tran_mat = tmp_t

    return tran_mat


def findTheta(im1, im2):
    min_mse = 1000
    theta = 0
    for t in range(360):  # find the best angle by the minimum mse
        tmp_t = np.array([[math.cos(t), -math.sin(t), 0],
                          [math.sin(t), math.cos(t), 0],
                          [0, 0, 1]], dtype=np.float64)
        img_by_t = cv2.warpPerspective(im1, tmp_t, im1.shape[::-1])
        mse = np.square(np.subtract(im2, img_by_t)).mean()
        if mse < min_mse:
            min_mse = mse
            tran_mat = tmp_t
            theta = t
    return theta


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """
    t = findTheta(im1, im2)
    rigid_mat = np.array([[math.cos(t), math.sin(t), 0],
                          [-math.sin(t), math.cos(t), 0],
                          [0, 0, 1]], dtype=np.float64)
    # take the Reverse matrix of rigid_mat and warp with it - then we get only translation without rigid
    revers_img = cv2.warpPerspective(im2, rigid_mat, im2.shape[::-1])
    # now we need to find the translation only
    tran_mat = findTranslationLK(im1, revers_img)
    tx = tran_mat[0, 2]
    ty = tran_mat[1, 2]
    # A combination of the two matrices we found
    ans = np.array([[math.cos(t), -math.sin(t), tx],
                    [math.sin(t), math.cos(t), ty],
                    [0, 0, 1]], dtype=np.float64)

    return ans


""""findTranslationCorr function find the optimum solution but with 45 min of running- 
# so we need to resize the image at least to shape (140,140)
"""


def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by correlation.
    """
    uvs = opticalFlowCrossCorr(im1, im2, step_size=32, win_size=13)
    u, v = np.ma.median(np.ma.masked_where(
        uvs == np.zeros((2)), uvs), axis=(0, 1)).filled(0)
    return np.array([[1, 0, u],
                     [0, 1, v],
                     [0, 0, 1]])


def opticalFlowCrossCorr(im1: np.ndarray, im2: np.ndarray, step_size, win_size):
    half = win_size // 2
    uv = np.zeros((*im1.shape, 2))

    def Best_Corr(win: np.ndarray):
        max_corr = -1
        best_corr = 0
        window1 = win.copy().flatten() - win.mean()
        norm1 = np.linalg.norm(window1, 2)  # normalize
        for y in range(half, im2.shape[0] - half - 1):
            for x in range(half, im2.shape[1] - half - 1):
                window2 = im2[y - half: y + half + 1, x - half: x + half + 1]
                window2 = window2.copy().flatten() - window2.mean()
                norms = norm1 * np.linalg.norm(window2, 2)  # ||win1||*||win2||
                corr = 0 if norms == 0 else np.sum(window1 * window2) / norms  # the correlation
                if corr > max_corr:
                    max_corr = corr
                    best_corr = (y, x)
        return best_corr

    # take a window from image 1 and find the maximum correlation for it in image 2
    for y in range(half, im1.shape[0] - half - 1, step_size):
        for x in range(half, im1.shape[1] - half - 1, step_size):
            window = im1[y - half: y + half + 1, x - half: x + half + 1]
            if cv2.countNonZero(window) == 0:
                continue
            top_correlation = Best_Corr(window)
            uv[y - half, x -
               half] = np.flip(top_correlation - np.array([y, x]))

    return uv


""""findRigidCorr function find the optimum solution but with 45 min of running- 
# so we need to resize the image at least to shape (140,140)
"""


def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """
    t = findTheta(im1, im2)
    rigid_mat = np.array([[math.cos(t), -math.sin(t), 0],
                          [math.sin(t), math.cos(t), 0],
                          [0, 0, 1]], dtype=np.float64)
    # reverse im1 with the rigid we found
    revers_img = cv2.warpPerspective(im1, rigid_mat, im2.shape[::-1])

    # after we turn the image back, find the translation with corralation only
    tran_mat = findTranslationCorr(im2, revers_img)
    tx = tran_mat[0, 2]
    ty = tran_mat[1, 2]
    ans = np.array([[math.cos(t), -math.sin(t), tx],
                    [math.sin(t), math.cos(t), ty],
                    [0, 0, 1]], dtype=np.float64)

    return ans


def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: input image 2 in grayscale format.
    :param T: is a 3x3 matrix such that each pixel in image 2
    is mapped under homogenous coordinates to image 1 (p2=Tp1).
    :return: warp image 2 according to T and display both image1
    and the wrapped version of the image2 in the same figure.
    """
    if im1.ndim > 2:  # this is RGB image
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    h, w = im1.shape
    new_img = np.zeros((h, w))
    T_inv = np.linalg.inv(T)
    for i in range(im2.shape[0]):
        for j in range(im2.shape[1]):
            new_index = np.array([i, j, 1])
            newarr = T_inv.dot(new_index)
            x = newarr[0].astype(float)
            y = newarr[1].astype(float)
            x_ceil = int(math.ceil(x))
            y_ceil = int(math.ceil(y))
            x_floor = int(math.floor(x))
            y_floor = int(math.floor(y))
            a = np.round(x % 1, 3)
            b = np.round(y % 1, 3)
            intance = 0
            if x_floor < h and y_ceil < w and x_floor >= 0 and y_ceil >= 0:
                intance += (1 - a) * (1 - b) * im1[x_floor, y_ceil]
            if x_ceil < h and y_floor < w and x_ceil >= 0 and y_floor >= 0:
                intance += a * (1 - b) * im1[x_ceil, y_floor]
            if x_ceil < h and y_ceil < w and x_ceil >= 0 and y_ceil >= 0:
                intance += a * b * im1[x_ceil, y_ceil]
            if x_floor < h and y_ceil < w and x_floor >= 0 and y_ceil >= 0:
                intance += (1 - a) * b * im1[x_floor, y_ceil]
            new_img[i, j] = intance

    f, ax = plt.subplots(2)
    plt.gray()
    ax[0].imshow(im1)
    ax[0].set_title('im 1 before warping')
    ax[1].imshow(new_img)
    ax[1].set_title('after my warping')
    plt.show()
    return new_img


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    ans = []
    width = (2 ** levels) * int(img.shape[1] / (2 ** levels))
    height = (2 ** levels) * int(img.shape[0] / (2 ** levels))
    img = cv2.resize(img, (width, height))  # resize the image to dimensions that can be divided into 2 x times
    img = img.astype(np.float64)
    ans.append(img)  # level 0 - the original image
    temp_img = img.copy()
    for i in range(1, levels):
        temp_img = reduce(temp_img)  # 2 times smaller image
        ans.append(temp_img)
    return ans


def get_gaussian():  # create Gaussian kernel
    kernel = cv2.getGaussianKernel(5, -1)
    kernel = kernel.dot(kernel.T)
    return kernel


def reduce(img: np.ndarray) -> np.ndarray:  # reduce the image of one iterate
    g_kernel = get_gaussian()
    blur_img = cv2.filter2D(img, -1, g_kernel)
    new_img = blur_img[::2, ::2]  # sampling : took every second in each x , y
    return new_img


def gaussExpand(img: np.ndarray, gs_k: np.ndarray) -> np.ndarray:
    if img.ndim == 3:  # RGB img
        padded_im = np.zeros(((img.shape[0] * 2), (img.shape[1] * 2), 3))  # adding zeros because of the sampling
    else:
        padded_im = np.zeros((img.shape[0] * 2, img.shape[1] * 2))  #
    padded_im[::2, ::2] = img  # add to the sample places the values
    return cv2.filter2D(padded_im, -1, gs_k)


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    gauss_pyr = gaussianPyr(img, levels)
    g_kernel = get_gaussian() * 4
    for i in range(levels - 1):
        gauss_pyr[i] = gauss_pyr[i] - gaussExpand(gauss_pyr[i + 1], g_kernel)
    return gauss_pyr


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Restores the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    levels = len(lap_pyr)
    temp = lap_pyr[-1]  # the smallest image (from the gaussPyramid)
    gs_k = get_gaussian() * 4
    i = levels - 1
    while i > 0:  # go through the list from end to start
        expand = gaussExpand(temp, gs_k)
        temp = expand + lap_pyr[i - 1]
        i -= 1
    return temp


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray,
             mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: (Naive blend, Blended Image)
    """
    img_1, img_2 = resizeToMask(img_1, mask), resizeToMask(img_2, mask)
    lap_a = laplaceianReduce(img_1, levels)
    lap_b = laplaceianReduce(img_2, levels)
    ga_m = gaussianPyr(mask, levels)
    pyr_bland = ga_m[-1] * lap_a[-1] + (1 - ga_m[-1]) * lap_b[-1]
    gs_k = get_gaussian() * 4
    k = levels - 2
    while k >= 0:  # go through the list from end to start
        pyr_bland = gaussExpand(pyr_bland, gs_k) + lap_a[k] * ga_m[k] + (1 - ga_m[k]) * lap_b[k]
        k -= 1

    # Naive Blending
    naiveBlend = mask * img_1 + (1 - mask) * img_2
    naiveBlend = cv2.resize(naiveBlend, (pyr_bland.shape[1], pyr_bland.shape[0]))
    return naiveBlend, pyr_bland


def resizeToMask(img: np.ndarray, mask: np.ndarray):
    return np.resize(img, mask.shape)
