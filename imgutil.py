'''
imgutil: set of wrapper functions to deal with images as arrays.

 ／l、               KP
（ﾟ､ ｡ ７
  l  ~ヽ
  じしf_,)ノ
'''

from PIL import Image
import numpy as np
import cv2 as cv

def load_img_array(ipath):
    '''load image path as ndarray'''
    return np.asarray(Image.open(ipath)) # use PIL to open, then return the array

def invert(img):
    '''simple invert colors functino'''
    return (255 - img)

def bin_thresh(img, pix_thresh = 100):
    _, thresh = cv.threshold(img, pix_thresh, 255, cv.THRESH_BINARY) # apply threshold according to specified value
    return thresh

def split_channels(img):
    """returns red, green, blue channels from given RGB image array"""
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2] # split channels, output grayscale
    # return RGB images
    zero = np.zeros(img.shape[0:2], np.uint8)  # zero array w same shape as image, must be 8-bit unsigned int
    red_ch = cv.merge([r, zero, zero])
    green_ch = cv.merge([zero, g, zero])
    blue_ch = cv.merge([zero, zero, b])
    return red_ch, green_ch, blue_ch

def reconstruct_splits(r, g, b):
    '''undo the split_channels fxn'''
    return cv.merge([r[:,:,0], g[:,:,1], b[:,:,2]])

def erode(img, iterations = 2, kernel_size = 3):
    '''applies erosion according to the specified parameters'''
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img = cv.erode(img, kernel, iterations = iterations, anchor = (1, 1))
    return img

def dilate(img, iterations = 2, kernel_size = 3):
    '''applies erosion according to the specified parameters'''
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img = cv.dilate(img, kernel, iterations = iterations, anchor = (1, 1))
    return img
    
def get_all_contours(img):
    """will return all contours for given (processed - eroded or grayscale) image"""
    contours, hierarchy = cv.findContours(
        image=img,
        mode=cv.RETR_TREE,
        method=cv.CHAIN_APPROX_NONE
    )  # find contours using openCV
    return contours, hierarchy

def get_largest_contours(img, n_contours=1, min_area=0):
    """returns n largest contours for a given processed image"""
    contours, hierarchy = cv.findContours(
        image=img,
        mode=cv.RETR_TREE,
        method=cv.CHAIN_APPROX_NONE
    )
    sorted_contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)
    largest_contours = sorted_contours[:n_contours]
    return [contour for contour in largest_contours if cv.contourArea(contour) >= min_area]

def approx_contours(contours, ep=0.015):
    """returns list of approximate contours for a given set of contours -- smaller epsilon values represent a sharper approximation. given by Ramer-Douglas-Peucker algorithm."""
    approx_rois = []
    for contour in contours:
        perimeter = ep * cv.arcLength(contour, True)
        dp = cv.approxPolyDP(contour, perimeter, True)
        approx_rois.append(dp)
    return approx_rois

def crop_rect(img, rect, pad = 50):
    '''simple function to return an image cropped according to a rectangle. padding is in pixels'''
    x, y, w, h = rect
    # for whatever dumbass reason, if x or y < 50, it gives an error. took me forever to figure this out
    x = max(x, 50)
    y = max(y, 50)
    img_crop = img[y - pad:y + h + pad, 
                   x - pad:x + w + pad]
    return img_crop

def save_img_array(img, dst):
    out_img = Image.fromarray(img)
    out_img.save(dst)



