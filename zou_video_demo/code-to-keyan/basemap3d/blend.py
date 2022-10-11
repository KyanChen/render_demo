import cv2
import numpy as np
import skimage.morphology as mp
import skimage.measure as measure
from skimage.exposure import match_histograms


def harmonization(img_fg, mask, img_bg, color_correction=0.5):

    img_bg = img_bg.astype(np.float32) / 255.
    img_fg = img_fg.astype(np.float32) / 255.
    mask = mask.astype(np.float32) / 255.

    # histogram matching
    # img_fg = img_fg**gamma_correction
    matched = match_histograms(img_fg, img_bg, channel_axis=-1)
    img_fg = color_correction*matched + (1-color_correction)*img_fg

    # shadow
    mask = create_shadow(mask)

    # noise and blur
    img_fg, mask = noise_and_blur(img_fg, mask)

    img_fg = (img_fg * 255.).astype(np.uint8)
    mask = (mask * 255.).astype(np.uint8)

    return img_fg, mask



def blend_images(img_bg, img_fg, alpha, mode='defaalt'):

    if img_bg.dtype == np.uint8:
        img_bg = img_bg.astype(np.float32) / 255.
    if img_fg.dtype == np.uint8:
        img_fg = img_fg.astype(np.float32) / 255.
    if alpha.dtype == np.uint8:
        alpha = alpha.astype(np.float32) / 255.

    if mode == 'default':
        img_blend = img_fg * alpha + img_bg * (1 - alpha)
    else: # screen blend
        img_blend = img_fg * alpha + img_bg * (1 - alpha)
    img_blend = (img_blend*255.).astype(np.uint8)

    return img_blend


def noise_and_blur(img_fg, mask):

    # forgrund blur
    img_fg = cv2.blur(img_fg, ksize=(3, 3))

    # forgrund noise
    img_fg = img_fg + 0.02*np.random.randn(*img_fg.shape)
    img_fg[img_fg < 0] = 0
    img_fg[img_fg > 1.0] = 1.0

    # mask feather
    mask = cv2.erode(mask, np.ones([3, 3]))
    mask = cv2.blur(mask, ksize=(3, 3))

    return img_fg, mask


def create_shadow(mask):

    if np.any(mask) == 0:
        return mask

    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    chull = mp.convex_hull_object(mask)
    labels = mp.label(chull)

    props = measure.regionprops(labels)
    shadow_mask = np.zeros_like(mask)

    bb_sizes = []
    for id in range(len(props)):
        bb = props[id].bbox
        y1, x1, y2, x2 = bb
        bbh, bbw = y2 - y1, x2 - x1
        bb_sizes.append((bbh**2 + bbw*2)**0.5)
        shadow_mask[y1+int(0.5*bbh):y2, x1:x2] = chull[y1+int(0.5*bbh):y2, x1:x2]

    ksize = int(0.5 * max(bb_sizes))
    shadow_mask = shadow_mask + 2*cv2.blur(shadow_mask, ksize=(ksize, ksize))

    # import matplotlib.pyplot as plt
    # plt.figure(), plt.imshow(mask), plt.title('mask')
    # plt.figure(), plt.imshow(chull), plt.title('chull')
    # plt.figure(), plt.imshow(labels, cmap='gray'), plt.title('labels')
    # plt.figure(), plt.imshow(shadow_mask, cmap='gray'), plt.title('shadow_mask')
    # plt.show()

    mask = mask + shadow_mask
    mask[mask > 1.0] = 1.0

    return np.stack([mask, mask, mask], axis=-1)


