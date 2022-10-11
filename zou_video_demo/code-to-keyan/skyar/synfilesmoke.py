import numpy as np
import cv2
import os
import glob
import random


class FireSmoke():

    def __init__(self, img_h, img_w):

        self.img_h, self.img_w = img_h, img_w
        self.layer_cap = cv2.VideoCapture(r'./skyar/smoke1.mp4')
        self.bg_color_R, self.bg_color_G, self.bg_color_B = 15, 244, 3

        self.layer_h = self.layer_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.layer_w = self.layer_cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    def _get_firesmoke_layer(self):

        ret, frame = self.layer_cap.read()
        if ret:
            layer = frame
        else: # if reach the last frame, read from the begining
            self.layer_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.layer_cap.read()
            layer = frame

        return layer

    def _parse_location_mask(self, mask):

        # Apply the Component analysis function
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        mask = cv2.threshold(mask, 0, 255, cv2.THRESH_OTSU)[1]
        analysis = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
        (totalLabels, label_ids, values, centroid) = analysis

        if totalLabels == 1:
            return None, None, None, None

        X1, Y1, X2, Y2 = self.img_w, self.img_h, 0, 0
        # Loop through each component
        for i in range(1, totalLabels):
            # Now extract the coordinate points
            x1 = values[i, cv2.CC_STAT_LEFT]
            y1 = values[i, cv2.CC_STAT_TOP]
            x2 = x1 + values[i, cv2.CC_STAT_WIDTH]
            y2 = y1 + values[i, cv2.CC_STAT_HEIGHT]
            # update X1, Y1, X2, Y2
            X1, Y1, X2, Y2 = min(X1, x1), min(Y1, y1), max(X2, x2), max(Y2, y2)

        return X1, Y1, X2, Y2


    def reder(self, location_mask):

        x1, y1, x2, y2 = self._parse_location_mask(location_mask)
        l = ((x2-x1)**2 + (y2-y1)**2)**0.5

        layer = self._get_firesmoke_layer()

        new_w = int(5.0 * l)
        new_h = int(5.0 * l / self.layer_w * self.layer_h)
        layer = cv2.resize(layer, (new_w, new_h))

        dx, dy = x1 - l/4.0, y2 - new_h
        transform = np.array([[1., - 0., dx], [0., 1., dy]], dtype=np.float)
        layer = cv2.warpAffine(layer, transform, (self.img_w, self.img_h), borderMode=cv2.BORDER_REPLICATE)

        layer = layer.astype(np.float32) / 255.0
        B, G, R = layer[:,:,0], layer[:,:,1], layer[:,:,2]
        dB = np.abs(B - self.bg_color_B/255.)
        dG = np.abs(G - self.bg_color_G/255.)
        dR = np.abs(R - self.bg_color_R/255.)
        diff = np.stack([dB, dG, dR], axis=-1)
        a = np.max(diff, axis=-1, keepdims=False)
        alpha = np.stack([a, a, a], axis=-1)

        return layer, alpha

