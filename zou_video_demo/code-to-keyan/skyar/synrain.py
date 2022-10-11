import numpy as np
import cv2
import os
import glob
import random


class Rain():

    def __init__(self, rain_intensity=1.0, haze_intensity=4.0, gamma=2.0, light_correction=0.9,
                 with_rain_layer=True, with_windhield_layer=True):

        self.rain_intensity = rain_intensity
        self.haze_intensity = haze_intensity
        self.gamma = gamma
        self.light_correction = light_correction
        self.frame_id = 1

        self.with_rain_layer = with_rain_layer
        self.with_windhield_layer = with_windhield_layer
        self.rain_layer_cap = cv2.VideoCapture(r'./skyar/rain_layer.mp4')
        self.windshield_layer_cap = cv2.VideoCapture(r'./skyar/windshield_layer.mp4')

    def _get_rain_layer(self):

        ret, frame = self.rain_layer_cap.read()
        if ret:
            rain_layer = frame
        else: # if reach the last frame, read from the begining
            self.rain_layer_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.rain_layer_cap.read()
            rain_layer = frame

        rain_layer = cv2.cvtColor(rain_layer, cv2.COLOR_BGR2RGB) / 255.0
        rain_layer = np.array(rain_layer, dtype=np.float32)

        return rain_layer

    def _get_windshield_layer(self):

        ret, frame = self.windshield_layer_cap.read()
        if ret:
            windshield_layer = frame
        else: # if reach the last frame, read from the begining
            self.windshield_layer_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.windshield_layer_cap.read()
            windshield_layer = frame

        windshield_layer = cv2.cvtColor(windshield_layer, cv2.COLOR_BGR2RGB)
        r = windshield_layer[:,:,0]
        windshield_layer = np.stack([r, r, r], axis=-1)
        windshield_layer = np.array(windshield_layer, dtype=np.float32) / 255.

        return windshield_layer


    def _create_haze_layer(self, rain_layer):
        return 0.1*np.ones_like(rain_layer)

    def forward(self, img):

        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)

        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.

        # get input image size
        h, w, c = img.shape

        if self.with_windhield_layer:
            # create a windshield layer
            windshield_layer = self._get_windshield_layer()
            windshield_layer = cv2.resize(windshield_layer, (w, h))
            windshield_layer = cv2.blur(windshield_layer, (3, 3))
            windshield_layer = self.rain_intensity * windshield_layer
            # synthesize an output image (screen blend)
            img = 1 - (1 - windshield_layer) * (1 - img)

        if self.with_rain_layer:
            # create a rain layer
            rain_layer = self._get_rain_layer()
            rain_layer = cv2.resize(rain_layer, (w, h))
            rain_layer = cv2.blur(rain_layer, (3, 3))
            rain_layer = rain_layer * (1 - cv2.boxFilter(img, -1, (int(w/10), int(h/10))))

            # create a haze layer
            haze_layer = self._create_haze_layer(rain_layer)

            # combine the rain layer and haze layer together
            rain_layer = 0.2*self.rain_intensity*rain_layer + \
                         self.haze_intensity*haze_layer

            # synthesize an output image (screen blend)
            img = 1 - (1 - rain_layer) * (1 - img)

        # gamma and light correction
        img = self.light_correction*(img**self.gamma)

        # check boundary
        img = np.clip(img, a_min=0, a_max=1.)

        img = (255.*img).astype(np.uint8)

        return img

