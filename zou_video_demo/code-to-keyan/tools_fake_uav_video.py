import cv2
import os
import numpy as np
import random


class BGVibrator(object):
    def __init__(self, img, mode='random_walk'):
        self.img = img
        self.dx = 0
        self.dy = 0
        self.da = 0
        self.mode = mode
        self.t = 0
    def process(self):
        if self.mode == 'random_walk':
            self.dx += random.uniform(-3, 4)
            self.dy += random.uniform(-4, 4)
            self.da += random.uniform(-0.45, 0.5) / 180. * np.pi
        else:
            self.dx = 3*np.sin(2*np.pi*self.t/103)
            self.dy = 2*np.sin(2*np.pi*self.t/173)
            self.da = 1*np.sin(2*np.pi*self.t/101) / 180. * np.pi

        h, w = self.img.shape[0:2]
        transform = np.array([[np.cos(self.da), -np.sin(self.da),  self.dx], [np.sin(self.da),  np.cos(self.da),  self.dy]], dtype=np.float)
        self.t += 1
        return cv2.warpAffine(self.img, transform, (w, h), borderMode=cv2.BORDER_REFLECT)


if __name__ == '__main__':

    img_path = r'../../tmp/raw/WeChat Screenshot_20221005161704.bmp'
    output_folder = r'../../tmp/01'
    bg_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    bg_img = np.stack([bg_img, bg_img, bg_img], axis=-1)

    bg_vibrator = BGVibrator(bg_img, mode='sincos')

    m_frames = 200
    for i in range(m_frames):
        print('generating fake images %d/%d' % (i, m_frames))
        img_fake = bg_vibrator.process()

        cv2.imwrite(os.path.join(output_folder, str(i).zfill(5)+'.bmp'), img_fake)
        cv2.imshow('vis', img_fake)
        cv2.waitKey(100)

