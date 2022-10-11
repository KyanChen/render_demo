import cv2
import os
import numpy as np

# gamma correction
def gamma_correction(img, c=1, g=2.2):
    out = img.copy().astype(np.float32)
    out /= 255.
    out = (c * out) ** g
    out = np.clip(out, a_min=0, a_max=1.0)

    out *= 255
    out = out.astype(np.uint8)

    return out


video_path = r'../../tmp/01.mp4'
output_folder = r'../../tmp/01'
with_enhance = True
with_resize = True

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

cap = cv2.VideoCapture(video_path)
m_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
img_HD_prev = None

idx = 0

while (1):
    ret, frame = cap.read()
    if ret:
        if with_enhance:
            frame = gamma_correction(frame, c=1.5, g=2)
        if with_resize:
            frame = cv2.resize(frame, (1024, 1024))
        cv2.imwrite(os.path.join(output_folder, str(idx).zfill(5)+'.bmp'), frame)
        print(idx)
        idx += 1
    else:  # if reach the last frame
        break