import cv2
import time
import glob

fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
videoWriter = cv2.VideoWriter('test_video.mp4', fourcc, 19, (1100, 800))

for i in range(1, 8492, 10):
    img = cv2.imread(f'lines/line_{i}.png')
    img = cv2.resize(img, (1100, 800))
    videoWriter.write(img)
videoWriter.release()
