import numpy as np
import cv2


def _l(x, y):
    return (x**2 + y**2)**0.5


class ObjDet():

    def __init__(self):
        pass

    @staticmethod
    def visualize(img, bbs, with_interactions=True, with_bbs=True, with_text=True):
        vis = np.copy(img)

        # draw interactions
        if with_interactions:
            new_layer = np.zeros_like(vis)
            for i in range(len(bbs)):
                for j in range(i + 1, len(bbs)):
                    bb_i, bb_j = bbs[i], bbs[j]
                    l_i, l_j = _l(bb_i['w'], bb_i['h']), _l(bb_j['w'], bb_j['h'])
                    pt1, pt2 = (int(bb_i['xc']), int(bb_i['yc'])), (int(bb_j['xc']), int(bb_j['yc']))
                    d = _l(bb_i['xc'] - bb_j['xc'], bb_i['yc'] - bb_j['yc'])
                    s = int(255 * np.exp(- 0.5*d / max(l_i, l_j)))  # interaction intensity
                    cv2.line(new_layer, pt1, pt2, (0, 0, s), 1)
            vis = cv2.add(vis, new_layer)

        # draw bbs
        if with_bbs:
            for bb in bbs:
                w, h, xc, yc = bb['w'], bb['h'], bb['xc'], bb['yc']
                l = _l(w, h)
                pt1, pt2 = (int(xc - l / 2.0), int(yc - l / 2.0)), (int(xc + l / 2.0), int(yc + l / 2.0))
                cv2.rectangle(vis, pt1, pt2, (0, 0, 255), 1)  # bb
                cv2.circle(vis, (int(xc), int(yc)), 2, (0, 255, 255), -1)  # center

        # draw text
        if with_text:
            for i in range(len(bbs)):
                text = 'v' + str(i)
                tsize, _ = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                           fontScale=0.5, thickness=1)
                dx, dy = int(tsize[0] / 2.0), int(tsize[1] / 2.0)
                pt = (int(bbs[i]['xc'] + dx), int(bbs[i]['yc'] + dy))
                cv2.putText(vis, text=text, org=pt, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        return vis

    def detect(self, img, mask):

        bbs = []

        # Apply the Component analysis function
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        mask = cv2.threshold(mask, 0, 255, cv2.THRESH_OTSU)[1]
        analysis = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
        (totalLabels, label_ids, values, centroid) = analysis

        # Loop through each component
        for i in range(1, totalLabels):

            # filter out small regions
            if values[i, cv2.CC_STAT_AREA] < 25:
                continue

            # Now extract the coordinate points
            w = values[i, cv2.CC_STAT_WIDTH]
            h = values[i, cv2.CC_STAT_HEIGHT]
            x1 = values[i, cv2.CC_STAT_LEFT]
            y1 = values[i, cv2.CC_STAT_TOP]
            x2 = x1 + w
            y2 = y1 + h

            ext = 5
            y1, x1, y2, x2 = y1 - ext, x1 - ext, y2 + ext, x2 + ext
            (xc, yc) = centroid[i]
            bbs.append({'x1': x1, 'y1':y1, 'x2': x2, 'y2':y2, 'w': w, 'h': h, 'xc': xc, 'yc': yc})

        vis = self.visualize(img, bbs, with_interactions=False)

        return vis, bbs

