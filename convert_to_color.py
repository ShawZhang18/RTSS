import cv2
import os.path
import glob
import numpy as np
from PIL import Image


def convertPNG(pngfile, outdir):
    # READ THE DEPTH
    im_depth = cv2.imread(pngfile)
    # apply colormap on deoth image(image must be converted to 8-bit per pixel first)
    color_map = cv2.convertScaleAbs(im_depth, alpha=1.0)
    im_color = cv2.applyColorMap(color_map, cv2.COLORMAP_RAINBOW)
    for i in range(400):
        for j in range(640):
            if im_color[i][j][2] > 250 and im_color[i][j][0] <= 10  and  im_color[i][j][1] <= 10:
                im_color[i][j][2] = 0
            # if im_color[i][j][0] > 240 and im_color[i][j][1] == 0  and  im_color[i][j][2] == 0:
            #     im_color[i][j][0] = 0
            # if im_color[i][j][1] > 240 and im_color[i][j][0] == 0  and  im_color[i][j][2] == 0:
            #     im_color[i][j][1] = 0
    # convert to mat png
    im = Image.fromarray(im_color)
    # save image
    im.save(os.path.join(outdir, os.path.basename(pngfile)))


if __name__ == '__main__':
    convertPNG('/home/zhangxiao/data/zacaoKitti1000/disp_occ_0/001199.png','origincolor/')
    convertPNG('/home/zhangxiao/code/AnyNet/results/zacao_kitti/png_result/001199.png','color/')