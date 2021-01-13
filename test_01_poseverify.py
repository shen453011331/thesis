# 2021.1.12
# 用来对弓网数据集的姿态计算结果通过画极线的方式来看是否计算准确
import os
import numpy as np
import cv2 #.cv2 as cv2
from PIL import Image

from matplotlib import pyplot as plt
from pose_estimator import *
from basic_function import *

from scipy.spatial.transform import Rotation as R


def convert_euler(r_vec):
    r = R.from_rotvec(list(map(float,r_vec)))
    angle = r.as_euler('zyx', degrees=False)
    return ['{:.6f}'.format(i) for i in angle]


global_x, global_y = 0.0, 0.0

def read_txt(filename, line_num):
    # 读取文件名和要读取的行数
    str = None
    with open(filename, 'r') as f:
        for i in range(line_num):
            f.readline()
        str = f.readline()
    str_list = str.split(' ')
    return str_list[0], str_list[1], np.array(list(map(float,str_list[-12:-6])))

def get_ploar(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    # we select only inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    def drawlines(img1, img2, lines, pts1, pts2):
        ''' img1 - image on which we draw the epilines for the points in img2
            lines - corresponding epilines
        '''
        r, c = img1.shape
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        for r, pt1, pt2 in zip(lines, pts1, pts2):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
            img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
            img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
            img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
            break
        return img1, img2

    # find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

    # find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

    plt.subplot(121), plt.imshow(img5)
    plt.subplot(122), plt.imshow(img3)
    # plt.show()
    return F


def plot_polar(base_path, img_name_1, img_name_2, pose):
    img_1 = cv2.imread(os.path.join(base_path, img_name_1), cv2.IMREAD_GRAYSCALE)
    img_2 = cv2.imread(os.path.join(base_path, img_name_2), cv2.IMREAD_GRAYSCALE)
    return get_ploar(img_1, img_2)
    # print('show image 1')
    # cv2.namedWindow('img_1', 0)
    # loc = cv2.setMouseCallback('img_1', on_EVENT_LBUTTONDOWN)
    # cv2.imshow('img_1', img_1)
    # cv2.waitKey()
    # print('plot polar line and show on img 2')
    # lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    #
    # cv2.namedWindow('img_2', 0)
    # cv2.imshow('img_2', img_2)
    # cv2.waitKey()

def get_essitial_matrix(pose):
    t = pose[3:]
    angles = pose[:3][::-1]
    r = R.from_euler('zxy', pose[:3][::-1], degrees=False)
    mat = r.as_matrix()
    t_x = np.array([[0, -t[2], t[1]],
                    [t[2], 0, -t[0]],
                    [-t[1], t[0], 0]])
    return np.matmul(t_x, mat)

def get_fundmental_matrix(intrisic, E):
    K_inv = np.linalg.inv(intrisic.Matrix)
    return np.matmul(K_inv.T, np.matmul(E, K_inv))


def get_E_from_F(intrisic, F):
    K = intrisic.Matrix
    return np.matmul(K.T, np.matmul(F, K))

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = '%d,%d' % (x, y)
        print('x, y = {}, {}'.format(x, y))
        global_x, global_y = x,y
        # cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
        # cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
        # 1.0, (0, 0, 0), thickness=1)
        # cv2.imshow(“image”, img)

def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    depth_png = np.array(img_file, dtype=int)
    img_file.close()
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255, \
        "np.max(depth_png)={}, path={}".format(np.max(depth_png),filename)

    depth = depth_png.astype(np.float) / 256.
    # depth[depth_png == 0] = -1.
    depth = np.expand_dims(depth, -1)
    return depth


def rgb_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    # rgb_png = np.array(img_file, dtype=float) / 255.0 # scale pixels to the range [0,1]
    rgb_png = np.array(img_file, dtype='uint8')  # in the range [0,255]
    img_file.close()
    return rgb_png



# if __name__ == '__main__':
#     txt_file = os.path.join('D:\download\lidar_data_0107', '0107_true_pose_inv_rot.txt')
#     base_path = 'D:\FULL_KITTI_PAN'
#     img_name_1, img_name_2, pose = read_txt(txt_file, 0)
#     base_para_path = 'D:\download\lidar_data_0107'
#     cam_in_para_new, cam_ex_para = read_parameters(os.path.join(base_para_path, 'intrinsic_1207_refine.txt'),
#                                                    os.path.join(base_para_path, 'extrinsic.txt'))
#     print(get_essitial_matrix(pose))
#     # print(get_fundmental_matrix(cam_in_para_new, get_essitial_matrix(pose)))
#     # [[-2.46643314e-11 - 1.97299994e-08  1.62818469e-05]
#     #  [1.97328147e-08 - 3.05037344e-11 - 1.52616332e-05]
#     #  [-1.62006179e-05  1.51552404e-05  5.84976699e-05]]
#
#     F = plot_polar(base_path, img_name_1, img_name_2, pose)
#     # [[-7.25342006e-08  6.09158761e-05 - 6.80481225e-02]
#     #  [-6.02010691e-05  4.84512499e-07  4.66516704e-02]
#     # [6.72866208e-02 - 4.80743440e-02    1.00000000e+00]]
#     print(get_E_from_F(cam_in_para_new, F))
#
#     #结果，确实存在问题，但是暂时不解决