from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import re
import pandas as pd



def ransac(points, threshold):
    # 进行ransac，用于保障定位的鲁棒性
    ransac_model = RANSACRegressor(LinearRegression(), max_trials=20, min_samples=3,
                                   loss='squared_loss', stop_n_inliers=8,
                                   residual_threshold=threshold, random_state=None)
    line_model = LinearRegression()
    x = points[0:2, :].T
    y = points[2:, :].T
    ransac_model.fit(x, y)
    inlier_mask = ransac_model.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    if inlier_mask.tolist().count(True) < 3:
        d = 0
        k = 0
    else:
        line_model.fit(x[inlier_mask, :], y[inlier_mask, :])

        k = line_model.coef_
        d = line_model.intercept_
        # d = ransac_model.predict([[0]])[0, 0]
        # k = ransac_model.predict([[1]])[0, 0] - ransac_model.predict([[0]])[0, 0]
    return k, d, inlier_mask

def get_distance(k,d, points):
    scale = np.linalg.norm(np.hstack([k[0],-1]))
    dis = (np.matmul(points.T ,np.hstack([k[0],-1])) + d)/scale
    return dis

def get_points_from_time(df, min_time, stage):
    # time ， 1ms is  1000000
    ms_num = 1000000
    df_temp = df[(df.Timestamp < min_time + ms_num*stage) & (df.Timestamp >= min_time)]
    return np.vstack([df_temp.X.values, df_temp.Y.values, df_temp.Z.values])


class init_para(object):
    def __init__(self, fx, fy, u0, v0, k1, k2):
        self.fx = fx
        self.fy = fy
        self.u0 = u0
        self.v0 = v0
        self.k1 = k1
        self.k2 = k2
        self.Matrix= np.array([[fx,0,u0], [0,fy,v0], [0,0,1]], dtype='float')
        self.distCoeffs = np.zeros([5, 1], dtype='float')
        self.distCoeffs[0], self.distCoeffs[1] = k1, k2



class exte_para(object):
    def __init__(self, R, T):
        self.R = np.array(R)
        self.T = np.array(T).T

def read_parameters(in_path, ex_path):

    with open(in_path, 'r+') as f:
        str = f.readline()
        l1 = list(map(float, f.readline().split(" ")))
        l2 = list(map(float, f.readline().split(" ")))
        l3 = list(map(float, f.readline().split(" ")))
        f.readline()
        f.readline()
        k1 = list(map(float, f.readline().split(" ")))
        cam_in_para = init_para(l1[0], l2[1], l1[2], l2[2], k1[0], k1[1])
    with open(ex_path, 'r+') as f:
        str = f.readline()
        l1 = list(map(float, re.split(r"[ ]+", f.readline())))
        l2 = list(map(float, re.split(r"[ ]+", f.readline())))
        l3 = list(map(float, re.split(r"[ ]+", f.readline())))
        R = np.vstack([np.array(l1[:3]), np.array(l2[:3]), np.array(l3[:3])])
        T = [l1[-1], l2[-1], l3[-1]]
        cam_ex_para = exte_para(R, T)
    return cam_in_para, cam_ex_para


def get_points_from_df(csv_file, seq_num):
    # seq_num can be float, which means it will get percents of one 10000 points
    seq_df = pd.read_csv(csv_file, usecols=range(1,2))
    head_num = 10  # 每行数据，头数据个数
    time_idx, seq_idx, num_idx = 0, 1, 5  # 头数据中时间、序列号、以及行包含点云数量的对应位置
    points_data_len = 7  # 行数据中每个点的数据长度
    x_idx = 1 # 行数据中每个点数据中x坐标对应位置，其后分别是y，z，reflectivity

    start_num, end_num = int(np.floor(seq_num[0])), int(np.ceil(seq_num[1]))

    start_idx, end_idx = seq_df[seq_df['field.header.seq']==start_num].index.values[0],\
                 seq_df[seq_df['field.header.seq']==end_num].index.values[0]
    temp_df = pd.read_csv(csv_file, skiprows=start_idx, nrows=end_idx-start_idx+1)
    # temp_df = df[(df['field.header.seq'] >= seq_num[0]) & (df['field.header.seq'] <= seq_num[1])]
    points_data = None
    for i in range(len(temp_df)):
        head_data = temp_df.iloc[i].values[:head_num]
        points_num = int(head_data[num_idx])
        points_tmp = temp_df.iloc[i].values[head_num:].reshape([points_num, points_data_len])
        if i == 0 and start_num < seq_num[0]:
            points_tmp = points_tmp[int((seq_num[0]-start_num)*points_num):, :]
        if i == len(temp_df)-1 and end_num > seq_num[1]:
            points_tmp = points_tmp[:int((seq_num[1] - end_num + 1) * points_num), :]
        points_data = points_tmp if points_data is None else np.vstack([points_data, points_tmp])
    # if time_num is not None:
    #     points_data = points_data[(points_data[:, 0] >= time_num[0]) & (points_data[:, 0] <= time_num[1])]
    tmp_points = points_data[:, x_idx:x_idx + 3]
    return tmp_points


def get_rgb(cur_depth, max_depth, min_depth):
    scale = (max_depth - min_depth) / 10
    if cur_depth < min_depth:
        result_r = 0
        result_g = 0
        result_b = 0xff
    elif cur_depth < min_depth + scale:
        result_r = 0
        result_g = int((cur_depth - min_depth) / scale * 255) & 0xff
        result_b = 0xff
    elif cur_depth < min_depth + scale * 2:
        result_r = 0
        result_g = 0xff
        result_b = (0xff - int((cur_depth - min_depth - scale) / scale * 255)) & 0xff
    elif cur_depth < min_depth + scale * 4:
        result_r = int((cur_depth - min_depth - scale * 2) / scale * 255) & 0xff
        result_g = 0xff
        result_b = 0
    elif cur_depth < min_depth + scale * 7:
        result_r = 0xff
        result_g = (0xff - int((cur_depth - min_depth - scale * 4) / scale * 255)) & 0xff
        result_b = 0
    elif cur_depth < min_depth + scale * 10:
        result_r = 0xff
        result_g = 0
        result_b = int((cur_depth - min_depth - scale * 7) / scale * 255) & 0xff
    else:
        result_r = 0xff
        result_g = 0
        result_b = 0xff
    return [result_r, result_g, result_b]


def project_points(points, cam_in_para, cam_ex_para):
    # points 10000*3
    # output uv 2*10000
    points = np.hstack([points, np.ones([points.shape[0], 1])])
    # tmp_points = points[0,:]
    coordinate = points.T.astype(np.float64)
    matrixIn = cam_in_para.Matrix
    matrixOut = np.hstack([cam_ex_para.R, cam_ex_para.T.reshape([3,1])])
    result = np.matmul(matrixIn, np.matmul(matrixOut, coordinate))
    u, v, depth = result[0, :], result[1, :], result[2, :]
    theoryUV = np.vstack([u/depth, v/depth])
    return theoryUV, depth