# process txt file with pose
# the pose 12 vector
# 6*1 + 6*1
# 6 :{trans_vec 3*1  rot_vec 3*1}
# rot_vec eular angle
from scipy.spatial.transform import Rotation as R


def convert_euler(r_vec):
    r = R.from_rotvec(list(map(float, r_vec)))
    angle = r.as_euler('zyx', degrees=False)[::-1]
    return ['{:.6f}'.format(i) for i in angle]

filename = 'filename_shuffle_1207.txt'
out_name = '1207_true_pose.txt'
with open(out_name, 'a+') as out:
    with open(filename, 'r') as f:
        for str in f:
        # str = f.readline()
            str1 = str.split(' ')
            base = str1[:6]
            r_vec_1 = str1[6:9]
            r_vec_1_new = convert_euler(r_vec_1)
            t_vec_1 = str1[9:12]
            r_vec_2 = str1[12:15]
            r_vec_2_new = convert_euler(r_vec_2)
            t_vec_2 = str1[15:]
            full_list = base + t_vec_1 + r_vec_1_new + t_vec_2 + r_vec_2_new
            full_str = ' '.join(full_list)
            print(full_str, file=out)

filename = 'train_with_poses.txt'
out_name = 'full_kitti_true_poses.txt'
with open(out_name, 'a+') as out:
    with open(filename, 'r') as f:
        for str in f:
        # str = f.readline()
            str1 = str.split(' ')
            base = str1[:6]
            r_vec_1 = str1[9:12]
            r_vec_1_new = convert_euler(r_vec_1)
            t_vec_1 = str1[6:9]
            r_vec_2 = str1[15:]
            r_vec_2_new = convert_euler(r_vec_2)
            t_vec_2 = str1[12:15]
            full_list = base + t_vec_1 + r_vec_1_new + t_vec_2 + r_vec_2_new
            full_str = ' '.join(full_list)
            print(full_str, file=out)