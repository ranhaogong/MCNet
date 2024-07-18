from os.path import join, exists, dirname, abspath
import numpy as np
import colorsys, random, os, sys
import open3d as o3d
from helper_ply import read_ply, write_ply
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import nearest_neighbors.lib.python.nearest_neighbors as nearest_neighbors


class ConfigSensatUrban:
    k_n = 25  # KNN
    num_layers = 5  # Number of layers
    # num_points = 1024 * 16  # Number of input points
    num_points = 1024 * 32  # Number of input points
    drop_num = 1 / 16
    num_classes = 13   # Number of valid classes
    sub_grid_size = 0.2  # preprocess_parameter
    # sub_grid_size = 0.06  # preprocess_parameter
    batch_n = 1 * 1

    num_n = 1024 * 16
    # volex = [0.2, 1, 1024, 256, 64, 16]
    num_po = [1024 * 16, 4096, 1024, 256, 64, 16]

    #batch_size = 4 * 2  # batch_size during training
    batch_size = 4  # batch_size during training
    val_batch_size = 10 * 2  # batch_size during validation and test
    train_steps = 600  # Number of steps per epochs
    val_steps = 200  # Number of validation steps per epoch

    b_n = batch_n * batch_size

    sub_sampling_ratio = [4, 4, 4, 4, 4, 6]  # sampling ratio of random sampling at each layer
    # d_out = [16, 64, 128, 256, 512]  # feature dimension
    # d_out = [16, 48, 96, 192, 384]
    # d_out = [16, 32, 64, 128, 256]
    d_out = [16, 64, 128, 256, 512]
    # d_out = [16, 64, 128, 256, 512, 1024]
    # d_out = [16, 32, 64, 128, 256]
    radio = [0.3, 0.6, 0.9, 1.2]
    token = [5000000, 500000, 150000, 50000]
    # token = [150, 150, 150, 150]
    dx = [1,2,4,8]

    num_head = 4
    v_r = [0.2, 0.4, 0.6, 0.8, 1]
    noise_init = 3  # noise initial parameter
    max_epoch = 1000  # maximum epoch during training
    learning_rate = 0.01  # initial learning rate
    lr_decays = {i: 0.95 for i in range(0, 1000)}  # decay rate of learning rate

    train_sum_dir = 'train_log_SensatUrban'
    saving = True
    saving_path = None

    num_sample_layers = 1
    n_layer_sample_ratio = [1, 2, 2, 2, 2]
    weight = [1, 1, 1, 1, 1]


class DataProcessing:

    def downsample(self, label, radio):

        n = int(len(label) / radio)
        n_s = n
        label_idx = np.zeros_like(label)
        for i in range(13):
            nn = np.arange(len(label))
            idx = (label == i)
            num_i = np.sum(idx == True)

            idx = nn[idx]

            if num_i < 4:
                n = n - num_i
                continue
            if n < 4:
                continue
            idx_i = np.arange(num_i)
            nn = int(num_i / 4)
            np.random.shuffle(idx_i)
            idx_i = idx[idx_i[: nn]]
            label_idx[idx_i] += 1
            print(num_i, len(idx_i))
        n = np.sum(label_idx == 1)
        if n < n_s:
            print(n, n_s)

            n = n_s - n

            nn = np.arange(len(label))
            idx = (label_idx == 0)
            nn = nn[idx]
            np.random.shuffle(nn)
            nn = nn[:n]
            label_idx[nn] += 1


        return label_idx


    def semantic_downsample(self, xyz, label, radio):
        xyz = np.array(xyz)
        label = np.array(label)
        batch = xyz.shape[0]
        xyz = self.downsample(label[0], radio)
        xyz = np.expand_dims(xyz, axis=1)
        for i in range(batch-1):
            xyz_i = self.downsample(label[i + 1], radio)
            xyz_i = np.expand_dims(xyz_i, axis=1)
            xyz = np.concatenate([xyz, xyz_i], axis=1)

        return xyz





    @staticmethod
    def prob_neig(logits, label_loss, label_neig):
        logits = np.array(logits)  # b, n, 9, 13
        label_loss = np.array(label_loss)
        label_neig = np.array(label_neig)

        label_loss_ = (label_loss >= 0.5)
        label_loss_ = label_loss_.astype(np.int32)
        label_loss = label_loss_ * label_loss

        logits = label_loss * logits

        zeors = np.zeros_like(logits[:,:,0,:])
        for i in range(label_neig.shape[0]):


            for j in range(label_neig.shape[1]):

                for k in range(label_neig.shape[2]):
                    idx = label_neig[i, j ,k]
                    zeors[i, idx] += logits[i, j, k]

        return zeors

    @staticmethod
    def get_label_from_neigh(labels, b, n):
        labels = np.array(labels)
        b = int(b)
        n = int(n)
        # b, n, 9, 13
        labels = np.sum(labels, axis=2)  # b, n, 13

        labels = np.argmax(labels, axis=2).reshape([b, n])
        labels = labels.astype(np.int32)

        return labels



    @staticmethod
    def label_nei(label):
        label = np.array(label)
        d = label.shape[-1]
        label_i = label[:, :, 0]
        label_i = np.expand_dims(label_i, [2])
        label_i = np.tile(label_i, (1, 1, d))
        label = (label == label_i)
        label = label.astype(np.int32)

        return label

    @staticmethod
    def label_nei_loss(label, batch_label):
        label = np.array(label)
        d = label.shape[-1]
        label_i = batch_label
        label_i = np.expand_dims(label_i, [2])
        label_i = np.tile(label_i, (1,1,d))
        label = (label == label_i)
        label = label.astype(np.float32)

        return label

    @staticmethod
    def min_xyz(xyz):
        xyz = np.array(xyz)
        xyz = xyz - np.expand_dims(np.min(xyz, axis=1), axis=1)

        return xyz

    @staticmethod
    def get_num_class_from_label(labels, total_class):
        num_pts_per_class = np.zeros(total_class, dtype=np.int32)
        # original class distribution
        val_list, counts = np.unique(labels, return_counts=True)
        for idx, val in enumerate(val_list):
            num_pts_per_class[val] += counts[idx]
        # for idx, nums in enumerate(num_pts_per_class):
        #     print(idx, ':', nums)
        return num_pts_per_class

    @staticmethod
    def knn_search(support_pts, query_pts, k):
        """
        :param support_pts: points you have, B*N1*3
        :param query_pts: points you want to know the neighbour index, B*N2*3
        :param k: Number of neighbours in knn search
        :return: neighbor_idx: neighboring points indexes, B*N2*k
        """

        neighbor_idx = nearest_neighbors.knn_batch(support_pts, query_pts, k, omp=True)
        return neighbor_idx.astype(np.int32)

    @staticmethod
    def knn_search_0(support_pts, query_pts, k):
        """
        :param support_pts: points you have, B*N1*3
        :param query_pts: points you want to know the neighbour index, B*N2*3
        :param k: Number of neighbours in knn search
        :return: neighbor_idx: neighboring points indexes, B*N2*k
        """

        neighbor_idx = nearest_neighbors.knn_batch(support_pts, query_pts, k, omp=True)

        # neighbor_idx = np.array(neighbor_idx)
        # np.random.shuffle(neighbor_idx[:, :, 1:].T)


        return neighbor_idx.astype(np.int32)

    @staticmethod
    def L12345(f_label_list):
        """
        :param support_pts: points you have, B*N1*3
        :param query_pts: points you want to know the neighbour index, B*N2*3
        :param k: Number of neighbours in knn search
        :return: neighbor_idx: neighboring points indexes, B*N2*k
        """

        # f_out = tf.reduce_sum(f_out, axis=2, keepdims=True) / self.config.num_sample_layers
        # f_out = f_label_list[0]
        # f_out = f_label_list[0] + f_label_list[1] + f_label_list[2]
        # f_out = f_label_list[0] + f_label_list[1] + f_label_list[2] + f_label_list[3]
        f_out = f_label_list[0] + f_label_list[1] + f_label_list[2] + f_label_list[3] + f_label_list[4]
        return f_out

    @staticmethod
    def voxel_data(point_cloud, leaf_size):
        filtered_points = []
        # 作业3
        # 屏蔽开始
        # step1 计算边界点
        x_max, y_max, z_max = np.amax(point_cloud, axis=0)  # 计算 x,y,z三个维度的最值
        x_min, y_min, z_min = np.amin(point_cloud, axis=0)
        # print(x_max, y_max, z_max)
        # print(x_min, y_min, z_min)
        # step2 确定体素的尺寸
        size_r = leaf_size

        # step3 计算每个 volex的维度
        Dx = (x_max - x_min) / size_r
        Dy = (y_max - y_min) / size_r
        Dz = (z_max - z_min) / size_r
        # step4 计算每个点在volex grid内每一个维度的值
        h = list()
        for i in range(len(point_cloud)):
            hx = np.floor((point_cloud[i][0] - x_min) // size_r)
            hy = np.floor((point_cloud[i][1] - y_min) // size_r)
            hz = np.floor((point_cloud[i][2] - z_min) // size_r)
            h.append(hx + hy * Dx + hz * Dx * Dy)
        # step5 对h值进行排序
        h = np.array(h)
        h_indice = np.argsort(h)  # 提取索引
        h_sorted = h[h_indice]  # 升序
        count = 0  # 用于维度的累计
        # 将h值相同的点放入到同一个grid中，并进行筛选
        for i in range(len(h_sorted) - 1):  # 0-9999个数据点
            if h_sorted[i] == h_sorted[i + 1]:  # 当前的点与后面的相同，放在同一个volex grid中
                continue
            else:
                point_idx = h_indice[count: i + 1]
                filtered_points.append(np.mean(point_cloud[point_idx], axis=0))  # 取同一个grid的均值
                count = i

            # 屏蔽结束

            # 把点云格式改成array，并对外返回
        filtered_points = np.array(filtered_points, dtype=np.float32)
        return filtered_points

    @staticmethod
    def data_load(xyz, neighter):
        b = neighter.shape[0]
        neighter = np.array(neighter).reshape(b, -1)

        data = np.array(xyz[0, neighter[0, :], :]).reshape(1, -1, xyz.shape[-1])
        for i in range(b-1):
            data_ = np.array(xyz[i+1, neighter[i+1, :], :]).reshape(1, -1, xyz.shape[-1])
            data = np.append(data, data_, axis=0)


        return data

    @staticmethod
    def test1(x, y, n):
        print(x.shape, y.shape, n, 'jjjjjjjjjjjjjjjjjjjjjj')
        a = 1
        return a

    @staticmethod
    def test2(xyz,b):

        print(xyz.shape, '15631453615314131351313515')
        print(b.shape, '15631453615314131351313515')
        # print(c.shape, '15631453615314131351313515')
        # print(d.shape, '15631453615314131351313515')
        # print(e.shape, '15631453615314131351313515')
        a = 1
        return a

    @staticmethod
    def test3(x):

        print(x, '15631453615314131351313515')

        a = 1
        return a



    @staticmethod
    def data_aug(xyz, color, labels, idx, num_out, dropout_p):
        num_in = len(xyz)
        dup = np.random.choice(num_in, num_out - num_in)
        xyz_dup = xyz[dup, ...]
        xyz_aug = np.concatenate([xyz, xyz_dup], 0)
        color_dup = color[dup, ...]
        color_aug = np.concatenate([color, color_dup], 0)

        idx_dup = list(range(num_in)) + list(dup)
        idx_aug = idx[idx_dup]
        label_aug = labels[idx_dup]

        dropout_p_dup = dropout_p[dup, ...]
        dropout_p_dup = np.concatenate([dropout_p, dropout_p_dup], 0)
        return xyz_aug, color_aug, idx_aug, label_aug, dropout_p_dup

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):
        # gather the coordinates or features of neighboring points
        batch_size = tf.shape(pc)[0] # B
        num_points = tf.shape(pc)[1] # N
        d = pc.get_shape()[2].value # 32
        index_input = tf.reshape(neighbor_idx, shape=[batch_size, -1])
        features = tf.batch_gather(pc, index_input)
        features = tf.reshape(features, [batch_size, num_points, tf.shape(neighbor_idx)[-1], d])
        return features

    @staticmethod
    def shuffle_idx(x):
        # random shuffle the index
        # a = len(x)
        # b = ConfigSensatUrban.num_points
        # if a > b:
        #     idx = np.arange(a)
        #     np.random.shuffle(idx[:b])
        #     np.random.shuffle(idx[b:])
        # else:
        #     idx = np.arange(a)
        #     np.random.shuffle(idx)

        idx = np.arange(len(x))
        np.random.shuffle(idx)
        return x[idx]

    @staticmethod
    def shuffle_idx_new(x):

        for i in range(x.shape[0]):
            temp = x[i]
            np.random.shuffle(temp)
            x[i] = temp



        return x

    @staticmethod
    def xyz_feature_shuffle(xyz):
        idx = np.arange(np.shape(xyz)[1])
        np.random.shuffle(idx)
        return xyz[:, idx, :]

    @staticmethod
    def shuffle_list(data_list):
        indices = np.arange(np.shape(data_list)[0])
        np.random.shuffle(indices)
        data_list = data_list[indices]
        return data_list

    @staticmethod
    def grid_sub_sampling(points, features=None, labels=None, grid_size=0.1, verbose=0):
        """
        CPP wrapper for a grid sub_sampling (method = barycenter for points and features
        :param points: (N, 3) matrix of input points
        :param features: optional (N, d) matrix of features (floating number)
        :param labels: optional (N,) matrix of integer labels
        :param grid_size: parameter defining the size of grid voxels
        :param verbose: 1 to display
        :return: sub_sampled points, with features and/or labels depending of the input
        """

        if (features is None) and (labels is None):
            return cpp_subsampling.compute(points, sampleDl=grid_size, verbose=verbose)
        elif labels is None:
            return cpp_subsampling.compute(points, features=features, sampleDl=grid_size, verbose=verbose)
        elif features is None:
            return cpp_subsampling.compute(points, classes=labels, sampleDl=grid_size, verbose=verbose)
        else:
            return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=grid_size,
                                           verbose=verbose)

    @staticmethod
    def grid_sub_sampling_1(points, features, grid_size=0.1):
        verbose = 0
        return cpp_subsampling.compute(points, features=features, sampleDl=grid_size, verbose=verbose)

    @staticmethod
    def shuffle_idx_nei(x):

        x = np.array(x)

        np.random.shuffle(x[:, :, 9:].T)
        return x.astype(np.int32)


    @staticmethod
    def grid_sub_sample(points, features, grid_size=0.1, verbose=0):
        """
        CPP wrapper for a grid sub_sampling (method = barycenter for points and features
        :param points: (N, 3) matrix of input points
        :param features: optional (N, d) matrix of features (floating number)
        :param labels: optional (N,) matrix of integer labels
        :param grid_size: parameter defining the size of grid voxels
        :param verbose: 1 to display
        :return: sub_sampled points, with features and/or labels depending of the input
        """
        # point, feature = cpp_subsampling.compute(points, features=features, sampleDl=grid_size, verbose=verbose)

        batch = points.shape[0]
        f = []
        b = np.array([0])
        for i in range(batch):
            point, feature = cpp_subsampling.compute(points[i, :, :], features=features[i, :, :], sampleDl=grid_size, verbose=verbose)
            filtered_points_i = np.concatenate([point, feature], 1)
            # filtered_points_i = voxel_data(points[i, :, :], grid_size, features[i, :, :])
            # np.concatenate([xyz, xyz_dup], 0)
            f.append(filtered_points_i)
            b = np.append(b, len(filtered_points_i))
        b_max = np.max(b)
        for j in range(batch):
            if b[j + 1] < b_max:
                f[j] = data_augm(f[j], b_max)

        filtered_points = np.array(f[0].reshape(1, -1, 6))

        for k in range(batch - 1):
            filtered_points_i = np.array(f[k + 1].reshape(1, -1, 6))

            filtered_points = np.append(filtered_points, filtered_points_i, axis=0)

        return filtered_points

    @staticmethod
    def read_ply_data_test(path, with_rgb=True, with_label=True):
        data = read_ply(path)
        xyz = np.vstack((data['x'], data['y'], data['z'])).T
        if with_rgb and with_label:
            rgb = np.vstack((data['red'], data['green'], data['blue'])).T
            labels = data['class']
            return xyz.astype(np.float32), rgb.astype(np.float32), labels.astype(np.uint8)
        elif with_rgb and not with_label:
            rgb = np.vstack((data['red'], data['green'], data['blue'])).T
            return xyz.astype(np.float32), rgb.astype(np.float32)
        elif not with_rgb and with_label:
            labels = data['class']
            return xyz.astype(np.float32), labels.astype(np.uint8)
        elif not with_rgb and not with_label:
            return xyz.astype(np.float32)

    @staticmethod
    def voxel_filter(b_point_cloud, leaf_size):
        batch = b_point_cloud.shape[0]
        f = []
        b = np.array([0])

        for i in range(batch):
            filtered_points_i = voxel_data(b_point_cloud[i, :, :], leaf_size)
            f.append(filtered_points_i)
            b = np.append(b, len(filtered_points_i))
        b_max = np.max(b)

        for j in range(batch):
            if b[j + 1] < b_max:
                f[j] = data_augm(f[j], b_max)

        filtered_points = np.array(f[0].reshape(1, -1, 6))

        for k in range(batch - 1):
            filtered_points_i = np.array(f[k + 1].reshape(1, -1, 6))

            filtered_points = np.append(filtered_points, filtered_points_i, axis=0)

        return filtered_points
    @staticmethod
    def IoU_from_confusions(confusions):
        """
        Computes IoU from confusion matrices.
        :param confusions: ([..., n_c, n_c] np.int32). Can be any dimension, the confusion matrices should be described by
        the last axes. n_c = number of classes
        :return: ([..., n_c] np.float32) IoU score
        """

        # Compute TP, FP, FN. This assume that the second to last axis counts the truths (like the first axis of a
        # confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
        TP = np.diagonal(confusions, axis1=-2, axis2=-1)
        TP_plus_FN = np.sum(confusions, axis=-1)
        TP_plus_FP = np.sum(confusions, axis=-2)

        # Compute IoU
        IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-6)

        # Compute mIoU with only the actual classes
        mask = TP_plus_FN < 1e-3
        counts = np.sum(1 - mask, axis=-1, keepdims=True)
        mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)

        # If class is absent, place mIoU in place of 0 IoU to get the actual mean later
        IoU += mask * mIoU
        return IoU

    @staticmethod
    def read_ply_data(path, with_rgb=True, with_label=True):
        data = read_ply(path)
        xyz = np.vstack((data['x'], data['y'], data['z'])).T
        if with_rgb and with_label:
            rgb = np.vstack((data['red'], data['green'], data['blue'])).T
            labels = data['class']
            return xyz.astype(np.float32), rgb.astype(np.uint8), labels.astype(np.uint8)
        elif with_rgb and not with_label:
            rgb = np.vstack((data['red'], data['green'], data['blue'])).T
            return xyz.astype(np.float32), rgb.astype(np.uint8)
        elif not with_rgb and with_label:
            labels = data['class']
            return xyz.astype(np.float32), labels.astype(np.uint8)
        elif not with_rgb and not with_label:
            return xyz.astype(np.float32)

    @staticmethod
    def random_sub_sampling(points, features=None, labels=None, sub_ratio=10, verbose=0):
        num_input = np.shape(points)[0]
        num_output = num_input // sub_ratio
        idx = np.random.choice(num_input, num_output)
        if (features is None) and (labels is None):
            return points[idx]
        elif labels is None:
            return points[idx], features[idx]
        elif features is None:
            return points[idx], labels[idx]
        else:
            return points[idx], features[idx], labels[idx]


    @staticmethod
    def get_class_weights(num_per_class, name='sqrt'):
        # # pre-calculate the number of points in each category
        # frequency = num_per_class / float(sum(num_per_class))
        frequency = np.array(num_per_class)

        # num_class = frequency
        # a = frequency
        # max = np.max(num_class)
        # num_class = num_class / max
        # num_class = np.sqrt(num_class)
        # min = 1 / num_class
        # tt = min * a
        # tt = tt.astype(np.int)
        # frequency = tt

        # frequency = frequency / np.sum(frequency)
        frequency = frequency / np.max(frequency)

        if name == 'sqrt' or name == 'lovas':
            # ce_label_weight = 1 / np.sqrt(frequency)
            ce_label_weight = 1 / np.sqrt(frequency)
            # ce_label_weight = np.log(ce_label_weight + 1)
        elif name == 'wce':
            ce_label_weight = 1 / (frequency + 0.02)
        else:
            raise ValueError('Only support sqrt and wce')
        # ce_label_weight[0] = 0
        return np.expand_dims(ce_label_weight, axis=0)

    @staticmethod
    def dropout_points_or(queried_idx, label, class_num, num):

        per_class = np.zeros(len(class_num), dtype=np.int32)
        val_list, counts = np.unique(label, return_counts=True)
        for idx, val in enumerate(val_list):
            per_class[val] += counts[idx]
        per_class = per_class / np.sum(per_class)

        class_num = np.array(class_num)
        class_num = class_num / np.sum(class_num)

        class_num = class_num * per_class

        class_num = (class_num / np.sum(class_num) * num).astype(np.int)
        class_num[val_list[-1]] = num - np.sum(class_num[:val_list[-1]-1])

        num_range = np.arange(len(label))
        num_zero = np.zeros(len(label))
        for i in range(len(class_num)):
            if class_num[i] > 0:
                idx = (label == i)
                num_zero[num_range[idx][:class_num[i]]] = 1
        idx = (num_zero == 0)

        return queried_idx[idx]

    @staticmethod
    def dropout_points(queried_idx, label, class_num, num):

        class_num = np.array(class_num)
        class_num = np.sqrt(class_num / np.sum(class_num))
        class_num = class_num / np.max(class_num)
        class_num = class_num[label]
        weight = np.random.rand(len(label)) * class_num

        idx = np.argsort(weight)
        idx = idx[int(len(label) - num):]
        ones = np.ones(len(queried_idx))
        ones[idx] = 0

        return ones.astype(np.int32)


class Plot:
    @staticmethod
    def random_colors(N, bright=True, seed=0):
        brightness = 1.0 if bright else 0.7
        hsv = [(0.15 + i / float(N), 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.seed(seed)
        random.shuffle(colors)
        return colors

    @staticmethod
    def draw_pc(pc_xyzrgb):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pc_xyzrgb[:, 0:3])
        if pc_xyzrgb.shape[1] == 3:
            o3d.visualization.draw_geometries([pc])
            return 0
        if np.max(pc_xyzrgb[:, 3:6]) > 20:  ## 0-255
            pc.colors = o3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6] / 255.)
        else:
            pc.colors = o3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6])

        o3d.geometry.PointCloud.estimate_normals(pc)
        o3d.visualization.draw_geometries([pc], width=1000, height=1000)
        return 0

    @staticmethod
    def draw_pc_sem_ins(pc_xyz, pc_sem_ins, plot_colors=None):
        # only visualize a number of points to save memory
        if plot_colors is not None:
            ins_colors = plot_colors
        else:
            # ins_colors = Plot.random_colors(len(np.unique(pc_sem_ins)) + 1, seed=1)
            ins_colors = [[85, 107, 47],  # ground -> OliveDrab
                          [0, 255, 0],  # tree -> Green
                          [255, 165, 0],  # building -> orange
                          [41, 49, 101],  # Walls ->  darkblue
                          [0, 0, 0],  # Bridge -> black
                          [0, 0, 255],  # parking -> blue
                          [255, 0, 255],  # rail -> Magenta
                          [200, 200, 200],  # traffic Roads ->  grey
                          [89, 47, 95],  # Street Furniture  ->  DimGray
                          [255, 0, 0],  # cars -> red
                          [255, 255, 0],  # Footpath  ->  deeppink
                          [0, 255, 255],  # bikes -> cyan
                          [0, 191, 255]  # water ->  skyblue
                          ]

        ##############################
        sem_ins_labels = np.unique(pc_sem_ins)
        sem_ins_bbox = []
        Y_colors = np.zeros((pc_sem_ins.shape[0], 3))
        for id, semins in enumerate(sem_ins_labels):
            valid_ind = np.argwhere(pc_sem_ins == semins)[:, 0]
            if semins <= -1:
                tp = [0, 0, 0]
            else:
                if plot_colors is not None:
                    tp = ins_colors[semins]
                else:
                    tp = ins_colors[id]

            Y_colors[valid_ind] = tp

            ### bbox
            valid_xyz = pc_xyz[valid_ind]

            xmin = np.min(valid_xyz[:, 0]);
            xmax = np.max(valid_xyz[:, 0])
            ymin = np.min(valid_xyz[:, 1]);
            ymax = np.max(valid_xyz[:, 1])
            zmin = np.min(valid_xyz[:, 2]);
            zmax = np.max(valid_xyz[:, 2])
            sem_ins_bbox.append(
                [[xmin, ymin, zmin], [xmax, ymax, zmax], [min(tp[0], 1.), min(tp[1], 1.), min(tp[2], 1.)]])

        Y_semins = np.concatenate([pc_xyz[:, 0:3], Y_colors], axis=-1)
        Plot.draw_pc(Y_semins)
        return Y_semins

    @staticmethod
    def save_ply_o3d(data, save_name):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data[:, 0:3])
        if np.shape(data)[1] == 3:
            o3d.io.write_point_cloud(save_name, pcd)
        elif np.shape(data)[1] == 6:
            if np.max(data[:, 3:6]) > 20:
                pcd.colors = o3d.utility.Vector3dVector(data[:, 3:6] / 255.)
            else:
                pcd.colors = o3d.utility.Vector3dVector(data[:, 3:6])
            o3d.io.write_point_cloud(save_name, pcd)
        return


def voxel_data(point_cloud, leaf_size, feature):
    filtered_points = []
    filtered_features = []
    # 作业3
    # 屏蔽开始
    # step1 计算边界点
    x_max, y_max, z_max = np.amax(point_cloud, axis=0)  # 计算 x,y,z三个维度的最值
    x_min, y_min, z_min = np.amin(point_cloud, axis=0)
    # print(x_max, y_max, z_max)
    # print(x_min, y_min, z_min)
    # step2 确定体素的尺寸
    size_r = leaf_size

    # step3 计算每个 volex的维度
    Dx = (x_max - x_min) / size_r
    Dy = (y_max - y_min) / size_r
    Dz = (z_max - z_min) / size_r
    # step4 计算每个点在volex grid内每一个维度的值
    h = list()
    for i in range(len(point_cloud)):
        hx = np.floor((point_cloud[i][0] - x_min) // size_r)
        hy = np.floor((point_cloud[i][1] - y_min) // size_r)
        hz = np.floor((point_cloud[i][2] - z_min) // size_r)
        h.append(hx + hy * Dx + hz * Dx * Dy)
    # step5 对h值进行排序
    h = np.array(h)
    h_indice = np.argsort(h)  # 提取索引
    h_sorted = h[h_indice]  # 升序
    count = 0  # 用于维度的累计
    # 将h值相同的点放入到同一个grid中，并进行筛选
    for i in range(len(h_sorted) - 1):  # 0-9999个数据点
        if h_sorted[i] == h_sorted[i + 1]:  # 当前的点与后面的相同，放在同一个volex grid中
            continue
        else:
            point_idx = h_indice[count: i + 1]
            # filtered_points.append(np.mean(point_cloud[point_idx], axis=0))  # 取同一个grid的均值
            a = np.random.randint(len(point_idx))
            filtered_points.append(point_cloud[point_idx[a]])
            filtered_features.append(feature[point_idx[a]])
            count = i

        # 屏蔽结束

        # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float32)
    filtered_features = np.array(filtered_features, dtype=np.float32)

    filtered_points = np.concatenate([filtered_points, filtered_features], 1)

    return filtered_points

def data_augm(xyz, num_out):
    num_in = len(xyz)
    dup = np.random.choice(num_in, num_out - num_in)
    # xyz_dup = xyz[dup, ...]
    xyz_aug = np.concatenate([xyz, xyz[dup, ...]], 0)

    return xyz_aug