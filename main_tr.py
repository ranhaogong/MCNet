from os.path import join, exists, dirname, abspath
# from RandLANet import Network
from rn_t import Network
# from rconv import Network

from tester_SensatUrban import ModelTester
# from tester_SensatUrban_output_label import ModelTester
# from tester_SensatUrban_test import ModelTester
from helper_ply import read_ply
from tool import ConfigSensatUrban as cfg
from tool import DataProcessing as DP
from tool import Plot
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import numpy as np
import tf_util
import time, pickle, argparse, glob, os, shutil, random, gc
# import megengine as mge
# mge.dtr.eviction_threshold = "23GB" # 设置显存阈值为 5GB
# mge.dtr.enable() # 开启 DTR 显存优化


class SensatUrban:
    def __init__(self):
        self.name = 'SensatUrban'
        root_path = 'data_pro/dataset_class13/'  # path to the dataset
        self.path = join(root_path, self.name)

        self.label_to_names = {0: 'Ground', 1: 'High Vegetation', 2: 'Buildings', 3: 'Walls',
                               4: 'Bridge', 5: 'Parking', 6: 'Rail', 7: 'traffic Roads', 8: 'Street Furniture',
                               9: 'Cars', 10: 'Footpath', 11: 'Bikes', 12: 'Water'}
        # self.label_to_names = {0: 'Ground', 1: 'High Vegetation', 2: 'Buildings', 3: 'Walls',
        #                        4: 'Bridge', 5: 'Parking', 6: 'Rail', 7: 'traffic Roads', 8: 'Street Furniture',
        #                        9: 'Cars', 10: 'Footpath', 11: 'Bikes'}
        # self.label_to_names = {0: 'Ground', 1: 'High Vegetation'}
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([])

        self.all_files = np.sort(glob.glob(join(self.path, 'original_block_ply', '*.ply')))
        self.val_file_name = ['birmingham_block_1',
                              'birmingham_block_5',
                              'cambridge_block_10',
                              'cambridge_block_7']

        self.test_file_name = ['birmingham_block_1',
                              'birmingham_block_5',
                              'cambridge_block_10',
                              'cambridge_block_7']
        # self.test_file_name = ['002']

        # self.test_file_name = ['birmingham_block_2', 'birmingham_block_8',
        #                        'cambridge_block_15', 'cambridge_block_22',
        #                        'cambridge_block_16', 'cambridge_block_27',
        #                        ]



        self.use_val = True  # whether use validation set or not

        # initialize
        self.num_per_class = np.zeros(self.num_classes)
        self.val_proj = []
        self.val_labels = []
        self.test_proj = []
        self.test_labels = []
        self.possibility = {'training': [], 'validation': [], 'test': []}
        self.min_possibility = {'training': [], 'validation': [], 'test': []}
        self.org_name = []
        self.input_trees = {'training': [], 'validation': [], 'test': []}
        self.input_nxyz = {'training': [], 'validation': [], 'test': []}
        self.input_labels = {'training': [], 'validation': [], 'test': []}
        self.input_names = {'training': [], 'validation': [], 'test': []}
        self.input_logits = {'training': [], 'validation': [], 'test': []}
        self.test_name = []
        self.load_sub_sampled_clouds(cfg.sub_grid_size)
        for ignore_label in self.ignored_labels:
           self.num_per_class = np.delete(self.num_per_class, ignore_label)

    def load_sub_sampled_clouds(self, sub_grid_size):
        tree_path = join(self.path, 'grid_{:.3f}'.format(sub_grid_size))

        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]
            if cloud_name in self.test_file_name:
                cloud_split = 'test'
                self.org_name.append(cloud_name)
            elif cloud_name in self.val_file_name:
                cloud_split = 'validation'
            else:
                cloud_split = 'training'

            # Name of the input files
            kd_tree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            data = read_ply(sub_ply_file)
            sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
            sub_labels = data['class']

            # sub_logits = np.ones([len(sub_colors), cfg.num_classes]) / 13


            # compute num_per_class in training set
            if cloud_split == 'training':
                self.num_per_class += DP.get_num_class_from_label(sub_labels, self.num_classes)

                # print(self.num_per_class)

            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            self.input_trees[cloud_split] += [search_tree]
            self.input_nxyz[cloud_split] += [sub_colors]
            self.input_labels[cloud_split] += [sub_labels]
            self.input_names[cloud_split] += [cloud_name]
            # self.input_logits[cloud_split] += [sub_logits]

            # size = sub_colors.shape[0] * 4 * 7
            # print('{:s} {:.1f} MB loaded in {:.1f}s'.format(kd_tree_file.split('/')[-1], size * 1e-6, time.time() - t0))

        print(self.num_per_class)

        print('\nPreparing reprojected indices for testing')

        # Get validation and test reprojected indices

        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]

            # val projection and labels
            if cloud_name in self.val_file_name:
                # proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                # with open(proj_file, 'rb') as f:
                #     proj_idx, labels = pickle.load(f)
                # self.val_proj += [proj_idx]
                # self.val_labels += [labels]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))

            # test projection and labels
            if cloud_name in self.test_file_name:
                # self.test_name.append(cloud_name)
                # proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                # with open(proj_file, 'rb') as f:
                #     proj_idx, labels = pickle.load(f)
                # self.test_proj += [proj_idx]
                # self.test_labels += [labels]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))

        # split = 'training'
        # n_split = 0
        # weight_ = self.num_per_class
        # n_class = np.sum(weight_).astype(np.int)
        # pobbb = np.random.rand(n_class)
        # for i, tree in enumerate(self.input_nxyz[split]):
        #
        #     weight_ = self.num_per_class
        #     weight_ = weight_ / np.max(weight_)
        #     weight_ = np.sqrt(weight_)
        #
        #     label = self.input_labels[split][i]
        #     weight_ = weight_[label]
        #     i_split = tree.data.shape[0]
        #     self.possibility[split] += [pobbb[n_split: i_split + n_split] * weight_]
        #     self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]
        #     n_split = n_split + i_split





    def get_batch_gen(self, split):
        if split == 'training':
            num_per_epoch = cfg.train_steps * cfg.batch_size
        elif split == 'validation':
            num_per_epoch = cfg.val_steps * cfg.val_batch_size
            self.possibility[split] = []
            self.min_possibility[split] = []
        else:
            num_per_epoch = cfg.val_steps * cfg.val_batch_size
            self.possibility[split] = []
            self.min_possibility[split] = []

        # Reset possibility

        n_split = 0
        weight_ = self.num_per_class
        n_class = np.sum(weight_).astype(np.int)
        pobbb = np.random.rand(n_class)

        for i, tree in enumerate(self.input_nxyz[split]):
            if split == 'training':
                a = 1
                weight_ = self.num_per_class
                weight_ = weight_ / np.max(weight_)
                weight_ = np.sqrt(weight_)

                label = self.input_labels[split][i]
                weight_ = weight_[label]
                i_split = tree.data.shape[0]

                # self.possibility[split] += [pobbb[n_split: i_split + n_split]]
                self.possibility[split] += [pobbb[n_split: i_split + n_split] * weight_]
                self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]
                n_split = n_split + i_split
            else:

                self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
                self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]

            # self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            # self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]

        def spatially_regular_gen():
            # Generator loop

            drop_prob = np.abs(np.random.randn(num_per_epoch))
            drop_prob_no = np.abs(np.random.randn(num_per_epoch))
            for i in range(num_per_epoch):  # num_per_epoch

                # Choose a random cloud
                cloud_idx = int(np.argmin(self.min_possibility[split]))

                # choose the point with the minimum of possibility as query point
                point_ind = np.argmin(self.possibility[split][cloud_idx])

                # Get points from tree structure
                points = np.array(self.input_trees[split][cloud_idx].data, copy=False)

                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)

                # Add noise to the center point
                # noise = np.random.normal(scale=cfg.noise_init / 10, size=center_point.shape)
                noise = np.random.normal(scale=cfg.noise_init, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)


                # num_points = np.random.uniform(0.125, 1, 64)[0] * cfg.num_points
                queried_idx_ = self.input_trees[split][cloud_idx].query(pick_point, k=int(cfg.num_points))[1][0]
                queried_idx_ = DP.shuffle_idx(queried_idx_)

                # Collect points and colors
                queried_pc_xyz = points[queried_idx_]
                min_point = np.min(queried_pc_xyz, axis=0)
                queried_pc_xyz = queried_pc_xyz - min_point
                # queried_pc_xyz = queried_pc_xyz - pick_point
                queried_pc_colors = self.input_nxyz[split][cloud_idx][queried_idx_]
                queried_pc_labels = self.input_labels[split][cloud_idx][queried_idx_]
                # queried_pc_logits = self.input_logits[split][cloud_idx][queried_idx_]
                queried_pc_logits = np.arange(26).reshape([-1, 13])

                # if drop_prob_no[i] > 0.5 and split == 'training':
                #     noise_en = np.random.normal(0, cfg.sub_grid_size / 2, size=queried_pc_xyz.shape)
                #     queried_pc_xyz = queried_pc_xyz + noise_en

                num_class = np.array(self.num_per_class)
                min = np.min(num_class)
                diyi = num_class / min
                diyi = np.sqrt(diyi)

                max = np.max(num_class)
                dier = num_class / max
                dier = np.sqrt(dier)

                num_class = diyi * dier
                num_class = np.log(num_class + 1)
                # num_class = dier
                ones = num_class[queried_pc_labels]

                dists = np.sum(np.square((points[queried_idx_] - pick_point).astype(np.float32)), axis=1)
                # delta = np.square(1 - dists / np.max(dists))
                if split == 'training':
                    # delta = (1 - dists / np.max(dists)) * ones
                    delta = np.square(1 - dists / np.max(dists)) * ones
                else:
                    # delta = 1 - dists / np.max(dists)
                    delta = np.square(1 - dists / np.max(dists))

                self.possibility[split][cloud_idx][queried_idx_] += delta
                self.min_possibility[split][cloud_idx] = float(np.min(self.possibility[split][cloud_idx]))
                dropout_point_idx = np.ones(len(queried_pc_labels))

                # if split == 'training':
                #     # num_class = np.array(self.num_per_class)
                #     # min = np.min(num_class)
                #     # diyi = num_class / min
                #     # diyi = np.sqrt(diyi)
                #     # diyi = 1 / diyi
                #     # num = sum(diyi[queried_pc_labels])
                #     # if num > len(queried_pc_labels) * cfg.drop_num:
                #     #     num = len(queried_pc_labels) * cfg.drop_num
                #     num = len(queried_pc_labels) * cfg.drop_num
                #     dropout_point_idx = DP.dropout_points(queried_idx_, queried_pc_labels, self.num_per_class, int(num))

                # 补全
                if len(queried_pc_xyz) < cfg.num_points:
                    queried_pc_xyz, queried_pc_colors, queried_idx_, queried_pc_labels, dropout_point_idx = \
                        DP.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx_, cfg.num_points, dropout_point_idx)



                if True:
                    yield (queried_pc_xyz.astype(np.float32),
                           queried_pc_colors.astype(np.float32),
                           queried_pc_labels,
                           queried_idx_.astype(np.int32),
                           np.array([cloud_idx], dtype=np.int32),
                           dropout_point_idx.astype(np.int32),
                           queried_pc_logits.astype(np.float32))

        gen_func = spatially_regular_gen
        gen_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.float32)
        gen_shapes = ([None, 3], [None, 3], [None], [None], [None], [None], [None, 13])
        return gen_func, gen_types, gen_shapes

    @staticmethod
    def get_tf_mapping2():

        def tf_map(batch_xyz, batch_features, batch_labels, batch_pc_idx, batch_cloud_idx, batch_dropout_point_idx, batch_logits):

            batch_col = batch_features
            # batch_col = tf.concat([batch_features, batch_logits], axis=-1)
            batch_features = tf.concat([batch_xyz, batch_features], axis=-1)

            input_points = []
            input_neighbors = []
            input_pools = []
            input_up_samples = []
            input_col = []
            input_end_idx = []
            input_xy_idx = []
            volex = [0.2, 0.8, 3.2, 12.8, 51.2]

            for i in range(cfg.num_layers):

                neighbour_idx = tf.py_func(DP.knn_search, [batch_xyz, batch_xyz, cfg.k_n], tf.int32)

                zeros = tf.expand_dims(tf.zeros_like(batch_xyz[:, :, 2], dtype=tf.float32), axis=2)
                batch_xy = tf.concat([batch_xyz[:, :, 0:2], zeros], axis=-1)
                neighbour_h_idx = tf.py_func(DP.knn_search, [batch_xy, batch_xy, cfg.k_n], tf.int32)
                neighbour_h_idx = tf.py_func(DP.shuffle_idx_nei, [neighbour_h_idx], tf.int32)
                # neighbour_idx = tf.concat([neighbour_idx, neighbour_h_idx[:, :, 1:]], axis=-1)
                input_xy_idx.append(neighbour_h_idx)

                sub_points = batch_xyz[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                pool_i = neighbour_idx[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                sub_col = batch_col[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                up_i = tf.py_func(DP.knn_search, [sub_points, batch_xyz, 1], tf.int32)
                input_points.append(batch_xyz)
                input_neighbors.append(neighbour_idx)
                input_pools.append(pool_i)
                input_up_samples.append(up_i)
                batch_xyz = sub_points
                input_col.append(batch_col)
                batch_col = sub_col

            # batch_dropout_point_idx = tf.expand_dims(batch_dropout_point_idx, axis=2)
            # batch_dropout_point_idx = gather_neighbour(batch_dropout_point_idx, input_neighbors[0])

            input_list = input_points + input_neighbors + input_pools + input_up_samples
            input_list += [batch_features, batch_labels, batch_pc_idx, batch_cloud_idx, batch_dropout_point_idx]
            input_list += input_col
            input_list += input_xy_idx
            input_list += [batch_logits]

            return input_list

        # @staticmethod
        def gather_neighbour(pc, neighbor_idx):
            # gather the coordinates or features of neighboring points
            batch_size = tf.shape(pc)[0]
            num_points = tf.shape(pc)[1]
            d = pc.get_shape()[2].value
            index_input = tf.reshape(neighbor_idx, shape=[batch_size, -1])
            features = tf.batch_gather(pc, index_input)
            features = tf.reshape(features, [batch_size, num_points, cfg.k_n])
            features = tf.reduce_sum(features, axis=-1)
            return features

        return tf_map






    def init_input_pipeline(self):
        print('Initiating input pipelines')
        cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
        gen_function, gen_types, gen_shapes = self.get_batch_gen('training')
        gen_function_val, _, _ = self.get_batch_gen('validation')
        gen_function_test, _, _ = self.get_batch_gen('test')
        self.train_data = tf.data.Dataset.from_generator(gen_function, gen_types, gen_shapes)
        self.val_data = tf.data.Dataset.from_generator(gen_function_val, gen_types, gen_shapes)
        self.test_data = tf.data.Dataset.from_generator(gen_function_test, gen_types, gen_shapes)

        self.batch_train_data = self.train_data.batch(cfg.batch_size)
        self.batch_val_data = self.val_data.batch(cfg.val_batch_size)
        self.batch_test_data = self.test_data.batch(cfg.val_batch_size)
        map_func = self.get_tf_mapping2()

        self.batch_train_data = self.batch_train_data.map(map_func=map_func)
        self.batch_val_data = self.batch_val_data.map(map_func=map_func)
        self.batch_test_data = self.batch_test_data.map(map_func=map_func)

        self.batch_train_data = self.batch_train_data.prefetch(cfg.batch_size)
        self.batch_val_data = self.batch_val_data.prefetch(cfg.val_batch_size)
        self.batch_test_data = self.batch_test_data.prefetch(cfg.val_batch_size)

        iter = tf.data.Iterator.from_structure(self.batch_train_data.output_types, self.batch_train_data.output_shapes)
        self.flat_inputs = iter.get_next()
        self.train_init_op = iter.make_initializer(self.batch_train_data)
        self.val_init_op = iter.make_initializer(self.batch_val_data)
        self.test_init_op = iter.make_initializer(self.batch_test_data)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='the number of GPUs to use [default: 0]')
    parser.add_argument('--mode', type=str, default='train', help='options: train, test, vis')
    FLAGS = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['FLAGS_eager_delete_tensor_gb'] = '1'

    # FLAGS_eager_delete_tensor_gb = 0
    Mode = FLAGS.mode

    shutil.rmtree('__pycache__') if exists('__pycache__') else None
    if Mode == 'train':
        shutil.rmtree('results') if exists('results') else None
        shutil.rmtree('train_log') if exists('train_log') else None
        for f in os.listdir(dirname(abspath(__file__))):
            if f.startswith('log_'):
                os.remove(f)

    dataset = SensatUrban()
    dataset.init_input_pipeline()

    if Mode == 'train':
        model = Network(dataset, cfg)
        model.train(dataset)
        
    elif Mode == 'test':
        cfg.saving = False
        model = Network(dataset, cfg)
        chosen_snapshot = -1
        logs = np.sort([os.path.join('results', f) for f in os.listdir('results') if f.startswith('Log')])
        chosen_folder = logs[-1]
        snap_path = join(chosen_folder, 'snapshots')
        snap_steps = [int(f[:-5].split('-')[-1]) for f in os.listdir(snap_path) if f[-5:] == '.meta']
        chosen_step = np.sort(snap_steps)[-1]
        chosen_snap = os.path.join(snap_path, 'snap-{:d}'.format(chosen_step))
        tester = ModelTester(model, dataset, restore_snap=chosen_snap)
        tester.test(model, dataset)
        shutil.rmtree('train_log') if exists('train_log') else None

    else:

        with tf.Session() as sess:
            
            sess.run(tf.global_variables_initializer())
            sess.run(dataset.train_init_op)
            while True:
                data_list = sess.run(dataset.flat_inputs)
                xyz = data_list[0]
                sub_xyz = data_list[1]
                label = data_list[21]
                Plot.draw_pc_sem_ins(xyz[0, :, :], label[0, :])
