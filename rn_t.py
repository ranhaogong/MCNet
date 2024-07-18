from os.path import exists, join
from os import makedirs
from sklearn.metrics import confusion_matrix
from tool import DataProcessing as DP
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import tf_util
import time
from tool import DataProcessing as DP
import pandas as pd


def log_out(out_str, f_out):
    f_out.write(out_str + '\n')
    f_out.flush()
    print(out_str)


class Network:
    def __init__(self, dataset, config):
        flat_inputs = dataset.flat_inputs
        self.config = config
        # Path of the result folder
        if self.config.saving:
            if self.config.saving_path is None:
                self.saving_path = time.strftime('results/Log_%Y-%m-%d', time.gmtime())
                self.saving_path = self.saving_path + '_' + dataset.name
            else:
                self.saving_path = self.config.saving_path
            makedirs(self.saving_path) if not exists(self.saving_path) else None

        # 此函数作用是共享变量不过包括tf.get_variable()的变量和tf.Variable()的变量
        with tf.variable_scope('inputs'):
            # dict() 函数用于创建一个字典。
            self.inputs = dict()
            num_layers = self.config.num_layers
            self.inputs['xyz'] = flat_inputs[:num_layers]
            self.inputs['neigh_idx'] = flat_inputs[num_layers: 2 * num_layers]
            self.inputs['sub_idx'] = flat_inputs[2 * num_layers:3 * num_layers]
            self.inputs['interp_idx'] = flat_inputs[3 * num_layers:4 * num_layers]
            self.inputs['features'] = flat_inputs[4 * num_layers]
            self.inputs['labels'] = flat_inputs[4 * num_layers + 1]
            self.inputs['input_inds'] = flat_inputs[4 * num_layers + 2]
            self.inputs['cloud_inds'] = flat_inputs[4 * num_layers + 3]
            self.inputs['dropout'] = flat_inputs[4 * num_layers + 4]

            self.inputs['col'] = flat_inputs[4 * num_layers + 5: 5 * num_layers + 5]
            self.inputs['high'] = flat_inputs[5 * num_layers + 5: 6 * num_layers + 5]

            # self.inputs['logtis'] = flat_inputs[6 * num_layers + 5]

            # self.inputs['neighbour_end_idx'] = flat_inputs[4 * num_layers + 4]
            # self.inputs['label_nei'] = flat_inputs[4 * num_layers + 4]
            # self.inputs['label_loss'] = flat_inputs[4 * num_layers + 5]

            self.labels = self.inputs['labels']
            # 此函数可以理解为形参，用于定义过程，在执行的时候再赋具体的值。
            self.is_training = tf.placeholder(tf.bool, shape=())
            self.training_step = 1
            self.training_epoch = 0
            self.correct_prediction = 0
            self.accuracy = 0
            self.mIou_list = [0]
            self.loss_type = 'sqrt'  # wce, lovas
            self.class_weights = DP.get_class_weights(dataset.num_per_class, self.loss_type)
            self.Log_file = open('log_train_' + dataset.name + '.txt', 'a')

        with tf.variable_scope('layers'):
            # self.logits, self.l1, self.l2, self.l3 = self.inference6(self.inputs, self.is_training)
            # self.logits, self.l1, self.l2 = self.inference9(self.inputs, self.is_training)
            self.logits = self.inference9_glue(self.inputs, self.is_training)

        with tf.variable_scope('loss'):

            # self.confi = tf.reshape(self.confi, [-1, 9])
            # self.confi1 = tf.reshape(self.confi1, [-1, 9])
            self.logits = tf.reshape(self.logits, [-1, self.config.num_classes])
            # self.l1 = tf.reshape(self.l1, [-1, self.config.num_classes])
            # self.l2 = tf.reshape(self.l2, [-1, self.config.num_classes])

            self.labels = tf.reshape(self.labels, [-1])
            self.dropout_idx = tf.reshape(self.inputs['dropout'], [-1])

            # valid_labels0 = tf.reshape(self.inputs['labels'][:, :4096], [-1])
            # valid_labels1 = tf.reshape(self.inputs['labels'][:, :1024], [-1])
            valid_logits = self.logits
            # valid_l1 = self.logits
            # valid_l2 = self.logits

            valid_labels = self.labels

            # valid_idx = tf.squeeze(tf.where(self.dropout_idx))
            # valid_logits_ = tf.gather(self.logits, valid_idx, axis=0)
            # valid_labels_ = tf.gather(self.labels, valid_idx, axis=0)
            # valid_drop = tf.gather(self.dropout_idx, valid_idx, axis=0)


            # self.loss = self.get_loss_org0(valid_logits_, valid_labels_, self.class_weights, valid_l1, valid_l2, valid_labels0, valid_labels1)
            self.loss = self.get_loss_org(valid_logits, valid_labels, self.class_weights)


        with tf.variable_scope('optimizer'):
            self.learning_rate = tf.Variable(config.learning_rate, trainable=False, name='learning_rate')
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.variable_scope('results'):

            # valid_lo = tf.nn.softmax(valid_logits) + tf.nn.softmax(valid_l1) * 0.5 + tf.nn.softmax(valid_l2) * 0.5

            self.correct_prediction = tf.nn.in_top_k(valid_logits, valid_labels, 1)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            # self.prob_logits = tf.nn.softmax(valid_logits) * 0.5 + tf.nn.softmax(valid_l1) * 0.3 + tf.nn.softmax(valid_l2) * 0.2
            self.prob_logits = tf.nn.softmax(self.logits)

            tf.summary.scalar('learning_rate', self.learning_rate)
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)

        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(my_vars, max_to_keep=1)
        c_proto = tf.ConfigProto()
        c_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=c_proto)
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(config.train_sum_dir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def mult_heat_or(self, feature, d_out, name, num_head, neigh, is_training, xyz):

        feature = tf.layers.dense(feature, d_out, activation=None, name=name + 'en_start')

        fea_i = self.gather_neighbour(feature, neigh)
        q_i = tf.layers.dense(feature, d_out // num_head, activation=None, name=name + 'q0or')  # (B, N, 32)
        k_i = tf.layers.dense(fea_i, d_out // num_head, activation=None, name=name + 'k0or')
        v_i = tf.layers.dense(fea_i, d_out // num_head, activation=None, name=name + 'v0or')

        xyz = tf.tile(tf.expand_dims(xyz[:, :, 0, :], axis=2), [1, 1, self.config.k_n, 1]) - xyz
        xyz = tf.layers.dense(xyz, d_out, activation=None, name=name + 'xyz0')
        xyz = tf.nn.leaky_relu(xyz, alpha=0.2)
        xyz = tf.layers.dense(xyz, d_out, activation=None, name=name + 'xyz1')

        # q_i = self.mlp(feature, d_out // num_head, name + 'q0or', is_training)
        # k_i = self.mlp(feature, d_out // num_head, name + 'k0or', is_training)
        # v_i = self.mlp(feature, d_out // num_head, name + 'v0or', is_training)

        q_i = tf.expand_dims(q_i, axis=2)  # (B, N, 1, D)
        # k_i = self.gather_neighbour(k_i, neigh)  # (B, N, 16, D)
        # v_i = self.gather_neighbour(v_i, neigh)  # (B, N, 16, D)

        q_i = tf.matmul(q_i, tf.transpose(k_i, perm=[0, 1, 3, 2]))  # (B, N, 16, 16)
        q_i = tf.nn.softmax(q_i / (d_out // num_head) ** 0.5, axis=-1)
        z_i = tf.squeeze(tf.matmul(q_i, v_i), axis=2)  # (B, N, 32)

        for i in range(num_head - 1):
            q_i = tf.layers.dense(feature, d_out // num_head, activation=None, name=name + 'q0' + str(i))  # (B, N, 32)
            k_i = tf.layers.dense(fea_i, d_out // num_head, activation=None, name=name + 'k0' + str(i))
            v_i = tf.layers.dense(fea_i, d_out // num_head, activation=None, name=name + 'v0' + str(i))
            # q_i = self.mlp(feature, d_out // num_head, name + 'q0or' + str(i), is_training)
            # k_i = self.mlp(feature, d_out // num_head, name + 'k0or' + str(i), is_training)
            # v_i = self.mlp(feature, d_out // num_head, name + 'v0or' + str(i), is_training)

            q_i = tf.expand_dims(q_i, axis=2)  # (B, N, 1, D)
            # k_i = self.gather_neighbour(k_i, neigh)  # (B, N, 16, D)
            # v_i = self.gather_neighbour(v_i, neigh)  # (B, N, 16, D)

            q_i = tf.matmul(q_i, tf.transpose(k_i, perm=[0, 1, 3, 2]))  # (B, N, 16, 16)
            q_i = tf.nn.softmax(q_i / (d_out // num_head) ** 0.5, axis=-1)

            z_i = tf.concat([z_i, tf.squeeze(tf.matmul(q_i, v_i), axis=2)], axis=-1)

        z_i = tf.layers.dense(z_i, d_out, activation=None, name=name + 'end')
        # z_i = tf.layers.batch_normalization(z_i, -1, 0.99, 1e-6, training=is_training)
        z_i = tf.keras.layers.LayerNormalization()(z_i)
        z_i = tf_util.dropout(z_i, keep_prob=0.5, is_training=is_training, scope=name + 'dp2')

        return z_i

    def mult_heat(self, feature, d_out, name, num_head, neigh, is_training, xyz):

        feature = tf.layers.dense(feature, d_out, activation=None, name=name + 'en_start')

        d_out_nei = d_out * 2

        fea_i = self.gather_neighbour(feature, neigh)
        q_i = tf.layers.dense(feature, d_out_nei, activation=None, name=name + 'q0or')  # (B, N, 32)
        k_i = tf.layers.dense(fea_i, d_out_nei, activation=None, name=name + 'k0or')
        v_i = tf.layers.dense(fea_i, d_out_nei, activation=None, name=name + 'v0or')

        xyz = tf.tile(tf.expand_dims(xyz[:, :, 0, :], axis=2), [1, 1, self.config.k_n, 1]) - xyz
        xyz = tf.layers.dense(xyz, d_out_nei, activation=None, name=name + 'xyz0')
        xyz = tf.nn.leaky_relu(xyz, alpha=0.2)
        xyz = tf.layers.dense(xyz, d_out_nei, activation=None, name=name + 'xyz1')

        q_i = tf.tile(tf.expand_dims(q_i, axis=2), [1, 1, self.config.k_n, 1])  # (B, N, 1, D)
        q_i = q_i - k_i + xyz
        q_i = tf.layers.dense(q_i, d_out_nei, activation=None, name=name + 'xyz0q_i')
        q_i = tf.nn.leaky_relu(q_i, alpha=0.2)
        q_i = tf.layers.dense(q_i, d_out_nei, activation=None, name=name + 'xyz1q_i')

        q_i = tf.nn.softmax(q_i / (d_out_nei ** 0.5), axis=-2)

        z_i = tf.reduce_sum(q_i * (v_i + xyz), axis=2)
        z_i = tf.layers.dense(z_i, d_out, activation=None, name=name + 'end')

        # z_i = tf_util.dropout(z_i, keep_prob=0.5, is_training=is_training, scope=name + 'dp2')

        return z_i

    def mult_heat_att_or(self, feature, d_out, name, num_head, neigh, is_training, xyz):

        # feature = tf.layers.dense(feature, d_out, activation=None, name=name + 'en_start')
        # feature = self.mlp(feature, d_out, name + 'start', is_training)

        d_out_nei = d_out * 1

        xyz = tf.tile(tf.expand_dims(xyz[:, :, 0, :], axis=2), [1, 1, self.config.k_n, 1]) - xyz
        # xyz = xyz - tf.tile(tf.reduce_min(xyz, axis=2, keepdims=True), [1, 1, self.config.k_n, 1])
        xyz = tf.layers.dense(xyz, d_out_nei, activation=None, name=name + 'xyz0')
        # xyz = tf.layers.batch_normalization(xyz, -1, 0.99, 1e-6, training=is_training)
        # xyz = tf.layers.dense(xyz, d_out_nei, activation=None, name=name + 'xyz1')

        # q_i = tf.layers.dense(fea_i, d_out_nei, activation=None, name=name + 'fff')  # (B, N, 32)
        # v_i = tf.layers.dense(fea_i, d_out_nei, activation=None, name=name + 'v0or')

        # q_i = tf_util.conv2d(fea_i, d_out_nei, [1, 1], name + 'fff', [1, 1], 'VALID', True, is_training)
        # v_i = tf_util.conv2d(fea_i, d_out_nei, [1, 1], name + 'v0or', [1, 1], 'VALID', True, is_training)
        q_i = self.mlp(feature, d_out_nei, name + 'q0or', is_training)
        v_i = self.mlp(feature, d_out_nei, name + 'v0or', is_training)
        q_i = self.gather_neighbour(q_i, neigh)
        v_i = self.gather_neighbour(v_i, neigh)


        q_i = q_i + xyz
        # q_i = tf.transpose(q_i, [0, 1, 3, 2])
        # q_i = tf.layers.dense(q_i, d_out_nei, activation=None, name=name + 'q0or')  # (B, N, 32)
        # q_i = tf.transpose(q_i, [0, 1, 3, 2])
        # q_i = tf.matmul(v_i, q_i)  # (B, N, 16, 16)
        # q_i = tf.nn.softmax(q_i / (16 ** 0.5), axis=-1)
        # z_i = tf.matmul(q_i, (v_i + xyz))  # (B, N, 32)
        # z_i = z_i[:, :, 0, :]

        # q_i = tf.tile(tf.expand_dims(q_i, axis=2), [1, 1, 16, 1])  # (B, N, 1, D)
        # q_i = q_i + xyz
        # q_i = tf.layers.dense(q_i, d_out_nei, activation=None, name=name + 'xyz0q_i')
        # q_i = tf.nn.leaky_relu(q_i, alpha=0.2)
        # q_i = tf.layers.dense(q_i, d_out_nei, activation=None, name=name + 'xyz1q_i')
        # v_i = tf.layers.dense(v_i + xyz, d_out_nei, activation=None, name=name + 'xyz1v_i')

        # q_i = tf.nn.softmax(q_i / (self.config.k_n ** 0.5), axis=-2)
        q_i = tf.nn.softmax(q_i, axis=-2)

        # q_i = q_i - tf.tile(tf.reduce_min(q_i, axis=2, keepdims=True), [1, 1, self.config.k_n, 1])
        # q_i = q_i / tf.tile(tf.reduce_sum(q_i, axis=2, keepdims=True), [1, 1, self.config.k_n, 1])

        z_i = tf.reduce_sum(q_i * (v_i + xyz), axis=2)
        # z_i = tf.layers.dense(z_i, d_out, activation=None, name=name + 'end')
        # z_i = tf.layers.batch_normalization(z_i, -1, 0.99, 1e-6, training=is_training)
        z_i = self.mlp(z_i, d_out, name + 'end', is_training)
        # z_i = self.nearest_interpolation_new(z_i, interp_idx)

        return z_i

    def mult_heat_att(self, feature, d_out, name, num_head, neigh, is_training, xyz, nn):

        d_out_nei = d_out
        # feature = self.mlp(feature, d_out_nei, name + 'star', is_training)
        num = tf.shape(xyz)[2]

        # xyz = xyz - tf.tile(tf.reduce_min(xyz, axis=2, keepdims=True), [1, 1, num, 1])

        xyz = tf.tile(tf.expand_dims(xyz[:, :, 0, :], axis=2), [1, 1, num, 1]) - xyz
        xyz = tf.layers.dense(xyz, d_out_nei, activation=None, name=name + 'xyz0')
        xyz = tf.nn.leaky_relu(tf.layers.batch_normalization(xyz, -1, 0.99, 1e-6, training=is_training))

        feature = tf.layers.dense(feature, d_out_nei, activation=None, name=name + 'fea0')
        feature = tf.nn.leaky_relu(tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training))
        # q_i = self.mlp(feature, d_out_nei, name + 'q0or', is_training)
        # v_i = self.mlp(feature, d_out_nei, name + 'v0or', is_training)
        # q_i = self.gather_neighbour(feature, neigh) + xyz
        # q_i = tf.transpose(q_i, perm=[0, 1, 3, 2])

        v_i = self.random_gather(feature, neigh)
        q_i = self.random_gather(feature, neigh)
        q_i = tf.concat([q_i, xyz], axis=-1)
        v_i = tf.concat([v_i, xyz], axis=-1)
        q_i = tf.layers.dense(q_i, d_out_nei, activation=None, name=name + 'fff')  # (B, N, 32)
        v_i = tf.layers.dense(v_i, d_out_nei, activation=None, name=name + 'v0or')

        # q_i = q_i + xyz
        q_i = tf.nn.softmax(q_i, axis=-2)
        z_i = tf.reduce_sum(q_i * v_i, axis=2)

        z_i = tf.layers.dense(z_i, d_out, activation=None, name=name + 'end')
        z_i = tf.nn.leaky_relu(tf.layers.batch_normalization(z_i, -1, 0.99, 1e-6, training=is_training))
        # z_i = self.mlp(z_i, d_out, name + 'end', is_training)
        # z_i = self.nearest_interpolation_new(z_i, interp_idx)

        return z_i

    def mult_heat_att_1(self, feature, d_out, name, num_head, neigh, is_training, xyz, nn):

        d_out_nei = d_out
        # feature = self.mlp(feature, d_out_nei, name + 'star', is_training)
        feature = tf.concat([xyz[:, :, 0, :], feature], axis=-1)

        feature = tf.layers.dense(feature, d_out_nei, activation=None, name=name + 'fea0')
        feature = tf.nn.leaky_relu(tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training))

        num = tf.shape(xyz)[2]
        xyz = xyz - tf.tile(tf.reduce_min(xyz, axis=2, keepdims=True), [1, 1, num, 1])
        xyz = tf.tile(tf.expand_dims(xyz[:, :, 0, :], axis=2), [1, 1, num, 1]) - xyz
        # xyz = tf.layers.dense(xyz, d_out_nei, activation=None, name=name + 'xyz0')
        # xyz = tf.nn.leaky_relu(tf.layers.batch_normalization(xyz, -1, 0.99, 1e-6, training=is_training))


        v_i = self.random_gather(feature, neigh)
        q_i = self.random_gather(feature, neigh)
        q_i = tf.concat([xyz, q_i], axis=-1)
        q_i = tf.layers.dense(q_i, d_out_nei, activation=None, name=name + 'fff')  # (B, N, 32)
        # v_i = tf.layers.dense(v_i, d_out_nei, activation=None, name=name + 'v0or')

        # q_i = q_i + xyz
        q_i = tf.nn.softmax(q_i, axis=-2)
        z_i = tf.reduce_sum(q_i * v_i, axis=2)

        z_i = tf.layers.dense(z_i, d_out, activation=None, name=name + 'end')
        z_i = tf.nn.leaky_relu(tf.layers.batch_normalization(z_i, -1, 0.99, 1e-6, training=is_training))
        # z_i = self.mlp(z_i, d_out, name + 'end', is_training)
        # z_i = self.nearest_interpolation_new(z_i, interp_idx)

        return z_i

    def mult_heat_att_2(self, feature, d_out, name, num_head, neigh, is_training, xyz, nn):

        d_out_nei = d_out
        # feature = self.m
        # lp(feature, d_out_nei, name + 'star', is_training)
        num = tf.shape(xyz)[2]

        # xyz = xyz - tf.tile(tf.reduce_min(xyz, axis=2, keepdims=True), [1, 1, num, 1])

        # xyz_r = tf.layers.dense(xyz, 16, activation=None, name=name + 'xyz02')
        # xyz_r = tf.nn.leaky_relu(tf.layers.batch_normalization(xyz_r, -1, 0.99, 1e-6, training=is_training))

        feature = tf.layers.dense(feature, d_out_nei, activation=None, name=name + 'fea0')
        feature = tf.nn.leaky_relu(tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training))


        xyz = tf.tile(tf.expand_dims(xyz[:, :, 0, :], axis=2), [1, 1, num, 1]) - xyz
        # xyz = tf.concat([xyz / 50, re_xyz], axis=-1)
        xyz = tf.layers.dense(xyz, d_out_nei, activation=None, name=name + 'xyz0')
        xyz = tf.nn.leaky_relu(tf.layers.batch_normalization(xyz, -1, 0.99, 1e-6, training=is_training))



        v_i = self.random_gather(feature, neigh)
        q_i = self.random_gather(feature, neigh)
        q_i = tf.concat([q_i, xyz], axis=-1)
        v_i = tf.concat([v_i, xyz], axis=-1)

        # q_i = tf.tile(tf.expand_dims(q_i[:, :, 0, :], axis=2), [1, 1, num, 1]) - q_i
        q_i = tf.layers.dense(q_i, d_out_nei, activation=None, name=name + 'fff')  # (B, N, 32)
        v_i = tf.layers.dense(v_i, d_out_nei, activation=None, name=name + 'v0or')

        # q_i = q_i + xyz
        q_i = tf.nn.softmax(q_i, axis=-2)
        z_i = tf.reduce_sum(q_i * v_i, axis=2)

        z_i = tf.layers.dense(z_i, d_out, activation=None, name=name + 'end')
        z_i = tf.nn.leaky_relu(tf.layers.batch_normalization(z_i, -1, 0.99, 1e-6, training=is_training))
        # z_i = self.mlp(z_i, d_out, name + 'end', is_training)
        # z_i = self.nearest_interpolation_new(z_i, interp_idx)

        return z_i

    def mult_heat_att_3(self, feature, d_out, name, num_head, neigh, is_training, xyz, nn):

        d_out_nei = d_out
        # feature = self.mlp(feature, d_out_nei, name + 'star', is_training)
        num = tf.shape(xyz)[2]

        xyz = tf.tile(tf.expand_dims(xyz[:, :, 0, :], axis=2), [1, 1, num, 1]) - xyz
        xyz = tf.layers.dense(xyz, d_out_nei, activation=None, name=name + 'xyz0')
        xyz = tf.nn.leaky_relu(tf.layers.batch_normalization(xyz, -1, 0.99, 1e-6, training=is_training))


        feature = tf.layers.dense(feature, d_out_nei, activation=None, name=name + 'fea0')
        feature = tf.nn.leaky_relu(tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training))

        v_i = self.random_gather(feature, neigh) + xyz
        q_i = self.random_gather(feature, neigh) + xyz
        # q_i = tf.concat([q_i, xyz], axis=-1)
        # v_i = tf.concat([v_i, xyz], axis=-1)
        q_i = tf.layers.dense(q_i, d_out_nei, activation=None, name=name + 'fff')  # (B, N, 32)
        v_i = tf.layers.dense(v_i, d_out_nei, activation=None, name=name + 'v0o')

        # q_i = q_i + xyz
        q_i = tf.nn.softmax(q_i, axis=-2)
        z_i = tf.reduce_sum(q_i * v_i, axis=2)

        z_i = tf.layers.dense(z_i, d_out, activation=None, name=name + 'end')
        z_i = tf.nn.leaky_relu(tf.layers.batch_normalization(z_i, -1, 0.99, 1e-6, training=is_training))

        return z_i

    def mult_heat_att_glue(self, feature, d_out, name, num_head, neigh, is_training, xyz, nn):

        d_out_nei = d_out
        # feature = self.mlp(feature, d_out_nei, name + 'star', is_training)
        num = tf.shape(xyz)[2]

        xyz = tf.tile(tf.expand_dims(xyz[:, :, 0, :], axis=2), [1, 1, num, 1]) - xyz
        xyz = tf.layers.dense(xyz, d_out_nei, activation=None, name=name + 'xyz0')
        xyz = self.GLUE(tf.layers.batch_normalization(xyz, -1, 0.99, 1e-6, training=is_training))

        # feature = tf.layers.dense(feature, d_out_nei, activation=None, name=name + 'fea0')
        # feature = self.GLUE(tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training))
        feature = self.mlp(feature, d_out_nei, name + 'fea0', is_training)

        v_i = self.random_gather(feature, neigh)
        q_i = self.random_gather(feature, neigh)
        q_i = tf.concat([q_i, xyz], axis=-1)
        v_i = tf.concat([v_i, xyz], axis=-1)
        q_i = tf.layers.dense(q_i, d_out_nei, activation=None, name=name + 'fff')  # (B, N, 32)
        v_i = tf.layers.dense(v_i, d_out_nei, activation=None, name=name + 'v0o')

        # q_i = q_i + xyz
        q_i = tf.nn.softmax(q_i, axis=-2)
        z_i = tf.reduce_sum(q_i * v_i, axis=2)

        z_i = tf.layers.dense(z_i, d_out, activation=None, name=name + 'end')
        z_i = self.GLUE(tf.layers.batch_normalization(z_i, -1, 0.99, 1e-6, training=is_training))

        return z_i

    def mult_heat_en(self, feature, d_out, name, num_head, neigh, is_training, PE):

        feature = self.gather_neighbour(feature, neigh)
        batch = tf.shape(feature)[0]
        nei = tf.shape(feature)[2]
        feature = tf.reshape(feature, [-1, nei, d_out]) + PE
        feature = tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training)

        q_i = tf.layers.dense(feature, d_out // num_head, activation=None, name=name + 'q0or')  # (B, N, 32)
        k_i = tf.layers.dense(feature, d_out // num_head, activation=None, name=name + 'k0or')
        v_i = tf.layers.dense(feature, d_out // num_head, activation=None, name=name + 'v0or')
        q_i = tf.expand_dims(q_i[:, 0, :], axis=1)

        q_i = tf.matmul(q_i, tf.transpose(k_i, perm=[0, 2, 1]))  # (B, 16, 16)
        q_i = tf.nn.softmax(q_i / (d_out // num_head) ** 0.5, axis=-1)
        z_i = tf.matmul(q_i, v_i)  # (B, N, 32)

        for i in range(num_head - 1):

            q_i = tf.layers.dense(feature, d_out // num_head, activation=None, name=name + 'q0' + str(i))  # (B, N, 32)
            k_i = tf.layers.dense(feature, d_out // num_head, activation=None, name=name + 'k0' + str(i))
            v_i = tf.layers.dense(feature, d_out // num_head, activation=None, name=name + 'v0' + str(i))
            q_i = tf.expand_dims(q_i[:, 0, :], axis=1)
            # q_i = tf.expand_dims(q_i, axis=2)  # (B, N, 1, D)
            # k_i = self.gather_neighbour(k_i, neigh)  # (B, N, 16, D)
            # v_i = self.gather_neighbour(v_i, neigh)  # (B, N, 16, D)

            q_i = tf.matmul(q_i, tf.transpose(k_i, perm=[0, 2, 1]))  # (B, 16, 16)
            q_i = tf.nn.softmax(q_i / (d_out // num_head) ** 0.5, axis=-1)

            z_i = tf.concat([z_i, tf.matmul(q_i, v_i)], axis=-1)

        z_i = tf.reshape(z_i, [batch, -1, d_out])
        z_i = tf.layers.dense(z_i, d_out, activation=None, name=name + 'end')
        z_i = tf.layers.batch_normalization(z_i, -1, 0.99, 1e-6, training=is_training)
        # z_i = tf.keras.layers.LayerNormalization()(z_i)
        # z_i = tf_util.dropout(z_i, keep_prob=0.5, is_training=is_training, scope=name + 'dp2')

        return z_i

    def mult_pool(self, feature, d_out, name, num_head, is_training):

        q_i = tf.layers.dense(feature, d_out // num_head, activation=None, name=name + 'q0or')  # (B, N, 32)
        k_i = tf.layers.dense(feature, d_out // num_head, activation=None, name=name + 'k0or')
        v_i = tf.layers.dense(feature, d_out // num_head, activation=None, name=name + 'v0or')
        # q_i = self.mlp(feature, d_out // num_head, name + 'q0or', is_training)
        # k_i = self.mlp(feature, d_out // num_head, name + 'k0or', is_training)
        # v_i = self.mlp(feature, d_out // num_head, name + 'v0or', is_training)

        q_i = tf.matmul(q_i, tf.transpose(k_i, perm=[0, 2, 1]))  # (B, N, 16, 16)
        q_i = tf.nn.softmax(q_i / (d_out // num_head) ** 0.5, axis=-1)
        z_i = tf.matmul(q_i, v_i)  # (B, N, 32)

        for i in range(num_head - 1):

            q_i = tf.layers.dense(feature, d_out // num_head, activation=None, name=name + 'q0' + str(i))  # (B, N, 32)
            k_i = tf.layers.dense(feature, d_out // num_head, activation=None, name=name + 'k0' + str(i))
            v_i = tf.layers.dense(feature, d_out // num_head, activation=None, name=name + 'v0' + str(i))
            # q_i = self.mlp(feature, d_out // num_head, name + 'q0or' + str(i), is_training)
            # k_i = self.mlp(feature, d_out // num_head, name + 'k0or' + str(i), is_training)
            # v_i = self.mlp(feature, d_out // num_head, name + 'v0or' + str(i), is_training)

            q_i = tf.matmul(q_i, tf.transpose(k_i, perm=[0, 2, 1]))  # (B, N, N)
            q_i = tf.nn.softmax(q_i / (d_out // num_head) ** 0.5, axis=-1)

            z_i = tf.concat([z_i, tf.matmul(q_i, v_i)], axis=-1)

        z_i = tf.layers.dense(z_i, d_out, activation=None, name=name + 'end')
        z_i = tf.layers.batch_normalization(z_i, -1, 0.99, 1e-6, training=is_training)
        # z_i = tf.keras.layers.LayerNormalization()(z_i)
        z_i = tf_util.dropout(z_i, keep_prob=0.5, is_training=is_training, scope=name + 'dp1')
        #
        return z_i

    def mult_pool_att(self, feature, d_out, name, num_head, is_training, num, xyz):

        n = tf.shape(xyz)[1]

        xyz = tf.layers.dense(xyz, d_out, activation=None, name=name + 'xyz0')
        xyz = tf.nn.leaky_relu(tf.layers.batch_normalization(xyz, -1, 0.99, 1e-6, training=is_training))
        # xyz = tf.layers.batch_normalization(xyz, -1, 0.99, 1e-6, training=is_training)
        # xyz = tf.layers.dense(xyz, d_out, activation=None, name=name + 'xyz1')
        xyz_ = tf.matmul(xyz, tf.transpose(xyz, perm=[0, 2, 1]))

        q_i = tf.layers.dense(feature, d_out, activation=None, name=name + 'q0or')  # (B, N, 32)
        # k_i = tf.layers.dense(feature, d_out, activation=None, name=name + 'k0or')
        v_i = tf.layers.dense(feature, d_out, activation=None, name=name + 'v0or')
        # q_i = self.mlp(feature, d_out, name + 'q0or', is_training)
        # v_i = self.mlp(feature, d_out, name + 'v0or', is_training)

        q_i = tf.matmul(q_i, tf.transpose(v_i, perm=[0, 2, 1]))  # (B, N, 16, 16)
        q_i = tf.nn.softmax((q_i + xyz_) / (num ** 0.5), axis=-1)
        # q_i = (q_i + xyz_) / (num ** 0.5)
        # q_i = q_i - tf.tile(tf.reduce_min(q_i, axis=-1, keepdims=True), [1, 1, num])
        # q_i = q_i / tf.tile(tf.reduce_sum(q_i, axis=-1, keepdims=True), [1, 1, num])


        z_i = tf.matmul(q_i, v_i + xyz)  # (B, N, 32)
        # z_i = q_i * v_i

        z_i = tf.layers.dense(z_i, d_out, activation=None, name=name + 'end')
        z_i = tf.nn.leaky_relu(tf.layers.batch_normalization(z_i, -1, 0.99, 1e-6, training=is_training))
        # z_i = self.mlp(z_i, d_out, name + 'end', is_training)


        return z_i

    def mult_pool_att_1(self, feature, d_out, name, num_head, is_training, num, xyz):

        n = tf.shape(xyz)[1]

        # xyz = tf.layers.dense(xyz, d_out, activation=None, name=name + 'xyz0')
        # xyz = tf.nn.leaky_relu(tf.layers.batch_normalization(xyz, -1, 0.99, 1e-6, training=is_training))
        # xyz_ = tf.matmul(xyz, tf.transpose(xyz, perm=[0, 2, 1]))

        feature = tf.concat([xyz, feature], axis=-1)
        feature = tf.layers.dense(feature, d_out, activation=None, name=name + 'feature0')
        feature = tf.nn.leaky_relu(tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training))

        q_i = tf.layers.dense(feature, d_out, activation=None, name=name + 'q0or')  # (B, N, 32)

        # v_i = tf.layers.dense(feature, d_out, activation=None, name=name + 'v0or')
        v_i = feature
        # q_i = self.mlp(feature, d_out, name + 'q0or', is_training)
        # v_i = self.mlp(feature, d_out, name + 'v0or', is_training)

        q_i = tf.matmul(q_i, tf.transpose(v_i, perm=[0, 2, 1]))  # (B, N, 16, 16)
        q_i = tf.nn.softmax(q_i / (num ** 0.5), axis=-1)
        # q_i = (q_i + xyz_) / (num ** 0.5)
        # q_i = q_i - tf.tile(tf.reduce_min(q_i, axis=-1, keepdims=True), [1, 1, num])
        # q_i = q_i / tf.tile(tf.reduce_sum(q_i, axis=-1, keepdims=True), [1, 1, num])


        z_i = tf.matmul(q_i, v_i)  # (B, N, 32)
        # z_i = q_i * v_i

        z_i = tf.layers.dense(z_i, d_out, activation=None, name=name + 'end')
        z_i = tf.nn.leaky_relu(tf.layers.batch_normalization(z_i, -1, 0.99, 1e-6, training=is_training))
        # z_i = self.mlp(z_i, d_out, name + 'end', is_training)


        return z_i

    def att_pool(self, feature, d_out, name, neigh, is_training):

        q_i = tf.layers.dense(feature, d_out, activation=None, name=name + 'q0or')  # (B, N, 32)
        k_i = tf.layers.dense(feature, d_out, activation=None, name=name + 'k0or')
        v_i = tf.layers.dense(feature, d_out, activation=None, name=name + 'v0or')

        q_i = tf.expand_dims(q_i, axis=2)  # (B, N, 1, D)
        k_i = self.gather_neighbour(k_i, neigh)  # (B, N, 16, D)
        v_i = self.gather_neighbour(v_i, neigh)  # (B, N, 16, D)

        q_i = tf.matmul(q_i, tf.transpose(k_i, perm=[0, 1, 3, 2]))  # (B, N, 16, 16)
        q_i = tf.nn.softmax(q_i / d_out ** 0.5, axis=-1)
        z_i = tf.squeeze(tf.matmul(q_i, v_i), axis=2)  # (B, N, 32)

        z_i = tf.layers.dense(z_i, d_out, activation=None, name=name + 'end')
        # z_i = tf.layers.batch_normalization(z_i, -1, 0.99, 1e-6, training=is_training)
        z_i = tf.keras.layers.LayerNormalization()(z_i)
        # z_i = tf_util.dropout(z_i, keep_prob=0.5, is_training=is_training, scope=name + 'dp1')

        return z_i

    def mlp(self, feature, d_out, name, is_training):

        feature = tf.expand_dims(feature, [2])
        feature = tf_util.conv2d(feature, d_out, [1, 1], name, [1, 1], 'VALID', True, is_training)
        feature = tf.squeeze(feature, [2])
        # feature = tf_util.dropout(feature, keep_prob=0.5, is_training=is_training, scope=name + 'dp1')

        return feature

    def mlp_no(self, feature, d_out, name, is_training):

        feature = tf.expand_dims(feature, [2])
        feature = tf_util.conv2d(feature, d_out, [1, 1], name, [1, 1], 'VALID', True, is_training, activation_fn=None)
        feature = tf.squeeze(feature, [2])
        # feature = tf_util.dropout(feature, keep_prob=0.5, is_training=is_training, scope=name + 'dp1')

        return feature

    def conv2d2(self, feature, d_out, kernels_size, name, stide, padding, bn, is_training, neigh_idx):

        feature = tf.expand_dims(feature, [2])
        feature = tf_util.conv2d(feature, d_out, [1, 1], name, stide, padding, bn, is_training)
        feature = tf.squeeze(feature, [2])

        return feature

    def GLUE(self, pc):

        weight = 0.5 * (1.0 * tf.erf(pc / tf.sqrt(2.0)))

        return pc * weight

    def inference9_glue(self, inputs, is_training):

        f_encode = []
        f_decode = []

        d_out = self.config.d_out

        feature = tf.concat([inputs['xyz'][0] / 50, inputs['features'][:, :, 3:]], axis=-1)

        feature = tf.layers.dense(feature, d_out[0], activation=None, name='fc0')
        feature = self.GLUE(tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training))
        # feature = tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training)

        num_head = self.config.num_head
        batch = tf.shape(feature)[0]
        layer = self.config.num_layers
        # volex = [0.2, 0.8, 3.2, 12.8, 51.2]
        # volex = [0.5, 1, 5, 10, 50]
        volex = [1, 1, 1, 1, 1]


        for i in range(layer):

            feature_shout = feature



            xyz = self.gather_neighbour(inputs['xyz'][i], inputs['neigh_idx'][i])
            feature0 = self.mult_heat_att_glue(feature, d_out[i] // 2, 'layers' + str(i), num_head, inputs['neigh_idx'][i], is_training, xyz, self.config.k_n * 2 - 1)
            xyz = self.gather_neighbour(inputs['xyz'][i], inputs['high'][i])
            feature1 = self.mult_heat_att_glue(feature, d_out[i] // 2, 'layers0' + str(i), num_head, inputs['high'][i], is_training, xyz, self.config.k_n * 2 - 1)


            feature = tf.concat([feature0, feature1], axis=-1)
            # feature = feature1 + feature0
            feature = self.mlp_no(feature, d_out[i], 'mlp_f01' + str(i), is_training)

            feature = feature + feature_shout
            feature = tf.nn.leaky_relu(feature)
            # feature = tf.layers.dense(feature, d_out[i], activation=None, name='fcen'+str(i))
            # feature = self.GLUE(tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training))

            f_encode.append(feature)

            if i < layer - 1:

                feature = self.random_sample_or(feature, inputs['sub_idx'][i])

                col = tf.concat([inputs['xyz'][i + 1] / 50, inputs['col'][i + 1]], axis=-1)
                col = tf.layers.dense(col, d_out[i], activation=None, name='fclayer_i' + str(i))
                col = self.GLUE(tf.layers.batch_normalization(col, -1, 0.99, 1e-6, training=is_training))
                # feature = feature + col
                feature = tf.concat([col, feature], axis=-1)

                feature = self.mlp(feature, d_out[i+1], 'mlp_f02' + str(i), is_training)
                # feature = tf.layers.dense(feature, d_out[i + 1], activation=None, name='fclayer' + str(i))
                # feature = self.GLUE(tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training))


        for i in range(layer - 1):

            feature = self.nearest_interpolation_new(feature, inputs['interp_idx'][layer - 2 - i])
            feature = self.random_sample_or(feature, inputs['neigh_idx'][layer - 2 - i])
            # feature = tf.reduce_max(feature, axis=2)
            feature = tf.concat([feature, f_encode[layer - 2 - i]], axis=-1)
            # feature = tf.layers.dense(feature, d_out[layer - 2 - i], activation=None, name='fcde' + str(i))
            # feature = self.GLUE(tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training))
            feature = self.mlp_no(feature, d_out[layer - 2 - i], 'def01' + str(i), is_training)

            feature = feature + f_encode[layer - 2 - i]
            feature = tf.nn.leaky_relu(feature)

            if i > layer - 5:
                f_decode.append(feature)


        # feature = tf.layers.dense(feature, 32, activation=None, name='fc20')
        # feature = self.GLUE(tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training))
        feature = self.mlp(feature, 64, 'class0', is_training)
        feature = self.mlp(feature, 32, 'class1', is_training)
        # feature = tf_util.dropout(feature, keep_prob=0.5, is_training=is_training, scope='dp1')
        feature = tf.layers.dense(feature, self.config.num_classes, activation=None, name='fc30')

        return feature



    def cnn_layer_2_x(self, feature, d_out, kernels_size, name, stide, padding, bn, is_training, neigh_idx):

        shoutcut = tf.expand_dims(feature, [2])
        shoutcut = tf_util.conv2d(shoutcut, d_out // 2, [1, 1], name + 'shoutcut0', [1, 1], 'VALID', True, is_training)
        shoutcut = tf.squeeze(shoutcut, [2])

        feature = tf.expand_dims(feature, [2])
        feature = tf_util.conv2d(feature, d_out // 2, [1, 1], name + 'mpl0', stide, padding, bn, is_training)

        feature = tf_util.conv2d(feature, d_out // 2, [1, 1], name + 'mpl1', stide, padding, bn, is_training)
        feature = tf.squeeze(feature, [2])

        feature = self.gather_neighbour(feature, neigh_idx)
        feature = tf_util.conv2d(feature, d_out // 2, [1, self.config.k_n], name + 'mpl2', stide, padding, bn, is_training)
        feature = tf.squeeze(feature, [2])

        feature = tf.concat([feature, shoutcut], axis=-1)
        feature = self.gather_neighbour(feature, neigh_idx)
        feature = tf_util.conv2d(feature, d_out, kernels_size, name + 'mpl3', stide, padding, bn, is_training)
        feature = tf.squeeze(feature, [2])

        return feature

    def train(self, dataset):
        log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)
        self.sess.run(dataset.train_init_op)
        acc_list = []
        l_out_list = []
        time_list = []
        num_list = []

        while self.training_epoch < self.config.max_epoch:
            t_start = time.time()
            try:
                ops = [self.train_op,
                       self.extra_update_ops,
                       self.merged,
                       self.loss,
                       self.logits,
                       self.labels,
                       self.accuracy,
                       self.inputs['input_inds'],
                       self.inputs['cloud_inds'],
                       self.prob_logits]

                _, _, summary, l_out, probs, labels, acc, point_idx, cloud_idx, prob_logits = self.sess.run(ops, {self.is_training: True})
                self.train_writer.add_summary(summary, self.training_step)
                t_end = time.time()

                # batch = np.shape(point_idx)[0]
                # prob_logits = np.array(prob_logits).reshape([batch, -1, self.config.num_classes])
                # for j in range(batch):
                #     p_idx = point_idx[j, :]
                #     c_i = cloud_idx[j][0]
                #     prob_logits[j, :, :] = prob_logits[j, :, :]
                #     # weight = np.max(prob_logits[j, :, :], axis=-1).reshape([-1, 1])
                #     # prob_logits[j, :, :] = prob_logits[j, :, :] * (1-weight) + dataset.input_logits['training'][c_i][p_idx] * weight
                #     dataset.input_logits['training'][c_i][p_idx] = prob_logits[j, :, :]
                #
                # prob_logits = np.argmax(prob_logits.reshape([-1, self.config.num_classes]), axis=-1)
                # acc = sum(prob_logits == labels) / len(prob_logits)

                acc_list.append(acc)
                l_out_list.append(l_out)
                time_list.append(t_end - t_start)



                if self.training_step % 100 == 0:
                    message = 'Step {:08d} L_out={:5.3f} Acc={:4.2f} ''---{:8.2f} ms/batch'
                    # log_out(message.format(self.training_step, l_out, acc, 1000 * (t_end - t_start)), self.Log_file)
                    acc_mean = np.mean(acc_list)
                    l_out_mean = np.mean(l_out_list)
                    time_mean = np.mean(time_list)
                    log_out(message.format(self.training_step, l_out_mean, acc_mean * 100, 1000 * time_mean), self.Log_file)
                    acc_list = []
                    l_out_list = []
                    time_list = []
                self.training_step += 1

            except tf.errors.OutOfRangeError:
                if dataset.use_val and self.training_epoch % 2 == 0:
                    m_iou = self.evaluate(dataset)
                    if m_iou > np.max(self.mIou_list):
                        # Save the best model
                        snapshot_directory = join(self.saving_path, 'snapshots')
                        makedirs(snapshot_directory) if not exists(snapshot_directory) else None
                        self.saver.save(self.sess, snapshot_directory + '/snap', global_step=self.training_step)
                    self.mIou_list.append(m_iou)
                    log_out('Best m_IoU of {} is: {:5.3f}'.format(dataset.name, max(self.mIou_list)), self.Log_file)
                else:
                    snapshot_directory = join(self.saving_path, 'snapshots')
                    makedirs(snapshot_directory) if not exists(snapshot_directory) else None
                    self.saver.save(self.sess, snapshot_directory + '/snap', self.training_step)

                self.training_epoch += 1
                self.sess.run(dataset.train_init_op)
                # Update learning rate
                op = self.learning_rate.assign(tf.multiply(self.learning_rate,
                                                           self.config.lr_decays[self.training_epoch]))
                self.sess.run(op)
                acc_list = []
                l_out_list = []
                time_list = []
                log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)

            except tf.errors.InvalidArgumentError as e:

                print('Caught a NaN error :')
                print(e.error_code)
                print(e.message)
                print(e.op)
                print(e.op.name)
                print([t.name for t in e.op.inputs])
                print([t.name for t in e.op.outputs])

                a = 1 / 0

        print('finished')
        self.sess.close()

    def evaluate(self, dataset):

        # Initialise iterator with validation data
        self.sess.run(dataset.val_init_op)

        gt_classes = [0 for _ in range(self.config.num_classes)]
        positive_classes = [0 for _ in range(self.config.num_classes)]
        true_positive_classes = [0 for _ in range(self.config.num_classes)]
        conf = np.zeros([self.config.num_classes, self.config.num_classes])
        val_total_correct = 0
        val_total_seen = 0
        time_list = []

        for step_id in range(self.config.val_steps):
            t_start = time.time()
            if step_id % 100 == 0:
                print(str(step_id) + ' / ' + str(self.config.val_steps))
            try:
                ops = (self.prob_logits, self.labels, self.accuracy, self.inputs['input_inds'], self.inputs['cloud_inds'])
                stacked_prob, labels, acc, point_idx, cloud_idx = self.sess.run(ops, {self.is_training: False})
                # print(stacked_prob.shape, labels.shape, 'ssssssssssssssssssssssss')
                t_end = time.time()
                time_list.append(t_end - t_start)

                # batch = np.shape(point_idx)[0]
                # prob_logits = np.array(stacked_prob).reshape([batch, -1, self.config.num_classes])
                # for j in range(batch):
                #     p_idx = point_idx[j, :]
                #     c_i = cloud_idx[j][0]
                #     prob_logits[j, :, :] = prob_logits[j, :, :]
                #     # weight = np.max(prob_logits[j, :, :], axis=-1).reshape([-1, 1])
                #     # prob_logits[j, :, :] = prob_logits[j, :, :] * (1 - weight) + dataset.input_logits['validation'][c_i][p_idx] * weight
                #     dataset.input_logits['validation'][c_i][p_idx] = prob_logits[j, :, :]
                # stacked_prob = prob_logits.reshape([-1, self.config.num_classes])


                labels = np.array(labels)
                pred = np.argmax(stacked_prob, 1)

                self.config.ignored_label_inds=[]
                if not self.config.ignored_label_inds:
                    pred_valid = pred
                    labels_valid = labels
                else:
                    invalid_idx = np.where(labels == self.config.ignored_label_inds)[0]
                    labels_valid = np.delete(labels, invalid_idx)
                    labels_valid = labels_valid - 1
                    pred_valid = np.delete(pred, invalid_idx)

                correct = np.sum(pred_valid == labels_valid)
                val_total_correct += correct
                val_total_seen += len(labels_valid)

                conf_matrix = confusion_matrix(labels_valid, pred_valid, np.arange(0, self.config.num_classes, 1))
                conf += np.array(conf_matrix)
                gt_classes += np.sum(conf_matrix, axis=1)
                positive_classes += np.sum(conf_matrix, axis=0)
                true_positive_classes += np.diagonal(conf_matrix)

            except tf.errors.OutOfRangeError:
                break

        iou_list = []
        print('-------time--------', int(np.mean(time_list) * 1000))
        for n in range(0, self.config.num_classes, 1):
            iou = true_positive_classes[n] / float(gt_classes[n] + positive_classes[n] - true_positive_classes[n] + 0.1)
            iou_list.append(iou)
        mean_iou = sum(iou_list) / float(self.config.num_classes)

        log_out('eval accuracy: {}'.format(val_total_correct / float(val_total_seen)), self.Log_file)
        log_out('mean IOU:{}'.format(mean_iou), self.Log_file)

        mean_iou = 100 * mean_iou
        log_out('Mean IoU = {:.1f}%'.format(mean_iou), self.Log_file)
        s = '{:5.2f} | '.format(mean_iou)
        for IoU in iou_list:
            s += '{:5.2f} '.format(100 * IoU)
        log_out('-' * len(s), self.Log_file)
        log_out(s, self.Log_file)
        log_out('-' * len(s) + '\n', self.Log_file)

        conf = np.array(conf) / 1000
        for i in range(conf.shape[0]):
            s = '{:5.2f}    '.format(conf[i, 0])
            for j in range(conf.shape[1] - 1):
                s += '{:5.2f}    '.format(conf[i, j + 1])
            log_out(s, self.Log_file)
        return mean_iou

    def get_loss_org(self, logits, labels, pre_cal_weights):
        # calculate the weighted cross entropy according to the inverse frequency
        class_weights = tf.convert_to_tensor(pre_cal_weights, dtype=tf.float32)
        one_hot_labels = tf.one_hot(labels, depth=self.config.num_classes)
        weights = tf.reduce_sum(class_weights * one_hot_labels, axis=1)
        # weights = 1
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=one_hot_labels)

        # weights1 = tf.nn.softmax(logits)
        # weights1 = tf.reduce_sum(weights1 * one_hot_labels, axis=1) - tf.reduce_max(weights1, axis=-1)
        # weights1 = tf.abs(weights1)

        # weights1 = tf.reduce_sum(tf.abs(weights1), axis=1)
        # weights1 = tf.exp(tf.reduce_sum(tf.abs(weights1), axis=1)) - 1
        # logits = tf.nn.softmax(logits, axis=-1)
        # weights1 = tf.reduce_sum(tf.abs(one_hot_labels - logits), axis=-1)
        # unweighted_l1 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=l1, labels=one_hot_labels)
        # unweighted_l2 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=l2, labels=one_hot_labels)
        # unweighted_l3 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=l3, labels=one_hot_labels)
        # square_loss = tf.reduce_sum(tf.square(logits - one_hot_labels) / 2, axis=1)

        # weighted_losses = (unweighted_losses ) * weights
        # weighted_losses = unweighted_losses  * weights
        # output_loss = tf.reduce_mean((unweighted_losses + unweighted_l1 * 0.6 + unweighted_l2 * 0.4) * weights)
        # output_loss = tf.reduce_mean(unweighted_losses * weights + weights1 * weights)
        output_loss = tf.reduce_mean(unweighted_losses * weights)

        return output_loss

    def get_loss_org0(self, logits, labels, pre_cal_weights, l1, l2, labels1, labels2):
        # calculate the weighted cross entropy according to the inverse frequency
        class_weights = tf.convert_to_tensor(pre_cal_weights, dtype=tf.float32)
        one_hot_labels = tf.one_hot(labels, depth=self.config.num_classes)
        weights = tf.reduce_sum(class_weights * one_hot_labels, axis=1)
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=one_hot_labels)

        one_hot_labels1 = tf.one_hot(labels1, depth=self.config.num_classes)
        unweighted_losses_xyz = tf.nn.softmax_cross_entropy_with_logits_v2(logits=l1, labels=one_hot_labels1)
        weights1 = tf.reduce_sum(class_weights * one_hot_labels1, axis=1)

        one_hot_labels2 = tf.one_hot(labels2, depth=self.config.num_classes)
        unweighted_losses_col = tf.nn.softmax_cross_entropy_with_logits_v2(logits=l2, labels=one_hot_labels2)
        weights2 = tf.reduce_sum(class_weights * one_hot_labels2, axis=1)

        # weighted_losses = (unweighted_losses ) * weights
        weighted_losses = unweighted_losses * weights
        weighted_losses1 = unweighted_losses_xyz * weights1
        weighted_losses2 = unweighted_losses_col * weights2
        output_loss = tf.reduce_mean(weighted_losses) + tf.reduce_mean(weighted_losses1) + tf.reduce_mean(weighted_losses2)

        return output_loss

    def topkloss(self, weighted_losses, weights):
        value, idx = tf.nn.top_k(weighted_losses, int(65536 * 0.2))
        weighted_losses = tf.gather(weighted_losses, idx)
        weights = tf.gather(weights, idx)
        weighted_losses = weighted_losses * weights
        weighted_losses = tf.reduce_mean(weighted_losses)
        return weighted_losses

    def get_loss_org3(self, logits, labels, pre_cal_weights, confi, confi1, confi_ture, c0, c1):


        class_weights = tf.convert_to_tensor(pre_cal_weights, dtype=tf.float32)
        one_hot_labels = tf.one_hot(labels, depth=self.config.num_classes)
        weights = tf.reduce_sum(class_weights * one_hot_labels, axis=1)
        # unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=one_hot_labels)
        unweighted_losses = tf.reduce_sum(tf.square(logits - one_hot_labels), axis=1)
        weighted_losses0 = unweighted_losses * weights
        # output_loss = tf.reduce_mean(weighted_losses0)

        confi = tf.sigmoid(confi)
        confi1 = tf.sigmoid(confi1)

        ones = tf.zeros_like(confi, dtype=tf.float32) + 1
        zero = tf.zeros_like(confi, dtype=tf.float32)
        confi_loss = tf.where(confi >= 0.5, ones, zero)
        confi_loss1 = tf.where(confi1 >= 0.5, ones, zero)

        confi_loss_and = tf.where(confi_loss == confi_ture, ones, zero)
        confi_loss_and1 = tf.where(confi_loss1 == confi_ture, ones, zero)

        confi_loss_or = tf.where((confi_loss + confi_ture) > 0, ones, zero)
        confi_loss_or1 = tf.where((confi_loss1 + confi_ture) > 0, ones, zero)

        confi_loss = tf.reduce_sum(confi_loss_and, axis=1) / tf.reduce_sum(confi_loss_or, axis=1)
        confi_loss1 = tf.reduce_sum(confi_loss_and1, axis=1) / tf.reduce_sum(confi_loss_or1, axis=1)

        c_loss = tf.square(confi_loss - c0) * weights
        c_loss1 = tf.square(confi_loss1 - c1) * weights

        loss = tf.reduce_sum(tf.square(confi - confi_ture), axis=1) * weights
        loss1 = tf.reduce_sum(tf.square(confi1 - confi_ture), axis=1) * weights

        output_loss = tf.reduce_mean(weighted_losses0) + tf.reduce_mean(loss) + tf.reduce_mean(loss1) + tf.reduce_mean(c_loss) + tf.reduce_mean(c_loss1)

        return output_loss

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)
        xyz_tile = tf.tile(tf.expand_dims(xyz, axis=2), [1, 1, tf.shape(neigh_idx)[-1], 1])
        relative_xyz = xyz_tile - neighbor_xyz
        relative_dis = tf.sqrt(tf.reduce_sum(tf.square(relative_xyz), axis=-1, keepdims=True))
        relative_feature = tf.concat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], axis=-1)
        return relative_feature

    def relative_pos_encoding1(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)
        xyz_tile = tf.tile(tf.expand_dims(xyz, axis=2), [1, 1, tf.shape(neigh_idx)[-1], 1])
        relative_xyz = xyz_tile - neighbor_xyz
        relative_feature = relative_xyz
        return relative_feature

    def relative_pos_encoding_or(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)
        xyz_tile = tf.tile(tf.expand_dims(xyz, axis=2), [1, 1, tf.shape(neigh_idx)[-1], 1])
        relative_xyz = xyz_tile - neighbor_xyz



        zero = tf.zeros_like(relative_xyz, dtype=tf.float32)
        ones = zero + 1
        relative_xyz_pros = tf.where(relative_xyz < 0, ones, zero)
        relative_xyz_pros = tf.abs(relative_xyz * relative_xyz_pros)
        relative_xyz = tf.concat([relative_xyz, relative_xyz_pros], axis=-1)


        relative_dis = tf.sqrt(tf.reduce_sum(tf.square(relative_xyz), axis=-1, keepdims=True))
        # relative_feature = tf.concat([relative_xyz, relative_dis], axis=-1)
        relative_feature = tf.concat([relative_xyz, relative_dis], axis=-1)
        # relative_feature = tf.concat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], axis=-1)

        # n = tf.shape(relative_feature)[1]
        # relative_feature = relative_feature - tf.tile(tf.reduce_min(relative_feature, axis=1, keepdims=True), [1, n, 1, 1])
        # relative_feature = relative_feature / tf.tile(tf.reduce_max(relative_feature, axis=1, keepdims=True), [1, n, 1, 1])


        # relative_feature = tf.concat([relative_dis, relative_xyz, neighbor_xyz], axis=-1)
        # relative_feature = tf.concat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], axis=-1)
        return relative_feature

    def relative_pos_encoding_new(self, xyz, neigh_idx):

        batch = tf.shape(xyz)[0]

        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)
        xyz_tile = tf.tile(tf.expand_dims(xyz, axis=2), [1, 1, tf.shape(neigh_idx)[-1], 1])
        relative_xyz = xyz_tile - neighbor_xyz


        relative_dis = tf.sqrt(tf.reduce_sum(tf.square(relative_xyz), axis=-1, keepdims=True))

        relative_feature = tf.concat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], axis=-1)
        relative_feature = tf.reshape(relative_feature, [batch, -1, self.config.k_n * 10])

        return relative_feature

    def relative_col_encoding_or(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)
        xyz_tile = tf.tile(tf.expand_dims(xyz, axis=2), [1, 1, tf.shape(neigh_idx)[-1], 1])
        relative_xyz = xyz_tile - neighbor_xyz

        # relative_feature = relative_xyz
        relative_feature = tf.concat([neighbor_xyz, relative_xyz], axis=-1)
        # relative_feature = tf.concat([relative_xyz, xyz_tile,neighbor_xyz], axis=-1)

        # n = tf.shape(relative_feature)[1]
        # relative_feature = relative_feature - tf.tile(tf.reduce_min(relative_feature, axis=1, keepdims=True),
        #                                               [1, n, 1, 1])
        # relative_feature = relative_feature / tf.tile(tf.reduce_max(relative_feature, axis=1, keepdims=True),
        #                                               [1, n, 1, 1])

        return relative_feature

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        # feature = tf.squeeze(feature, axis=2)
        num_neigh = tf.shape(pool_idx)[-1]
        d = feature.get_shape()[-1]
        batch_size = tf.shape(pool_idx)[0]
        pool_idx = tf.reshape(pool_idx, [batch_size, -1])
        pool_features = tf.batch_gather(feature, pool_idx)
        pool_features = tf.reshape(pool_features, [batch_size, -1, num_neigh, d])
        pool_features = tf.reduce_max(pool_features, axis=2)
        # pool_features = tf.nn.leaky_relu(tf.layers.batch_normalization(pool_features, -1, 0.99, 1e-6, training=is_training))
        return pool_features

    @staticmethod
    def random_sample_new(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        # feature = tf.squeeze(feature, axis=2)
        num_neigh = tf.shape(pool_idx)[-1]
        d = feature.get_shape()[-1]
        batch_size = tf.shape(pool_idx)[0]
        pool_idx = tf.reshape(pool_idx, [batch_size, -1])
        pool_features = tf.batch_gather(feature, pool_idx)
        pool_features = tf.reshape(pool_features, [batch_size, -1, num_neigh, d])
        # pool_features = pool_features[-1]
        return pool_features

    @staticmethod
    def random_sample_or(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        # feature = tf.squeeze(feature, axis=2)
        num_neigh = tf.shape(pool_idx)[-1]
        d = feature.get_shape()[-1]
        batch_size = tf.shape(pool_idx)[0]
        pool_idx = tf.reshape(pool_idx, [batch_size, -1])
        pool_features = tf.batch_gather(feature, pool_idx)
        pool_features = tf.reshape(pool_features, [batch_size, -1, num_neigh, d])
        pool_features = tf.reduce_max(pool_features, axis=2, keepdims=True)
        pool_features = tf.squeeze(pool_features, [2])
        # pool_features = tf.nn.leaky_relu(tf.layers.batch_normalization(pool_features, -1, 0.99, 1e-6, training=is_training))
        return pool_features

    @staticmethod
    def random_gather(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        # feature = tf.squeeze(feature, axis=2)
        num_neigh = tf.shape(pool_idx)[-1]
        d = feature.get_shape()[-1]
        batch_size = tf.shape(pool_idx)[0]
        pool_idx = tf.reshape(pool_idx, [batch_size, -1])
        pool_features = tf.batch_gather(feature, pool_idx)
        pool_features = tf.reshape(pool_features, [batch_size, -1, num_neigh, d])
        # pool_features = tf.nn.leaky_relu(tf.layers.batch_normalization(pool_features, -1, 0.99, 1e-6, training=is_training))
        return pool_features

    @staticmethod
    def random_sample_new_(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        pool_idx = tf.expand_dims(pool_idx, axis=2)

        d = feature.get_shape()[-1]
        batch_size = tf.shape(pool_idx)[0]
        pool_idx = tf.reshape(pool_idx, [batch_size, -1])
        pool_features = tf.batch_gather(feature, pool_idx)
        pool_features = tf.reshape(pool_features, [batch_size, -1, d])

        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = tf.squeeze(feature, axis=2)
        batch_size = tf.shape(interp_idx)[0]
        up_num_points = tf.shape(interp_idx)[1]
        interp_idx = tf.reshape(interp_idx, [batch_size, up_num_points])
        interpolated_features = tf.batch_gather(feature, interp_idx)
        interpolated_features = tf.expand_dims(interpolated_features, axis=2)
        return interpolated_features

    @staticmethod
    def nearest_interpolation_new(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        # feature = tf.squeeze(feature, axis=2)
        batch_size = tf.shape(interp_idx)[0]
        up_num_points = tf.shape(interp_idx)[1]
        interp_idx = tf.reshape(interp_idx, [batch_size, up_num_points])
        interpolated_features = tf.batch_gather(feature, interp_idx)
        # interpolated_features = tf.expand_dims(interpolated_features, axis=2)
        return interpolated_features



    @staticmethod
    def gather_neighbour(pc, neighbor_idx):
        # gather the coordinates or features of neighboring points
        batch_size = tf.shape(pc)[0]
        num_points = tf.shape(pc)[1]
        d = pc.get_shape()[2].value
        index_input = tf.reshape(neighbor_idx, shape=[batch_size, -1])
        features = tf.batch_gather(pc, index_input)
        features = tf.reshape(features, [batch_size, num_points, -1, d])

        return features

    @staticmethod
    def gather_neighbour_nei(pc, neighbor_idx):
        # gather the coordinates or features of neighboring points
        batch_size = tf.shape(pc)[0]
        num_points = tf.shape(pc)[1]
        # d=dim
        d = pc.get_shape()[2].value
        # d = tf.shape(pc)[2]

        index_input = tf.reshape(neighbor_idx, shape=[batch_size, -1])
        features = tf.batch_gather(pc, index_input)
        features = tf.reshape(features, [batch_size, -1, 25, d])
        return features

    @staticmethod
    def att_pooling_2(feature_set, d_out, name, is_training):
        batch_size = tf.shape(feature_set)[0]
        num_points = tf.shape(feature_set)[1]
        num_neigh = tf.shape(feature_set)[2]
        d = feature_set.get_shape()[3].value
        value = tf.reshape(feature_set, shape=[-1, num_neigh, d])

        query = tf.layers.dense(value, d, activation=None, use_bias=False, name=name + 'fc0')
        key = tf.layers.dense(value, d, activation=None, use_bias=False, name=name + 'fc1')
        value = tf.layers.dense(value, d, activation=None, use_bias=False, name=name + 'fc2')

        query = query * key
        query = tf.nn.softmax(query)

        value = query * value
        value = tf.reduce_sum(value, axis=1)

        value = tf.reshape(value, [batch_size, num_points, 1, d])
        value = tf_util.conv2d(value, d_out, [1, 1], name + 'mlp', [1, 1], 'VALID', True, is_training)
        return value

    @staticmethod
    def att_pooling_1(feature_set, d_out, name, is_training):
        batch_size = tf.shape(feature_set)[0]
        num_points = tf.shape(feature_set)[1]
        num_neigh = tf.shape(feature_set)[2]
        d = feature_set.get_shape()[3].value
        f_reshaped = tf.reshape(feature_set, shape=[-1, num_neigh, d])
        att_activation = tf.layers.dense(f_reshaped, d, activation=None, use_bias=False, name=name + 'fc')
        att_scores = tf.nn.softmax(att_activation, axis=1)
        f_agg = f_reshaped * att_scores

        f_agg = tf.reshape(f_agg, [batch_size, num_points, -1, d])
        f_agg = tf_util.conv2d(f_agg, d, [1, 9], name + 'mlp0', [1, 1], 'VALID', True, is_training)
        f_agg = tf.squeeze(f_agg, [2])

        f_agg = tf.reshape(f_agg, [batch_size, num_points, 1, d])
        f_agg = tf_util.conv2d(f_agg, d_out, [1, 1], name + 'mlp', [1, 1], 'VALID', True, is_training)
        return f_agg

    @staticmethod
    def att_pooling(feature_set, d_out, name, is_training):
        batch_size = tf.shape(feature_set)[0]
        num_points = tf.shape(feature_set)[1]
        num_neigh = tf.shape(feature_set)[2]
        d = feature_set.get_shape()[3].value
        f_reshaped = tf.reshape(feature_set, shape=[-1, num_neigh, d])
        att_activation = tf.layers.dense(f_reshaped, d, activation=None, use_bias=False, name=name + 'fc')
        att_scores = tf.nn.softmax(att_activation, axis=1)
        f_agg = f_reshaped * att_scores
        f_agg = tf.reduce_sum(f_agg, axis=1)
        f_agg = tf.reshape(f_agg, [batch_size, num_points, 1, d])
        f_agg = tf_util.conv2d(f_agg, d_out, [1, 1], name + 'mlp', [1, 1], 'VALID', True, is_training)
        return f_agg

    @staticmethod
    def att_pooling_(feature_set, d_out, name, is_training):

        f_reshaped = tf_util.conv2d(feature_set, d_out, [1, 1], name + 'mlp0', [1, 1], 'VALID', True, is_training)
        att_scores = tf.nn.softmax(f_reshaped, axis=2)

        f_agg = f_reshaped * att_scores
        f_agg = tf.reduce_sum(f_agg, axis=2)
        f_agg = tf.expand_dims(f_agg, axis=2)
        f_agg = tf_util.conv2d(f_agg, d_out, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)
        # f_agg = tf.squeeze(f_agg, axis=2)
        return f_agg
