from os import makedirs, system
from os.path import exists, join, dirname, abspath
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import time
from sklearn.metrics import confusion_matrix
import pandas as pd


def log_out(out_str, log_f_out):
    log_f_out.write(out_str + '\n')
    log_f_out.flush()
    print(out_str)


class ModelTester:
    def __init__(self, model, dataset, restore_snap=None):
        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)
        self.Log_file = open('log_test_' + dataset.name + '.txt', 'a')

        # Create a session for running Ops on the Graph.
        on_cpu = False
        if on_cpu:
            c_proto = tf.ConfigProto(device_count={'GPU': 0})
        else:
            c_proto = tf.ConfigProto()
            c_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=c_proto)
        self.sess.run(tf.global_variables_initializer())

        # Load trained model
        if restore_snap is not None:
            self.saver.restore(self.sess, restore_snap)
            print("Model restored from " + restore_snap)

        self.prob_logits = tf.nn.softmax(model.logits)

        # Initiate global prediction over all test clouds
        self.test_probs = [np.zeros(shape=[l.shape[0], model.config.num_classes], dtype=np.float32)
                           for l in dataset.input_labels['test']]

        self.test_labels = [np.zeros(shape=[l.shape[0]], dtype=np.int)
                           for l in dataset.input_labels['test']]

    def test(self, model, dataset, num_votes=100):

        # Smoothing parameter for votes
        test_smooth = 0.95

        # Initialise iterator with validation/test data
        self.sess.run(dataset.test_init_op)

        # Test saving path
        saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
        test_path = join('test', saving_path.split('/')[-1])
        makedirs(test_path) if not exists(test_path) else None
        makedirs(join(test_path, 'test_preds')) if not exists(join(test_path, 'test_preds')) else None

        step_id = 0
        epoch_id = 0
        last_min = -0.5

        while last_min < num_votes:
            try:
                ops = (self.prob_logits,
                       model.labels,
                       model.inputs['input_inds'],
                       model.inputs['cloud_inds'])

                stacked_probs, stacked_labels, point_idx, cloud_idx = self.sess.run(ops, {model.is_training: False})
                stacked_probs = np.reshape(stacked_probs, [model.config.val_batch_size, model.config.num_points,
                                                           model.config.num_classes])
                stacked_labels = np.reshape(stacked_labels,
                                            [model.config.val_batch_size, model.config.num_points])

                for j in range(np.shape(stacked_probs)[0]):
                    probs = stacked_probs[j, :, :]
                    label = stacked_labels[j, :]
                    p_idx = point_idx[j, :]
                    c_i = cloud_idx[j][0]
                    self.test_probs[c_i][p_idx] = self.test_probs[c_i][p_idx] + probs
                    self.test_labels[c_i][p_idx] = label
                step_id += 1

            except tf.errors.OutOfRangeError:

                new_min = np.min(dataset.min_possibility['test'])
                log_out('Epoch {:3d}, end. Min possibility = {:.1f}'.format(epoch_id, new_min), self.Log_file)

                if last_min + 0.1 < new_min:

                    # Update last_min
                    last_min += 1

                    # Show vote results (On subcloud so it is not the good values here)
                    log_out('\nConfusion on sub clouds', self.Log_file)
                    num_test = len(dataset.input_labels['test'])

                    # Project predictions
                    log_out('\nReproject Vote #{:d}'.format(int(np.floor(new_min))), self.Log_file)
                    proj_probs_list = []
                    proj_probs_label = []
                    for i_test in range(num_test):
                        # Reproject probs back to the evaluations points

                        probs = self.test_probs[i_test]
                        proj_probs_list += [probs]
                        proj_probs_label += [self.test_labels[i_test]]

                    gt_classes = [0 for _ in range(13)]
                    positive_classes = [0 for _ in range(13)]
                    true_positive_classes = [0 for _ in range(13)]
                    conf = np.zeros([13, 13])
                    val_total_correct = 0
                    val_total_seen = 0

                    # Show vote results
                    log_out('Confusion on full clouds', self.Log_file)
                    for i_test in range(num_test):
                        # Get the predicted labels
                        # preds = dataset.label_values[np.argmax(proj_probs_list[i_test], axis=1)].astype(np.uint8)
                        t_l = proj_probs_label[i_test]
                        pred_valid = np.argmax(proj_probs_list[i_test], axis=1)
                        conf_matrix = confusion_matrix(t_l, pred_valid, np.arange(0, 13, 1))
                        conf += np.array(conf_matrix)
                        gt_classes += np.sum(conf_matrix, axis=1)
                        positive_classes += np.sum(conf_matrix, axis=0)
                        true_positive_classes += np.diagonal(conf_matrix)

                        correct = np.sum(pred_valid == t_l)
                        val_total_correct += correct
                        val_total_seen += len(t_l)

                    iou_list = []
                    for n in range(0, 13, 1):
                        iou = true_positive_classes[n] / float(
                            gt_classes[n] + positive_classes[n] - true_positive_classes[n] + 0.1)
                        iou_list.append(iou)
                    mean_iou = sum(iou_list) / float(13)
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

                    return

                self.sess.run(dataset.test_init_op)
                epoch_id += 1
                step_id = 0
                continue
