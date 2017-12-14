import tensorflow as tf
import os
import json
import subprocess
from scipy.misc import imread, imresize
from scipy import misc
import time

from train import build_forward
from utils.annolist import AnnotationLib as al
from utils.train_utils import add_rectangles, rescale_boxes

import cv2
import argparse

def get_image_dir(args):
    weights_iteration = int(args.weights.split('-')[-1])
    expname = '_' + args.expname if args.expname else ''
    image_dir = '%s/images_%s_%d%s' % (os.path.dirname(args.weights), os.path.basename(args.test_boxes)[:-5], weights_iteration, expname)
    return image_dir

def get_results(args, H):

    tf.reset_default_graph()
    H["grid_width"] = H["image_width"] / H["region_size"]
    H["grid_height"] = H["image_height"] / H["region_size"]
    x_in = tf.placeholder(tf.float32, name='x_in', shape=[H['image_height'], H['image_width'], 3])
    if H['use_rezoom']:
        pred_boxes, pred_logits, pred_confidences, pred_confs_deltas, pred_boxes_deltas = build_forward(H, tf.expand_dims(x_in, 0), 'test', reuse=None)
        grid_area = H['grid_height'] * H['grid_width']
        pred_confidences = tf.reshape(tf.nn.softmax(tf.reshape(pred_confs_deltas, [grid_area * H['rnn_len'], 2])), [grid_area, H['rnn_len'], 2])
        if H['reregress']:
            pred_boxes = pred_boxes + pred_boxes_deltas
    else:
        pred_boxes, pred_logits, pred_confidences = build_forward(H, tf.expand_dims(x_in, 0), 'test', reuse=None)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, args.weights)

        orig_img = imread('data/tshirtslayer/tss-images/wearing.jpg')
        img = imresize(orig_img, (H["image_height"], H["image_width"]), interp='cubic')
        feed = {x_in: img}
        now = time.time()

        (np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes, pred_confidences], feed_dict=feed)

        print (time.time()-now)

        pred_anno = al.Annotation()
        pred_anno.imageName = "image.jpg"

        new_img, rects = add_rectangles(H, [img], np_pred_confidences, np_pred_boxes,
                                        use_stitching=True, rnn_len=H['rnn_len'], min_conf=args.min_conf, tau=args.tau, show_suppressed=args.show_suppressed)


        misc.imsave("data/xxx.jpg", new_img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--expname', default='')
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--logdir', default='output')
    parser.add_argument('--iou_threshold', default=0.9, type=float)
    parser.add_argument('--tau', default=0.25, type=float)
    parser.add_argument('--min_conf', default=0.2, type=float)
    parser.add_argument('--show_suppressed', default=True, type=bool)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    hypes_file = '%s/hypes.json' % os.path.dirname(args.weights)
    with open(hypes_file, 'r') as f:
        H = json.load(f)
    expname = args.expname + '_' if args.expname else ''


    get_results(args, H)

    try:
        rpc_cmd = './utils/annolist/doRPC.py --minOverlap %f ' % (args.iou_threshold)
        print('$ %s' % rpc_cmd)
        rpc_output = subprocess.check_output(rpc_cmd, shell=True)
        print(rpc_output)
        txt_file = [line for line in rpc_output.split('\n') if line.strip()][-1]
        output_png = '%s/results.png' % get_image_dir(args)
        plot_cmd = './utils/annolist/plotSimple.py %s --output %s' % (txt_file, output_png)
        print('$ %s' % plot_cmd)
        plot_output = subprocess.check_output(plot_cmd, shell=True)
        print('output results at: %s' % plot_output)
    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()