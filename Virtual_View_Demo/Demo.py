from __future__ import absolute_import, division, print_function
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

import tensorflow as tf
import tensorflow.contrib.slim as slim
from model import *
from dataloader import *
import cv2
import argparse

parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')

parser.add_argument('--mode',                      type=str,   help='train or test or demo', default='demo')
parser.add_argument('--model_name',                type=str,   help='name of model to be saved', default='basket')
parser.add_argument('--data_path',                 type=str,   help='path to the data', default='D:/mpeg_full/')
parser.add_argument('--input_height',              type=int,   help='input height', default=192)
parser.add_argument('--input_width',               type=int,   help='input width', default=640)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=3)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=5)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--alpha_image_loss',          type=float, help='weight between SSIM and L1 in the image loss', default=0.85)
parser.add_argument('--disp_gradient_loss_weight', type=float, help='disparity smoothness weigth', default=0.1)
parser.add_argument('--do_stereo',                             help='if set, will train the stereo model', action='store_true')
parser.add_argument('--num_gpus',                  type=int,   help='number of GPUs to use for training', default=1)
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=8)
parser.add_argument('--output_directory',          type=str,   help='output directory for test disparities, if empty outputs to checkpoint folder', default='')
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
parser.add_argument('--events_path',               type=str,   help='path to the events file', default='./events')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a specific checkpoint to load', default='./model/weight')

parser.add_argument('--synthesis_height',          type=int,   help='synthesis height', default=768)
parser.add_argument('--synthesis_width',           type=int,   help='synthesis width', default=1024)
parser.add_argument('--demo_path',                 type=str,   help='path to the data', default='./example/')
parser.add_argument('--demo_tum',           type=int,   help='tum', default=5)

args = parser.parse_args()

def demo(params, dataloader):
    checkpoint_path = args.checkpoint_path
    left = dataloader.left_image_batch
    right = dataloader.right_image_batch
    param_path = dataloader.param_path_batch
    tf_now_scale = tf.placeholder(tf.float32,[1])
    model = MonodepthModel(params,dataloader.mode,None,left,right,param_path,False,0,tf_now_scale)

    config = tf.ConfigProto(allow_soft_placement = True)
    sess = tf.Session(config = config)
    variables = slim.get_variables_to_restore()


    variables_to_restore = [v for v in variables if
                        v.name.split('/')[0] != 'image_sampling' and v.name.split('/')[0] != 'image_sampling_1']
    train_saver = tf.train.Saver(variables_to_restore)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)
    train_saver.restore(sess, checkpoint_path)

    num =20
    scale = 1/(args.demo_tum+1)
    tum_cnt = 0
    now_scale= 1
    frame =0
    flag = False
    while(1):
        if flag is False:
            recon,image = sess.run([model.reconstructed_image,dataloader.left_image_batch], feed_dict={tf_now_scale: [now_scale],
                                                                   dataloader.left_image_path: args.demo_path + 'test/{}/{}.jpg'.format(
                                                                       num, frame),
                                                                   dataloader.right_image_path: args.demo_path + 'test/{}/{}.jpg'.format(
                                                                       num + 1, frame),
                                                                   dataloader.left_param_path: [
                                                                       args.demo_path + 'cam_q/{}/camera.txt'.format(
                                                                           num)],
                                                                   dataloader.right_param_path: [
                                                                       args.demo_path + 'cam_q/{}/camera.txt'.format(
                                                                           num + 1)],
                                                                   })

            flag=True
        elif int(now_scale*10)%10==0:
            if now_scale==2:
                image = sess.run(dataloader.right_image_batch,feed_dict={dataloader.right_image_path:args.demo_path+'test/{}/{}.jpg'.format(num+1,frame)})
            else:
                image = sess.run(dataloader.left_image_batch,feed_dict={dataloader.left_image_path:args.demo_path+'test/{}/{}.jpg'.format(num,frame)})
        else:
            image = sess.run(model.reconstructed_image,feed_dict={tf_now_scale:[now_scale],
                                                                  dataloader.left_image_path:args.demo_path+'test/{}/{}.jpg'.format(num,frame),
                                                                    dataloader.right_image_path:args.demo_path+'test/{}/{}.jpg'.format(num+1,frame),
                                                                    dataloader.left_param_path:[args.demo_path+'cam_q/{}/camera.txt'.format(num)],
                                                                    dataloader.right_param_path:[args.demo_path+'cam_q/{}/camera.txt'.format(num+1)],
                                                                    })
            image=image[0]

        cv_image = image.squeeze().copy()
        cv_image[..., 0] = image[..., 2]
        cv_image[..., 2] = image[..., 0]
        cv2.imshow('', cv_image)

        ## keboard events
        key = cv2.waitKeyEx()
        print(f'num={num}')
        print(f'frame={frame}')
        if key == 2555904 or key==100:  # right_arrow_key
            tum_cnt += 1
            now_scale += scale
            if tum_cnt == args.demo_tum + 1:
                now_scale = 2
            elif tum_cnt > args.demo_tum + 1:
                num += 1
                if num > 39:
                    num = 39
                    now_scale = 2
                    tum_cnt = args.demo_tum + 1
                else:
                    tum_cnt = 1
                    now_scale = 1 + scale
        elif key == 2424832 or key==97:  # left_arrow_key
            tum_cnt -= 1
            now_scale -= scale
            if tum_cnt == 0:
                now_scale = 1
            elif tum_cnt < 0:
                num -= 1
                if num < 20:
                    num = 20
                    now_scale = 1
                    tum_cnt = 0
                else:
                    tum_cnt = args.demo_tum
                    now_scale = 2 - scale
        elif key == 2490368 or key ==119: # up_arrow_key
            if frame == 1798:
                pass
            else:
                frame+=1
        elif key == 2621440 or key ==115: # down_arrow_key
            if frame == 0:
                pass
            else:
                frame-=1
        elif key == 27:  # ESC
            break


def main(_):
    params = monodepth_parameters(
        height=args.input_height,
        width=args.input_width,
        num_threads=args.num_threads,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        do_stereo=args.do_stereo,
        alpha_image_loss=args.alpha_image_loss,
        disp_gradient_loss_weight=args.disp_gradient_loss_weight,
        synthesis_height=args.synthesis_height,
        synthesis_width=args.synthesis_width,
        demo_tum = args.demo_tum
    )

    dataloader = MonodepthDataloader(args.data_path,
                                     params,
                                     args.mode)
    demo(params, dataloader)

if __name__=='__main__':
    tf.app.run()