import time
import numpy as np
import argparse
import os
import cv2
import glob
import shutil

# philly
parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', default='0')
parser.add_argument('-classifier_model', default='')
parser.add_argument('-datatype', default='face')


argsk, argsu = parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = argsk.gpuid

import config
from config import ffhq_path, outputroot
tid = int(time.time())
outputpath = os.path.join(outputroot, '{}_filtered_{}'.format(argsk.datatype, tid))
os.makedirs(outputpath, exist_ok=True)

# filter ffhq dataset with trained classifier
from curation_system import curation_system

classifier = curation_system(len(argsk.gpuid.split(',')), argsk.datatype, classifier_only=True)
classifier.saver_userdis.restore(classifier.sess, argsk.classifier_model)

if argsk.datatype == 'face':
    imagefiles = glob.glob(os.path.join(ffhq_path, '*'))
    cnt = len(imagefiles)
    iters = int(np.ceil(cnt / 32))
    for i in range(iters):
        low = i * 32
        high = min((i+1)*32, cnt)
        images = np.array([cv2.resize(cv2.imread(imagefiles[j]),(160,160)) / 255.0 for j in range(low,high)])
        testresults = classifier.test_by_images(images)
        for j in range(low, high):
            if np.mean(testresults[j-low] > 0.667):
                shutil.copy2(imagefiles[j], os.path.join(outputpath, imagefiles[j].split('/')[-1]))

elif argsk.datatype == 'bedroom':
    passed = 0
    flag = True
    while flag:
        zs = np.random.normal(size=(32,512))
        images = classifier.gens(zs)
        testresults = classifier.test_by_images(images)
        for j in range(32):
            if np.mean(testresults[j] > 0.667):
                cv2.imwrite(os.path.join(outputpath, '{}.png'.format(passed)), images[j] * 255)
                passed += 1
                if passed == 50000:
                    flag = False
                    break



# generate tfrecord for StyleGAN finetuning
from dataset_tool import create_from_images
create_from_images(os.path.join(outputroot, '{}_filtered_tfrecord_{}'.format(argsk.datatype, tid)), outputpath, 1)


# Run StyleGAN finetuning

import copy
import dnnlib
from dnnlib import EasyDict

desc          = 'sgan'                                                                 # Description string included in result subdir name.
train         = EasyDict(run_func_name='training.training_loop.training_loop')         # Options for training loop.
G             = EasyDict(func_name='training.networks_stylegan.G_style')               # Options for generator network.
D             = EasyDict(func_name='training.networks_stylegan.D_basic')               # Options for discriminator network.
G_opt         = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                          # Options for generator optimizer.
D_opt         = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                          # Options for discriminator optimizer.
G_loss        = EasyDict(func_name='training.loss.G_logistic_nonsaturating')           # Options for generator loss.
D_loss        = EasyDict(func_name='training.loss.D_logistic_simplegp', r1_gamma=10.0) # Options for discriminator loss.
dataset       = EasyDict()                                                             # Options for load_dataset().
sched         = EasyDict()                                                             # Options for TrainingSchedule.
grid          = EasyDict(size='4k', layout='random')                                   # Options for setup_snapshot_image_grid().
metrics       = []                                                                     # Options for MetricGroup.
submit_config = dnnlib.SubmitConfig()                                                  # Options for dnnlib.submit_run().
tf_config     = {'rnd.np_random_seed': 1000}                                           # Options for tflib.init_tf().

# Dataset.
desc += '-ffhq';     dataset = EasyDict(tfrecord_dir='{}_filtered_tfrecord_{}'.format(argsk.datatype, tid));                 train.mirror_augment = True

# Number of GPUs.
n_gpu = len(argsk.gpuid.split(','))
if n_gpu == 2:
    desc += '-2gpu'; submit_config.num_gpus = 2; sched.minibatch_base = 8; sched.minibatch_dict = {4: 256, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8}
elif n_gpu == 4:
    desc += '-4gpu'; submit_config.num_gpus = 4; sched.minibatch_base = 16; sched.minibatch_dict = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16}
elif n_gpu == 8:
    desc += '-8gpu'; submit_config.num_gpus = 8; sched.minibatch_base = 32; sched.minibatch_dict = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32}
else:
    desc += '-1gpu'; submit_config.num_gpus = 1; sched.minibatch_base = 4; sched.minibatch_dict = {4: 128, 8: 128, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8, 512: 4}

# Default options.
sched.lod_training_kimg = 0; sched.lod_transition_kimg = 0; train.total_kimg = 200
sched.lod_initial_resolution = 1024
sched.G_lrate_dict = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
sched.D_lrate_dict = EasyDict(sched.G_lrate_dict)

if argsk.datatype == 'face':
    train.resume_run_id = './datasources/karras2019stylegan-ffhq-1024x1024.pkl'
elif argsk.datatype == 'bedroom':
    train.resume_run_id = './datasources/karras2019stylegan-bedrooms-256x256.pkl'

kwargs = EasyDict(train)
kwargs.update(G_args=G, D_args=D, G_opt_args=G_opt, D_opt_args=D_opt, G_loss_args=G_loss, D_loss_args=D_loss)
kwargs.update(dataset_args=dataset, sched_args=sched, grid_args=grid, metric_arg_list=metrics, tf_config=tf_config)
kwargs.submit_config = copy.deepcopy(submit_config)
kwargs.submit_config.run_dir_root = dnnlib.submission.submit.get_template_from_path(config.result_dir)
kwargs.submit_config.run_dir_ignore += config.run_dir_ignore
kwargs.submit_config.run_desc = desc
dnnlib.submit_run(**kwargs)
