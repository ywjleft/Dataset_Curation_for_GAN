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
parser.add_argument('-datatype', default='wood')
parser.add_argument('-classifier_n_gen', type=int, default=10000)

argsk, argsu = parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = argsk.gpuid

import config
from config import outputroot
tid = int(time.time())
outputpath = os.path.join(outputroot, '{}_finetune_{}'.format(argsk.datatype, tid))
os.makedirs(outputpath, exist_ok=True)

# filter ffhq dataset with trained classifier
from curation_system_texture import curation_system, saveBigImage

system = curation_system(len(argsk.gpuid.split(',')), argsk.datatype, output_path=outputpath, classifier_only=True)
system.saver_userdis.restore(system.sess, argsk.classifier_model)
system.logger.info('Restored: {}'.format(argsk.classifier_model))

system.logger.info('Generating training data...')
dataset = np.zeros([argsk.classifier_n_gen, system.imageres, system.imageres, 3], np.float32)
pointer = 0
flag = True
while flag:
    data_gen = system.gens(np.random.normal(size=(system.bs_gan, system.zdim)))
    testresults = system.test_by_images(data_gen)
    for b in range(system.bs_gan):
        if np.mean(testresults[b]) > 0.5:
            dataset[pointer] = data_gen[b]
            pointer += 1
            if pointer % 1000 == 0:
                print('{}/{} completed...'.format(pointer, argsk.classifier_n_gen))
            if pointer == argsk.classifier_n_gen:
                flag = False
                break

saveBigImage(os.path.join(outputpath, 'training_data_sample.png'), dataset[:128]**(1/2.2), 8, 16)
system.logger.info('Generation finished, shape of all data: {}'.format(np.shape(dataset)))

input_gaussian_array = np.random.normal(size = [128, 200])

#training
system.logger.info('---Start training...---')

for i in range(5000):
    data_dict = {}
    for i_d in range(2):
        realinput = np.reshape(dataset[np.random.choice(argsk.classifier_n_gen, system.bs_gan, replace=False)], (system.n_gpu, system.bs_gan_gpu, system.imageres, system.imageres, 3))
        feeddict = {}
        for gid in range(system.n_gpu):
            feeddict[system.tf_real_input_allgpu[gid]] = realinput[gid]
        system.sess.run(system.apply_gradient_op_d, feed_dict=feeddict)
    system.sess.run(system.apply_gradient_op_g)

    if((i+1) % 1000 == 0):
        system.logger.info('{}/{} completed...'.format(i+1,5000))
        images128 = system.gens(input_gaussian_array)
        saveBigImage(os.path.join(outputpath, 'gan_sample_{}.png'.format(i+1)), images128**(1/2.2), 8, 16)


system.logger.info('---Finished Training.---')
system.tf_saver.save(system.sess, os.path.join(outputpath, '{}_finetuned.tfmodel'.format(datatype)))
