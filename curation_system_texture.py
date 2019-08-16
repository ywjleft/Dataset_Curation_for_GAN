import sys, os, time, cv2, pickle, logging, math, glob
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from ctypes import *


def KLD(inputarray):
    m = np.mean(inputarray)
    KL = max(1e-20, np.mean([(inputarray[i] * np.log((inputarray[i]+1e-10)/(m+1e-10)) + (1-inputarray[i]) * np.log((1+1e-10-inputarray[i])/(1+1e-10-m))) for i in range(4)]))
    return KL

def saveBigImage(file, images, row=None, column=None, labels=None):
    if len(images) > 0:
        if column is None and row is None:
            column = 10
        if column is None:
            column = int(np.ceil(len(images) / row))
        if row is None:
            row = int(np.ceil(len(images) / column))
        height = np.shape(images[0])[0]
        width = np.shape(images[0])[1]
        bigimage = np.zeros((height*row,width*column,3))
        for i in range(len(images)):
            rowi = i // column
            columni = i % column
            bigimage[height*rowi:height*(rowi+1),width*columni:width*(columni+1),:] = images[i]
        if labels is None:
            cv2.imwrite(file, bigimage*255)
        else:
            ldrimage = (np.maximum(np.minimum(bigimage, 1.0), 0.0) * 255).astype(np.uint8)[:,:,::-1]
            img = Image.fromarray(ldrimage, 'RGB')
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype('FreeMono.ttf',20)
            for i in range(len(images)):
                rowi = i // column
                columni = i % column
                text = '{}'.format(labels[i])
                if np.mean(images[i]) < 0.5:
                    draw.multiline_text((columni*width+10,rowi*height+10), text, 'white', font)
                else:
                    draw.multiline_text((columni*width+10,rowi*height+10), text, 'black', font)
            img.save(file, 'PNG')

def saveSepImage(prefex, images):
    for i in range(len(images)):
        cv2.imwrite('{}_{}.png'.format(prefex, i), images[i]*255)


class curation_system:

    def __init__(self, n_gpu, datatype, uid=None, output_path=None, enable_simul=None, start_posx=None, start_negx=None, classifier_only=False):
        self.n_gpu = n_gpu
        self.datatype = datatype
        self.uid = uid
        self.output_path = output_path
        self.enable_simul = enable_simul
        self.start_posx = start_posx
        self.start_negx = start_negx

        #setup tf
        cfg = tf.ConfigProto()
        cfg.allow_soft_placement=True
        cfg.gpu_options.allow_growth=True
        self.sess = tf.InteractiveSession(config=cfg)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(self.output_path, 'training_log.txt'))
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)

        if not classifier_only:
            self.imageres = 256
            self.bs_gan = 32
            self.zdim = 200
            self.bs_clas = 32
            self.halfres = 64

            import net_texture_gan
            generator = getattr(net_texture_gan, 'generator_256')
            net_g = generator(0, 3)

            self.bs_gan_gpu = self.bs_gan // n_gpu

            self.tf_z_input_allgpu = []
            self.tf_test_image_allgpu = []
            with tf.name_scope('generator'):
                for gid in range(n_gpu):
                    with tf.device('/gpu:{}'.format(gid)):
                        tf_z_input = tf.placeholder(tf.float32, shape=[self.bs_gan_gpu, self.zdim])
                        tf_test_image = net_g.forward(tf_z_input, training=False)
                        self.tf_z_input_allgpu.append(tf_z_input)
                        self.tf_test_image_allgpu.append(tf_test_image)

            tf_saver = tf.train.Saver()
            tf_saver.restore(self.sess, './datasources/{}.tfmodel'.format(datatype))

            # create classifier for user preference
            classifier = getattr(net_texture_gan, 'image_classifier_half')
            self.image_phs = []
            self.pref_phs = []
            self.trainops = []
            self.losses = []
            self.preftests = []

            for i in range(4):
                with tf.variable_scope('userdis_{}'.format(i)):
                    with tf.device('gpu:{}'.format(i % self.n_gpu)):
                        net_user_d = classifier(i, n_cluster = 2)
                        imageph = tf.placeholder(tf.float32, [self.bs_clas, self.halfres, self.halfres, 3])
                        preference = net_user_d.forward(imageph*2-1)
                        preference_test = tf.nn.softmax(net_user_d.forward(imageph*2-1, training=False))

                        preference_gt = tf.placeholder(tf.float32, [self.bs_clas, 2])
                        preference_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = preference_gt, logits = preference)
                        allvars = [var for var in tf.global_variables() if 'userdis_{}'.format(i) in var.name]
                        updateop = [uop for uop in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'userdis_{}'.format(i) in uop.name]
                        with tf.control_dependencies(updateop):
                            preference_trainop = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(preference_loss, var_list = allvars)

                        self.image_phs.append(imageph)
                        self.pref_phs.append(preference_gt)
                        self.trainops.append(preference_trainop)
                        self.losses.append(preference_loss)
                        self.preftests.append(preference_test)

            var_to_train = [var for var in tf.global_variables() if 'userdis' in var.name]

            userdis_init = tf.variables_initializer(var_list=var_to_train)
            self.saver_userdis = tf.train.Saver(var_list = var_to_train, max_to_keep = 1)
            self.sess.run(userdis_init)

            path = './expand.so'
            self.selvar = CDLL(path)
            self.selvar.calc.argtypes = [POINTER(c_double), POINTER(c_double), c_int, c_int, c_int, c_int, POINTER(c_int), POINTER(c_double)]
            self.selvar.calc.restype = None

            self.posz = []
            self.negz = []
            self.midz = []

            self.selzs = []
            self.zsprev = None
            self.zscurr = None
            self.zspredprev = None
            self.zspredcurr = None

            os.makedirs('./static/RAMDISK/{}'.format(self.uid))
            self.status = 1
            self.round = 0


        else:
            self.imageres = 256
            self.bs_gan = 32
            self.zdim = 200
            self.bs_clas = 32
            self.halfres = 64

            from utils_gan_training import _loss_g_wgan, _loss_d_hinge, _loss_gp, _interpolate, average_gradient
            import net_texture_gan
            generator = getattr(net_texture_gan, 'generator_256')
            net_g = generator(0, 3)

            self.bs_gan_gpu = self.bs_gan // n_gpu

            self.tf_z_input_allgpu = []
            self.tf_test_image_allgpu = []
            tf_z_random_allgpu = []
            tf_fake_image_allgpu = []

            with tf.name_scope('generator'):
                for gid in range(n_gpu):
                    with tf.device('/gpu:{}'.format(gid)):
                        tf_z_input = tf.placeholder(tf.float32, shape=[self.bs_gan_gpu, self.zdim])
                        tf_test_image = net_g.forward(tf_z_input, training=False)
                        tf_z_random = tf.random_normal(shape=[self.bs_gan_gpu, self.zdim])
                        tf_fake_image = net_g.forward(tf_z_random)

                        self.tf_z_input_allgpu.append(tf_z_input)
                        self.tf_test_image_allgpu.append(tf_test_image)
                        tf_z_random_allgpu.append(tf_z_random)
                        tf_fake_image_allgpu.append(tf_fake_image)

            discriminator = getattr(net_texture_gan, 'discriminator_256')
            net_d = discriminator(0)

            self.tf_real_input_allgpu = []
            tf_d_real_allgpu = []
            tf_d_fake_allgpu = []

            with tf.name_scope('discrimintor'):
                for gid in range(n_gpu):
                    with tf.device('/gpu:{}'.format(gid)):
                        tf_real_input = tf.placeholder(tf.float32, shape=[self.bs_gan_gpu, self.imageres, self.imageres, 3])
                        tf_d_real = net_d.forward(tf_real_input*2-1)
                        tf_d_fake = net_d.forward(tf_fake_image_allgpu[gid]*2-1)

                        self.tf_real_input_allgpu.append(tf_real_input)
                        tf_d_real_allgpu.append(tf_d_real)
                        tf_d_fake_allgpu.append(tf_d_fake)

            tf_d_loss_allgpu = []
            tf_g_loss_allgpu = []
            tf_inter_loss_allgpu = []

            with tf.name_scope('loss_wgan_hinge'):
                for gid in range(n_gpu):
                    with tf.device('/gpu:{}'.format(gid)):
                        tf_d_loss = _loss_d_hinge(tf_d_real_allgpu[gid], tf_d_fake_allgpu[gid])
                        tf_g_loss = _loss_g_wgan(tf_d_fake_allgpu[gid])
                        _inter_data, _inter_d = _interpolate(self.tf_real_input_allgpu[gid], tf_fake_image_allgpu[gid], net_d, 0)
                        tf_inter_loss = 10.0 * _loss_gp(_inter_data, _inter_d)

                        tf_d_loss_allgpu.append(tf_d_loss)
                        tf_g_loss_allgpu.append(tf_g_loss)
                        tf_inter_loss_allgpu.append(tf_inter_loss)


            var_g = [var for var in tf.trainable_variables() if 'gen_' in var.name]
            var_d = [var for var in tf.trainable_variables() if 'dis_' in var.name]

            with tf.name_scope('Solver'):
                with tf.device('/cpu:0'):
                    tf_G_solver = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.999)
                    tf_D_solver = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.999)


                tower_grads_d = []
                tower_grads_g = []

                for gid in range(self.n_gpu):
                    with tf.device('/gpu:{}'.format(gid)):
                        grad_tower_d = tf_D_solver.compute_gradients(tf_d_loss_allgpu[gid] + tf_inter_loss_allgpu[gid], var_d)
                        grad_tower_g = tf_G_solver.compute_gradients(tf_g_loss_allgpu[gid], var_g)

                        tower_grads_d.append(grad_tower_d)
                        tower_grads_g.append(grad_tower_g)

                with tf.device('/cpu:0'):
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    with tf.control_dependencies(update_ops):
                        grad_avg_d = average_gradient(tower_grads_d)
                        self.apply_gradient_op_d = tf_D_solver.apply_gradients(grad_avg_d)

                        grad_avg_g = average_gradient(tower_grads_g)               
                        self.apply_gradient_op_g = tf_G_solver.apply_gradients(grad_avg_g)

            self.tf_saver = tf.train.Saver()
            self.tf_saver.restore(self.sess, './datasources/{}.tfmodel'.format(datatype))


            # create classifier for user preference
            classifier = getattr(net_texture_gan, 'image_classifier_half')
            self.image_phs = []
            self.preftests = []

            for i in range(4):
                with tf.variable_scope('userdis_{}'.format(i)):
                    with tf.device('gpu:{}'.format(i % self.n_gpu)):
                        net_user_d = classifier(i, n_cluster = 2)
                        imageph = tf.placeholder(tf.float32, [self.bs_clas, self.halfres, self.halfres, 3])
                        preference_test = tf.nn.softmax(net_user_d.forward(imageph*2-1, training=False))

                        self.image_phs.append(imageph)
                        self.preftests.append(preference_test)

            var_to_train = [var for var in tf.global_variables() if 'userdis' in var.name]

            userdis_init = tf.variables_initializer(var_list=var_to_train)
            self.saver_userdis = tf.train.Saver(var_list = var_to_train, max_to_keep = 1)
            self.sess.run(userdis_init)

    def gens(self, zs):
        cnt = len(zs)
        result = np.zeros((cnt, self.imageres, self.imageres, 3))
        fullrun = cnt // self.bs_gan
        remain = cnt % self.bs_gan
        for i in range(fullrun):
            zbatch = np.reshape(zs[i*self.bs_gan:(i+1)*self.bs_gan], (self.n_gpu, self.bs_gan//self.n_gpu, self.zdim))
            feeddict = {}
            for gid in range(self.n_gpu):
                feeddict[self.tf_z_input_allgpu[gid]] = zbatch[gid]
            data_gen = self.sess.run(self.tf_test_image_allgpu, feed_dict=feeddict)
            result[i*self.bs_gan:(i+1)*self.bs_gan] = np.reshape(np.minimum(np.maximum(data_gen,0.0),1.0), (self.bs_gan, self.imageres, self.imageres, 3))
        if remain > 0:
            zbatch = np.zeros((self.bs_gan, self.zdim))
            zbatch[0:remain] = zs[fullrun*self.bs_gan:]
            zbatch = np.reshape(zbatch, (self.n_gpu, self.bs_gan//self.n_gpu, self.zdim))
            feeddict = {}
            for gid in range(self.n_gpu):
                feeddict[self.tf_z_input_allgpu[gid]] = zbatch[gid]
            data_gen = self.sess.run(self.tf_test_image_allgpu, feed_dict=feeddict)
            result[fullrun*self.bs_gan:] = np.reshape(np.minimum(np.maximum(data_gen,0.0),1.0), (self.bs_gan, self.imageres, self.imageres, 3))[0:remain]
        return result

    def genBatch(self, posx, negx, dim, bs):
        imagesize = np.shape(posx[0])[0]
        imagebatch = np.zeros((bs,dim,dim,3))
        labelbatch = np.zeros((bs,2))
        for i in range(bs):
            cx = np.random.randint(imagesize-dim)
            cy = np.random.randint(imagesize-dim)
            if np.random.randint(2) == 1:
                imagebatch[i] = posx[np.random.randint(len(posx))][cx:(cx+dim),cy:(cy+dim),:]
                labelbatch[i] = (1.0,0.0)
            else:
                imagebatch[i] = negx[np.random.randint(len(negx))][cx:(cx+dim),cy:(cy+dim),:]
                labelbatch[i] = (0.0,1.0)
        return imagebatch, labelbatch

    def trainDis(self, iters, ifsave=False, filename='model.ckpt'):
        posx256 = self.gens(self.posz)
        negx256 = self.gens(self.negz)
        posx = np.array([cv2.resize(posx256[i], (128,128)) for i in range(len(self.posz))])
        negx = np.array([cv2.resize(negx256[i], (128,128)) for i in range(len(self.negz))])
        if not self.start_posx is None:
            posx = np.concatenate([posx, cv2.resize(self.start_posx[np.newaxis,...], (128,128))], axis=0)
        if not self.start_negx is None:
            negx = np.concatenate([negx, cv2.resize(self.start_negx[np.newaxis,...], (128,128))], axis=0)

        self.logger.info('Start training.')
        loss_sum = np.zeros(4)
        for i in range(1, iters+1):
            imagebatch, labelbatch = self.genBatch(posx, negx, self.halfres, self.bs_clas*4)
            feeddict = {self.image_phs[0]:imagebatch[0:self.bs_clas], self.image_phs[1]:imagebatch[self.bs_clas:2*self.bs_clas], self.image_phs[2]:imagebatch[2*self.bs_clas:3*self.bs_clas], self.image_phs[3]:imagebatch[3*self.bs_clas:4*self.bs_clas], self.pref_phs[0]:labelbatch[0:self.bs_clas], self.pref_phs[1]:labelbatch[self.bs_clas:2*self.bs_clas], self.pref_phs[2]:labelbatch[2*self.bs_clas:3*self.bs_clas], self.pref_phs[3]:labelbatch[3*self.bs_clas:4*self.bs_clas]}
            runout = self.sess.run(self.losses + self.trainops, feed_dict=feeddict)
            loss_sum += np.mean(runout[0:4], axis=1)

            if i % 500 == 0:
                self.logger.info('{}: {}'.format(i, loss_sum / 100))
                loss_sum = np.zeros(4)

        self.logger.info('Training finished.')
        if ifsave:
            self.saver_userdis.save(self.sess, os.path.join(self.output_path, 'model_{}.ckpt'.format(self.uid)))
            self.logger.info('Saved.')

    def test_by_images(self, images):
        cnt = len(images)
        result = np.zeros((cnt, 4))
        fullrun = (cnt * 16) // self.bs_clas
        remain = (cnt * 16) % self.bs_clas
        imagebatch = np.zeros((self.bs_clas, self.halfres, self.halfres, 3))
        for i in range(fullrun):
            for j in range(self.bs_clas // 16):
                x = cv2.resize(images[i*self.bs_clas//16+j], (128,128))
                imagebatch[j*16:(j+1)*16] = [x[0:64,0:64], x[21:85,0:64], x[43:107,0:64], x[64:128,0:64], x[0:64,21:85], x[21:85,21:85], x[43:107,21:85], x[64:128,21:85], x[0:64,43:107], x[21:85,43:107], x[43:107,43:107], x[64:128,43:107], x[0:64,64:128], x[21:85,64:128], x[43:107,64:128], x[64:128,64:128]]
            rbatch = np.array(self.sess.run(self.preftests, feed_dict={self.image_phs[0]:imagebatch,self.image_phs[1]:imagebatch,self.image_phs[2]:imagebatch,self.image_phs[3]:imagebatch}))[...,0] #4*self.bs_clas
            for j in range(self.bs_clas // 16):
                result[i*self.bs_clas//16+j] = np.mean(rbatch[:,j*16:(j+1)*16], axis=1)
        if remain > 0:
            imagebatch = np.zeros((self.bs_clas, self.halfres, self.halfres, 3))
            for j in range(remain // 16):
                x = cv2.resize(images[fullrun*self.bs_clas//16+j], (128,128))
                imagebatch[j*16:(j+1)*16] = [x[0:64,0:64], x[21:85,0:64], x[43:107,0:64], x[64:128,0:64], x[0:64,21:85], x[21:85,21:85], x[43:107,21:85], x[64:128,21:85], x[0:64,43:107], x[21:85,43:107], x[43:107,43:107], x[64:128,43:107], x[0:64,64:128], x[21:85,64:128], x[43:107,64:128], x[64:128,64:128]]
            rbatch = np.array(self.sess.run(self.preftests, feed_dict={self.image_phs[0]:imagebatch,self.image_phs[1]:imagebatch,self.image_phs[2]:imagebatch,self.image_phs[3]:imagebatch}))[...,0] #4*self.bs_clas
            for j in range(remain // 16):
                result[fullrun*self.bs_clas//16+j] = np.mean(rbatch[:,j*16:(j+1)*16], axis=1)
        return result

    def getQuery(self):
        self.round += 1
        self.logger.info('Start point selection.')

        if self.status == 1:
            zs = np.random.normal(size=(20, self.zdim))
            xs = self.gens(zs)
        elif self.status == 2:
            claslabeled = np.zeros(len(self.selzs)*4)
            userdisr = self.test_by_images(self.gens(self.selzs))
            for i in range(len(self.selzs)):
                claslabeled[i*4:(i+1)*4] = userdisr[i]

            sample_total = 50000
            zcs = np.zeros((sample_total, self.zdim))
            clasflat = np.zeros(sample_total*4)
            stdvars = np.zeros(sample_total)
            selected_id = (c_int*20)()

            for retry in range(10):
                zcs[retry*5000:(retry+1)*5000] = np.random.normal(size=(5000, self.zdim))

                xcs = self.gens(zcs[retry*5000:(retry+1)*5000])
                userdisr = self.test_by_images(xcs)
                for i in range(5000):
                    clasflat[(retry*5000+i)*4:(retry*5000+i+1)*4] = userdisr[i]
                    stdvars[retry*5000+i] = KLD(userdisr[i])

                self.selvar.calc(clasflat.ctypes.data_as(POINTER(c_double)), claslabeled.ctypes.data_as(POINTER(c_double)), 4, (retry+1)*5000, len(self.selzs), 20, selected_id, stdvars.ctypes.data_as(POINTER(c_double)))
                selected_id_array = np.array(selected_id)

                if all(stdvars[selected_id_array] > 0.05) or retry == 9:
                    self.logger.info('candidate number: {}*5000'.format(retry+1))
                    zs = [zcs[selected_id_array[i]] for i in range(20)]

                    xs = self.gens(zs)
                    zspred = self.test_by_images(xs)
                    self.zspredprev = self.zspredcurr
                    self.zspredcurr = zspred
                    break

        self.selzs += [zs[i] for i in range(len(zs))]
        self.zsprev = self.zscurr
        self.zscurr = zs

        saveSepImage('./static/RAMDISK/{}/round{}'.format(self.uid, self.round), xs**(1/2.2))
        self.logger.info('Selection finished.')
        d0 = {'number': 20, 'prefix': 'static/RAMDISK/{}/round{}'.format(self.uid, self.round)}
        return {'d0': d0, 'zs': zs}


    def sendResult(self, userinput):
        self.logger.info('got input.')

        if self.enable_simul:
            zs = self.zsprev
            zspred = self.zspredprev
        else:
            zs = self.zscurr
            zspred = self.zspredcurr

        if zspred is None:
            xs = self.gens(zs)
            saveBigImage(os.path.join(self.output_path, 'round{}_upd.png'.format(self.round)), xs**(1/2.2), None, 10, userinput)
        else:
            labeltexts = []
            xs = self.gens(zs)
            for i in range(len(zs)):
                labeltexts += ['{}\n{:.4f}\n{:.4f}\n{:.4f}\n{:.4f}'.format(userinput[i],zspred[i][0],zspred[i][1],zspred[i][2],zspred[i][3])]
            saveBigImage(os.path.join(self.output_path, 'round{}_upd.png'.format(self.round)), xs**(1/2.2), None, 10, labeltexts)

        for i in range(len(zs)):
            if userinput[i] == '2':
                self.posz += [zs[i].copy()]
            elif userinput[i] == '0':
                self.negz += [zs[i].copy()]
            elif userinput[i] == '1':
                self.midz += [zs[i].copy()]
        np.save(os.path.join(self.output_path, 'labeled_zs'), (self.posz,self.negz,self.midz))

        if self.status == 1:
            poscnt = len(self.posz) + (not self.start_posx is None)
            negcnt = len(self.negz) + (not self.start_negx is None)
            if poscnt >= 1 and negcnt >= 1:
                self.trainDis(5000, ifsave=True)
                self.status = 2

        elif self.status == 2:
            self.trainDis(2500, ifsave=True)

    def sendData(self, poszs, negzs, midzs = []):
        self.posz = poszs
        self.negz = negzs
        self.midz = midzs
