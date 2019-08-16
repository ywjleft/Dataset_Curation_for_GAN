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

        if datatype == 'face':

            import facenet
            import models.inception_resnet_v1 as featurenet

            self.images_placeholder = tf.placeholder(tf.float32, (None, 160, 160, 3))
            self.phase_train_placeholder = tf.placeholder(tf.bool, ())

            prelogits, end_points = featurenet.inference(self.images_placeholder, 1.0, phase_train=self.phase_train_placeholder, bottleneck_layer_size=512, weight_decay=0.0)
            self.featuremap = end_points['PreLogitsFlatten']
            self.featuremap_size = self.featuremap.get_shape()[1]
            self.img_size_for_fm = 160
            facenetloader = tf.train.Saver()
            facenetloader.restore(tf.get_default_session(), './datasources/model-20180402-114759.ckpt-275')

            if not classifier_only:
                if os.path.isfile('./datasources/ffhq256.npy'):
                    self.allimages = np.load('./datasources/ffhq256.npy')
                else:
                    from config import ffhq_path
                    imagefiles = glob.glob(os.path.join(ffhq_path, '*'))
                    cnt = len(imagefiles)
                    self.allimages = np.zeros((cnt,256,256,3), np.uint8)
                    for i in range(cnt):
                        self.allimages[i] = cv2.resize(cv2.imread(imagefiles[i]), (256,256))
                    np.save('./datasources/ffhq256.npy', self.allimages)

                if os.path.isfile('./datasources/allfeatures.npy'):
                    self.allfeatures = np.load('./datasources/allfeatures.npy')
                else:
                    self.allfeatures = np.zeros((len(self.allimages), self.featuremap_size), np.float32)
                    for i in range(int(np.ceil(len(self.allimages)/32))):
                        images = self.allimages[i*32:(i+1)*32][:,29:189,9:169,:] / 255.0
                        self.allfeatures[i*32:(i+1)*32] = self.extract_features(images)
                    np.save('./datasources/allfeatures.npy', self.allfeatures)

                self.build_userdis_classifier()
                self.init_general()
                self.logger.info('initialization finished.')

            else:
                self.build_userdis_classifier_test()


        elif datatype == 'bedroom':

            import dnnlib
            import dnnlib.tflib as tflib
            _G, _D, self.Gs = pickle.load(open('./datasources/karras2019stylegan-bedrooms-256x256.pkl', 'rb'))
            self.fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
            self.zdim = self.Gs.input_shape[1]

            latents = np.random.normal(size=(20, self.Gs.input_shape[1]))
            images = self.Gs.run(latents, None, truncation_psi=0.7, randomize_noise=False, output_transform=self.fmt)
            #print(np.shape(images))

            import vgg16
            self.vgg16input = tf.placeholder("float", [None, 224, 224, 3])
            self.vgg = vgg16.Vgg16('./datasources/vgg16.npy')
            with tf.name_scope("vgg16"):
                self.vgg.build(self.vgg16input)

            self.featuremap_size = 25088
            self.img_size_for_fm = 224
            if not classifier_only:
                self.build_userdis_classifier()
                self.init_general()
                self.logger.info('initialization finished.')
            else:
                self.build_userdis_classifier_test()


        else:
            print('Invalid datatype!')

    def init_general(self):
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

        path = './expand.so'
        self.selvar = CDLL(path)
        self.selvar.calc.argtypes = [POINTER(c_double), POINTER(c_double), c_int, c_int, c_int, c_int, POINTER(c_int), POINTER(c_double)]
        self.selvar.calc.restype = None

        self.posz = []
        self.negz = []
        self.midz = []

        self.posfms = []
        self.negfms = []
        self.midfms = []

        self.selzs = []
        self.zsprev = None
        self.zscurr = None
        self.zspredprev = None
        self.zspredcurr = None

        os.makedirs('./static/RAMDISK/{}'.format(self.uid))
        self.status = 1
        self.round = 0
        if not self.start_posx is None:
            precomputed_augmented_images = np.zeros((128,self.img_size_for_fm,self.img_size_for_fm,3))
            for j in range(128):
                precomputed_augmented_images[j] = self.augment(self.start_posx)
            precomputed_fms = self.extract_features(precomputed_augmented_images)
            self.posfms += [precomputed_fms,]
        if not self.start_negx is None:
            precomputed_augmented_images = np.zeros((128,self.img_size_for_fm,self.img_size_for_fm,3))
            for j in range(128):
                precomputed_augmented_images[j] = self.augment(self.start_negz)
            precomputed_fms = self.extract_features(precomputed_augmented_images)
            self.negfms += [precomputed_fms,]


    def build_userdis_classifier(self):
        # create classifier for user preference
        import net_classifier
        classifier = getattr(net_classifier, 'featuremap_classifier_{}'.format(self.datatype))
        self.feature_phs = []
        self.pref_phs = []
        self.trainops = []
        self.losses = []
        self.preftests = []

        for i in range(4):
            with tf.variable_scope('userdis_{}'.format(i)):
                with tf.device('gpu:{}'.format(i % self.n_gpu)):
                    net_user_d = classifier(i, n_cluster = 2)
                    featureph = tf.placeholder(tf.float32, [32, self.featuremap_size])
                    preference = net_user_d.forward(featureph)
                    preference_test = tf.nn.softmax(net_user_d.forward(featureph, training=False))

                    preference_gt = tf.placeholder(tf.float32, [32, 2])
                    preference_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = preference_gt, logits = preference)
                    allvars = [var for var in tf.global_variables() if 'userdis_{}'.format(i) in var.name]
                    updateop = [uop for uop in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'userdis_{}'.format(i) in uop.name]
                    with tf.control_dependencies(updateop):
                        preference_trainop = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(preference_loss, var_list = allvars)

                    self.feature_phs.append(featureph)
                    self.pref_phs.append(preference_gt)
                    self.trainops.append(preference_trainop)
                    self.losses.append(preference_loss)
                    self.preftests.append(preference_test)

        var_to_train = [var for var in tf.global_variables() if 'userdis' in var.name]

        userdis_init = tf.variables_initializer(var_list=var_to_train)
        self.saver_userdis = tf.train.Saver(var_list = var_to_train, max_to_keep = 1)
        self.sess.run(userdis_init)

    def build_userdis_classifier_test(self):
        # create classifier for user preference
        import net_classifier
        classifier = getattr(net_classifier, 'featuremap_classifier_{}'.format(self.datatype))
        self.feature_phs = []
        self.preftests = []

        for i in range(4):
            with tf.variable_scope('userdis_{}'.format(i)):
                with tf.device('gpu:{}'.format(i % self.n_gpu)):
                    net_user_d = classifier(i, n_cluster = 2)
                    featureph = tf.placeholder(tf.float32, [32, self.featuremap_size])
                    preference_test = tf.nn.softmax(net_user_d.forward(featureph, training=False))

                    self.feature_phs.append(featureph)
                    self.preftests.append(preference_test)

        var_to_train = [var for var in tf.global_variables() if 'userdis' in var.name]

        userdis_init = tf.variables_initializer(var_list=var_to_train)
        self.saver_userdis = tf.train.Saver(var_list = var_to_train, max_to_keep = 1)
        self.sess.run(userdis_init)

    def gens(self, zs):
        if self.datatype == 'face':
            return self.allimages[zs] / 255.0
        elif self.datatype == 'bedroom':
            with self.sess.as_default():
                cnt = len(zs)
                result = np.zeros((cnt, 256, 256, 3))
                for i in range(int(np.ceil(cnt/32))):
                    low = i*32
                    high = min((i+1)*32, cnt)
                    zbatch = zs[low:high]
                    data_gen = self.Gs.run(np.array(zbatch), None, truncation_psi=0.7, randomize_noise=False, output_transform=self.fmt)
                    result[low:high] = data_gen[...,::-1] / 255
                return result


    def extract_features(self, images):
        if self.datatype == 'face':
            images = (np.array(images) * 255.0 - 127.5) / 128.0
            nrof_images = len(images)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / 128))
            emb_array = np.zeros((nrof_images, self.featuremap_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i*128
                end_index = min((i+1)*128, nrof_images)
                images_batch = images[start_index:end_index]
                feed_dict = { self.images_placeholder:images_batch, self.phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = self.sess.run(self.featuremap, feed_dict=feed_dict)
            return emb_array
        elif self.datatype == 'bedroom':
            images = np.array(images)
            cnt = len(images)
            result = np.zeros((cnt, 25088))
            for i in range(int(np.ceil(cnt/32))):
                low = i*32
                high = min((i+1)*32, cnt)
                xbatch = images[low:high][...,::-1]
                data_gen = self.sess.run(self.vgg.pool5, feed_dict={self.vgg16input:xbatch})
                result[low:high] = np.reshape(data_gen, (-1, 25088))
            return result

    def augment(self, image):
        if self.datatype == 'face':
            # disturb contrast & bright
            image = np.clip((image - 0.5) * (np.random.rand() * 0.1 + 0.95) + 0.5 + (np.random.rand() - 0.5) * 0.1, 0.0, 1.0)
            # crop & resize
            x0 = np.random.randint(19) + 20
            y0 = np.random.randint(19)
            image = image[x0:(x0+160),y0:(y0+160)]
            return image
        elif self.datatype == 'bedroom':
            # disturb contrast & bright
            image = np.clip((image - 0.5) * (np.random.rand() * 0.1 + 0.95) + 0.5 + (np.random.rand() - 0.5) * 0.1, 0.0, 1.0)
            # crop & resize
            x0 = np.random.randint(10)
            x1 = np.random.randint(10)
            y0 = np.random.randint(max(0, x0+x1-9), min(x0+x1+1, 10))
            y1 = x0 + x1 - y0
            image = cv2.resize(image[x0:(256-x1),y0:(256-y1)], (224,224))
            return image

    def genBatch(self, posfms, negfms, bs):
        featurebatch = np.zeros((bs,self.featuremap_size))
        labelbatch = np.zeros((bs,2))
        for i in range(bs):
            if np.random.randint(2) == 1:
                featurebatch[i] = posfms[np.random.randint(len(posfms))]
                labelbatch[i] = (1.0,0.0)
            else:
                featurebatch[i] = negfms[np.random.randint(len(negfms))]
                labelbatch[i] = (0.0,1.0)

        return featurebatch, labelbatch

    def trainDis(self, iters, ifsave=False, filename='model.ckpt'):
        self.logger.info('Start training.')
        posfms = np.reshape(self.posfms, (-1, self.featuremap_size))
        negfms = np.reshape(self.negfms, (-1, self.featuremap_size))

        loss_sum = np.zeros(4)
        for i in range(1, iters+1):
            featurebatch, labelbatch = self.genBatch(posfms, negfms, 32*4)
            feeddict = {self.feature_phs[0]:featurebatch[0:32], self.feature_phs[1]:featurebatch[32:2*32], self.feature_phs[2]:featurebatch[2*32:3*32], self.feature_phs[3]:featurebatch[3*32:4*32], self.pref_phs[0]:labelbatch[0:32], self.pref_phs[1]:labelbatch[32:2*32], self.pref_phs[2]:labelbatch[2*32:3*32], self.pref_phs[3]:labelbatch[3*32:4*32]}
            runout = self.sess.run(self.losses + self.trainops, feed_dict=feeddict)
            loss_sum += np.mean(runout[0:4], axis=1)

            if i % 500 == 0:
                self.logger.info('{}: {}'.format(i, loss_sum / 100))
                loss_sum = np.zeros(4)

        self.logger.info('Training finished.')
        if ifsave:
            self.saver_userdis.save(self.sess, os.path.join(self.output_path, 'model_{}.ckpt'.format(self.uid)))
            self.logger.info('Saved.')

    def test_by_zs(self, zs):
        fms = self.allfeatures[zs]
        cnt = len(zs)
        result = np.zeros((cnt, 4))
        fullrun = cnt // 32
        remain = cnt % 32
        for i in range(fullrun):
            featurebatch = fms[i*32:(i+1)*32]
            rbatch = np.array(self.sess.run(self.preftests, feed_dict={self.feature_phs[0]:featurebatch,self.feature_phs[1]:featurebatch,self.feature_phs[2]:featurebatch,self.feature_phs[3]:featurebatch}))[...,0] #4*32
            result[i*32:(i+1)*32] = np.transpose(rbatch)
        if remain > 0:
            featurebatch = np.zeros([32, self.featuremap_size])
            featurebatch[0:remain] = fms[fullrun*32:]
            rbatch = np.array(self.sess.run(self.preftests, feed_dict={self.feature_phs[0]:featurebatch,self.feature_phs[1]:featurebatch,self.feature_phs[2]:featurebatch,self.feature_phs[3]:featurebatch}))[...,0] #4*32
            result[fullrun*32:] = np.transpose(rbatch)[0:remain]
        return result

    def test_by_images(self, images):
        cnt = len(images)
        images = [cv2.resize(images[i], (self.img_size_for_fm,self.img_size_for_fm)) for i in range(cnt)]
        features = self.extract_features(images)
        result = np.zeros((cnt, 4))
        fullrun = cnt // 32
        remain = cnt % 32
        for i in range(fullrun):
            featurebatch = features[i*32:(i+1)*32]
            rbatch = np.array(self.sess.run(self.preftests, feed_dict={self.feature_phs[0]:featurebatch,self.feature_phs[1]:featurebatch,self.feature_phs[2]:featurebatch,self.feature_phs[3]:featurebatch}))[...,0] #4*32
            result[i*32:(i+1)*32] = np.transpose(rbatch)
        if remain > 0:
            featurebatch = np.zeros([32, self.featuremap_size])
            featurebatch[0:remain] = features[fullrun*32:]
            rbatch = np.array(self.sess.run(self.preftests, feed_dict={self.feature_phs[0]:featurebatch,self.feature_phs[1]:featurebatch,self.feature_phs[2]:featurebatch,self.feature_phs[3]:featurebatch}))[...,0] #4*32
            result[fullrun*32:] = np.transpose(rbatch)[0:remain]
        return result

    def getQuery(self):
        self.round += 1
        self.logger.info('Start point selection.')

        if self.datatype == 'face':
            if self.status == 1:
                zs = np.random.choice(len(self.allimages), 20, replace=False)
            elif self.status == 2:
                claslabeled = np.zeros(len(self.selzs)*4)
                userdisr = self.test_by_zs(self.selzs)
                for i in range(len(self.selzs)):
                    claslabeled[i*4:(i+1)*4] = userdisr[i]

                unselected = np.delete(np.array(range(len(self.allimages))), self.selzs)
                zcs = np.random.choice(unselected, 5000, replace=False)
                clasflat = np.zeros(5000*4)
                stdvars = np.zeros(5000)
                selected_id = (c_int*20)()
                userdisr = self.test_by_zs(zcs)
                for i in range(5000):
                    clasflat[i*4:(i+1)*4] = userdisr[i]
                    stdvars[i] = KLD(userdisr[i])

                self.selvar.calc(clasflat.ctypes.data_as(POINTER(c_double)), claslabeled.ctypes.data_as(POINTER(c_double)), 4, 5000, len(self.selzs), 20, selected_id, stdvars.ctypes.data_as(POINTER(c_double)))
                selected_id_array = np.array(selected_id)
                zs = [zcs[selected_id_array[i]] for i in range(20)]
                zspred = self.test_by_zs(zs)
                self.zspredprev = self.zspredcurr
                self.zspredcurr = zspred

        elif self.datatype == 'bedroom':
            if self.status == 1:
                zs = np.random.normal(size=(20, self.zdim))
            elif self.status == 2:
                claslabeled = np.zeros(len(self.selzs)*4)
                userdisr = self.test_by_images(self.gens(self.selzs))
                for i in range(len(self.selzs)):
                    claslabeled[i*4:(i+1)*4] = userdisr[i]

                zcs = np.random.normal(size=(5000, self.zdim))
                clasflat = np.zeros(5000*4)
                stdvars = np.zeros(5000)
                selected_id = (c_int*20)()

                xs = self.gens(zcs)
                userdisr = self.test_by_images(xs)
                for i in range(5000):
                    clasflat[i*4:(i+1)*4] = userdisr[i]
                    stdvars[i] = KLD(userdisr[i])

                self.selvar.calc(clasflat.ctypes.data_as(POINTER(c_double)), claslabeled.ctypes.data_as(POINTER(c_double)), 4, 5000, len(self.selzs), 20, selected_id, stdvars.ctypes.data_as(POINTER(c_double)))
                selected_id_array = np.array(selected_id)
                zs = [zcs[selected_id_array[i]] for i in range(20)]
                zspred = [userdisr[selected_id_array[i]] for i in range(20)]
                self.zspredprev = self.zspredcurr
                self.zspredcurr = zspred

        self.selzs += [zs[i] for i in range(len(zs))]
        self.zsprev = self.zscurr
        self.zscurr = zs

        saveSepImage('./static/RAMDISK/{}/round{}'.format(self.uid, self.round), self.gens(zs))
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
            saveBigImage(os.path.join(self.output_path, 'round{}_upd.png'.format(self.round)), xs, None, 10, userinput)
        else:
            labeltexts = []
            xs = self.gens(zs)
            for i in range(len(zs)):
                labeltexts += ['{}\n{:.4f}\n{:.4f}\n{:.4f}\n{:.4f}'.format(userinput[i],zspred[i][0],zspred[i][1],zspred[i][2],zspred[i][3])]
            saveBigImage(os.path.join(self.output_path, 'round{}_upd.png'.format(self.round)), xs, None, 10, labeltexts)

        for i in range(len(zs)):
            precomputed_augmented_images = np.zeros((128,self.img_size_for_fm,self.img_size_for_fm,3))
            for j in range(128):
                precomputed_augmented_images[j] = self.augment(xs[i])
            precomputed_fms = self.extract_features(precomputed_augmented_images)
            if userinput[i] == '2':
                self.posz += [zs[i].copy()]
                self.posfms += [precomputed_fms,]
            elif userinput[i] == '0':
                self.negz += [zs[i].copy()]
                self.negfms += [precomputed_fms,]
            elif userinput[i] == '1':
                self.midz += [zs[i].copy()]
                self.midfms += [precomputed_fms,]
        np.save(os.path.join(self.output_path, 'labeled_zs'), (self.posz,self.negz,self.midz))

        if self.status == 1:
            poscnt = len(self.posz)
            negcnt = len(self.negz)
            if poscnt >= 1 and negcnt >= 1:
                self.trainDis(2000 if self.datatype=='bedroom' else 5000, ifsave=True)
                self.status = 2

        elif self.status == 2:
            self.trainDis(1000 if self.datatype=='bedroom' else 2500, ifsave=True)

    def sendData(self, poszs, negzs, midzs = []):
        self.posz = poszs
        self.negz = negzs
        self.midz = midzs
