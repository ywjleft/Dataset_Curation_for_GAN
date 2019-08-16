import tensorflow as tf


class generator_256:
    inited = False
    g_id = 0

    def __init__(self, id = 0, n_ch = 3):
        self.g_id = id
        self.inited = False
        self.n_ch = n_ch

    def forward(self, z_or_f, training = True, with_fc = True):
        with tf.variable_scope('gen_{}'.format(self.g_id), reuse = self.inited):
            if with_fc:
                with tf.variable_scope('stage_inputfc'):
                    output = linear_sn(z_or_f, 512 * 256)
                    output = tf.reshape(output, [-1, 16, 16, 512], name = 'reshape')
                    output = tf.nn.relu(tf.layers.batch_normalization(output, training = training), name = 'bnrelu')
            else:
                output = z_or_f

            with tf.variable_scope('stage_256x32x32'):
                output = deconvBlock(output, 256, 4, 2, training)

            with tf.variable_scope('stage_128x64x64'):
                output = deconvBlock(output, 128, 4, 2, training)
                output = attention(output)
                
            with tf.variable_scope('stage_64x128x128'):
                output = deconvBlock(output, 64, 4, 2, training)
                
            with tf.variable_scope('stage_32x256x256'):
                output = deconvBlock(output, 32, 4, 2, training)
                
            with tf.variable_scope('stage_3x256x256'):
                output = convBlock(output, self.n_ch, 3, 1, training)

        self.inited = True
        return output



class discriminator_256:
    inited = False
    d_id = 0

    def __init__(self, id = 0):
        self.inited = False
        self.d_id = id  

    def forward(self, input, training = True, with_fc = True):
        with tf.variable_scope('dis_{}'.format(self.d_id), reuse = self.inited):
            with tf.variable_scope('stage_32x256x256'):
                output = convBlock3(input, 32, 3, 1, training)

            with tf.variable_scope('stage_64x128x128_1'):
                output = convBlock3(output, 64, 4, 2, training)

            with tf.variable_scope('stage_64x128x128_2'):
                output = convBlock3(output, 64, 3, 1, training)

            with tf.variable_scope('stage_128x64x64_1'):
                output = convBlock3(output, 128, 4, 2, training)

            with tf.variable_scope('stage_128x64x64_2'):
                output = convBlock3(output, 128, 3, 1, training)

            with tf.variable_scope('stage_256x32x32_1'):
                output = convBlock3(output, 256, 4, 2, training)

            with tf.variable_scope('stage_256x32x32_2'):
                output = convBlock3(output, 256, 3, 1, training)
                output = attention(output)

            with tf.variable_scope('stage_512x16x16_1'):
                output = convBlock3(output, 512, 4, 2, training)

            with tf.variable_scope('stage_512x16x16_2'):
                output = convBlock3(output, 512, 3, 1, training)

            if with_fc:
                with tf.variable_scope('stage_outputfc'):
                    output = tf.layers.flatten(output)
                    output = linear_sn(output, 1, use_sn = True)

            self.inited = True
            return output




class image_classifier_half:
    inited = False
    c_id = 0

    def __init__(self, id = 0, n_cluster = 20):
        self.inited = False
        self.c_id = id      
        self.n_cluster = n_cluster    
    
    def forward(self, input, training = True, _output_feature = False):
        with tf.variable_scope('classifier_{}'.format(self.c_id), reuse = self.inited):
            self.inited = True

            with tf.variable_scope('stage_16x64x64'):
                output = convBlock2(input, 16, 3, 1, training)

            with tf.variable_scope('stage_32x32x32'):
                output = convBlock2(output, 32, 3, 2, training)

            with tf.variable_scope('stage_64x16x16'):
                output = convBlock2(output, 64, 3, 2, training)

            with tf.variable_scope('stage_128x8x8'):
                output = convBlock2(output, 128, 3, 2, training)

            with tf.variable_scope('stage_256x4x4'):
                output = convBlock2(output, 256, 3, 2, training)

            with tf.variable_scope('stage_512x2x2'):
                output = convBlock2(output, 512, 3, 2, training)

            with tf.variable_scope('stage_flatten'):
                feature = tf.layers.flatten(output)

            if _output_feature:
                return feature

            with tf.variable_scope('out_classify'):
                finaloutput = tf.layers.dense(feature, self.n_cluster, kernel_initializer = tf.contrib.layers.variance_scaling_initializer())    

            return finaloutput


def deconvBlock(input, channal_out, k_size, stride, training = True):
    output = deconv2d_sn(input, channal_out, k_size, k_size, stride, stride, use_sn = True, use_bias = True)
    output = tf.layers.batch_normalization(output, training = training, fused = True)
    output = tf.nn.relu(output)

    return output


def convBlock(input, channal_out, k_size, stride, training = True):
    output = conv2d_sn(input, channal_out, k_size, k_size, stride, stride, use_sn = True, use_bias = True)
    output = tf.sigmoid(output)

    return output


def convBlock2(input, channal_out, k_size, stride, training = True):
    output = tf.layers.conv2d(
        input, 
        filters = channal_out, 
        kernel_size = k_size, 
        strides = stride, 
        padding = 'same',
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),
        use_bias = True
    )

    output = tf.layers.batch_normalization(output, training=training, fused = True)
    output = lrelu(output, leak = 0.1)

    return output


def convBlock3(input, channal_out, k_size, stride, training = True):
    output = conv2d_sn(input, channal_out, k_size, k_size, stride, stride, use_sn = True, use_bias = True)
    output = lrelu(output, leak = 0.1)

    return output


def lrelu(x, leak=0.2, name='lrelu'): 
    return tf.maximum(x, leak*x, name = name)


#specturm normalization
#https://github.com/google/compare_gan/blob/master/compare_gan/src/gans/ops.py

def spectral_norm(input_):
    """Performs Spectral Normalization on a weight tensor."""
    if len(input_.shape) < 2:
        raise ValueError(
            "Spectral norm can only be applied to multi-dimensional tensors")

    # The paper says to flatten convnet kernel weights from (C_out, C_in, KH, KW)
    # to (C_out, C_in * KH * KW). But Sonnet's and Compare_gan's Conv2D kernel
    # weight shape is (KH, KW, C_in, C_out), so it should be reshaped to
    # (KH * KW * C_in, C_out), and similarly for other layers that put output
    # channels as last dimension.
    # n.b. this means that w here is equivalent to w.T in the paper.
    w = tf.reshape(input_, (-1, input_.shape[-1]))

    # Persisted approximation of first left singular vector of matrix `w`.

    u_var = tf.get_variable(
        input_.name.replace(":", "") + "/u_var",
        shape=(w.shape[0], 1),
        dtype=w.dtype,
        initializer=tf.random_normal_initializer(),
        trainable=False)
    u = u_var

    # Use power iteration method to approximate spectral norm.
    # The authors suggest that "one round of power iteration was sufficient in the
    # actual experiment to achieve satisfactory performance". According to
    # observation, the spectral norm become very accurate after ~20 steps.

    power_iteration_rounds = 1
    for _ in range(power_iteration_rounds):
        # `v` approximates the first right singular vector of matrix `w`.
        v = tf.nn.l2_normalize(
            tf.matmul(tf.transpose(w), u), axis=None, epsilon=1e-12)
        u = tf.nn.l2_normalize(tf.matmul(w, v), axis=None, epsilon=1e-12)

    # Update persisted approximation.
    with tf.control_dependencies([tf.assign(u_var, u, name="update_u")]):
        u = tf.identity(u)

    # The authors of SN-GAN chose to stop gradient propagating through u and v.
    # In johnme@'s experiments it wasn't clear that this helps, but it doesn't
    # seem to hinder either so it's kept in order to be a faithful implementation.
    u = tf.stop_gradient(u)
    v = tf.stop_gradient(v)

    # Largest singular value of `w`.
    norm_value = tf.matmul(tf.matmul(tf.transpose(u), w), v)
    norm_value.shape.assert_is_fully_defined()
    norm_value.shape.assert_is_compatible_with([1, 1])

    w_normalized = w / norm_value

    # Unflatten normalized weights to match the unnormalized tensor.
    w_tensor_normalized = tf.reshape(w_normalized, input_.shape)
    return w_tensor_normalized

def linear_sn(input_,
           output_size,
           scope=None,
           initializer=tf.contrib.layers.variance_scaling_initializer(),
           bias_start=0.0,
           use_sn=True,
           use_bias=True):

    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable(
            "Matrix", [shape[1], output_size],
            tf.float32,
            initializer)
        if use_sn:
            output = tf.matmul(input_, spectral_norm(matrix))
        else:
            output = tf.matmul(input_, matrix)
        
        if(use_bias):
            bias = tf.get_variable(
                "bias", [output_size], initializer=tf.constant_initializer(bias_start))
            output = output + bias
    return output   

def conv2d_sn(input_, output_dim, k_h, k_w, d_h, d_w, name="conv2d",
           initializer=tf.contrib.layers.variance_scaling_initializer(), use_sn=False, use_bias = True):
    with tf.variable_scope(name):
        w = tf.get_variable(
            "w", [k_h, k_w, input_.get_shape()[-1], output_dim],
            initializer=initializer)
        if use_sn:
            conv = tf.nn.conv2d(
                input_, spectral_norm(w), strides=[1, d_h, d_w, 1], padding="SAME")
        else:
            conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding="SAME")
        if use_bias:
            biases = tf.get_variable(
                "biases", [output_dim], initializer=tf.constant_initializer(0.0))
            return tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        else:
            return conv

def deconv2d_sn(input_, output_dim, k_h, k_w, d_h, d_w, name="deconv2d",
           initializer=tf.contrib.layers.variance_scaling_initializer(), use_sn=False, use_bias = True):

    input_shape = input_.get_shape().as_list()
    with tf.variable_scope(name):
        w = tf.get_variable(
            "w", [k_h, k_w, output_dim, input_.get_shape()[-1]],
            initializer=initializer)
        if use_sn:
            conv = tf.nn.conv2d_transpose(
                input_, spectral_norm(w), output_shape = [input_shape[0], input_shape[1] * 2, input_shape[2] * 2, output_dim], strides=[1, d_h, d_w, 1], padding="SAME")
        else:
            conv = tf.nn.conv2d_transpose(input_, w, strides=[1, d_h, d_w, 1], padding="SAME")
        if use_bias:
            biases = tf.get_variable(
                "biases", [output_dim], initializer=tf.constant_initializer(0.0))
            return tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        else:
            return conv         



def hw_flatten(x):
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])
#https://github.com/taki0112/Self-Attention-GAN-Tensorflow
def attention(input):
    input_shape = input.get_shape().as_list()
    ch = input_shape[-1]

    f = tf.layers.conv2d(input, 
        filters = ch // 8, 
        kernel_size = 1, 
        strides = 1, 
        padding = 'same',
        use_bias = False)

    g = tf.layers.conv2d(input, 
        filters = ch // 8, 
        kernel_size = 1, 
        strides = 1, 
        padding = 'same',
        use_bias = False)

    g_flatten = hw_flatten(g)
    f_flatten = hw_flatten(f)
    
    s = tf.matmul(g_flatten, f_flatten, transpose_b = True)

    h = tf.layers.conv2d(input, 
        filters = ch, 
        kernel_size = 1, 
        strides = 1, 
        padding = 'same',
        use_bias = False)

    beta = tf.nn.softmax(s, axis = -1)

    o = tf.matmul(beta, hw_flatten(h))
    gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

    o = tf.reshape(o, shape=input.shape)
    x = gamma * o + input
    
    return x