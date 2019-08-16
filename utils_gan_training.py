import tensorflow as tf

#function for averaging gradients
def average_gradient(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, var in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, axis=0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def _loss_g_wgan(fake_d):
    return -tf.reduce_mean(fake_d)

def _loss_d_hinge(real_d, fake_d):
    if(type(real_d) == list):
        print('Multi_Scale D.')
        _loss_list = []
        for i in range(len(real_d)):
            _loss_list.append(tf.reduce_mean(tf.nn.relu(1.0 - real_d[i]) + tf.nn.relu(1.0 + fake_d[i])))
        return tf.reduce_mean(_loss_list)
    else:
        return tf.reduce_mean(tf.nn.relu(1.0 - real_d) + tf.nn.relu(1.0 + fake_d))

#functions for comput gradient penalty   
def _loss_gp(interpolate_data, interpolate_d):
    if(type(interpolate_d) == list):
        _loss_list = []
        for i in range(len(interpolate_d)):
            gradients = tf.gradients(interpolate_d[i], interpolate_data)[0]
            gradient_squares = tf.reduce_sum(tf.square(gradients), reduction_indices = list(range(1, gradients.shape.ndims)))
            slope = tf.sqrt(gradient_squares + 1e-10)
            penalties = slope - 1.0
            _loss_list.append(tf.reduce_mean(tf.square(penalties)))
        return tf.reduce_mean(_loss_list)
    else:
        gradients = tf.gradients(interpolate_d, interpolate_data)[0]
        gradient_squares = tf.reduce_sum(tf.square(gradients), reduction_indices = list(range(1, gradients.shape.ndims)))
        slope = tf.sqrt(gradient_squares + 1e-10)
        penalties = slope - 1.0
        return tf.reduce_mean(tf.square(penalties))

def _interpolate(real_data, fake_data, net_d, proj_id = 0):
    batchsize = real_data.get_shape().as_list()[0]
    alpha_shape = [batchsize] + [1] * (real_data.shape.ndims - 1)
    _alpha = tf.random_uniform(shape=alpha_shape)

    differences = fake_data - real_data
    interpolates = real_data + (_alpha * differences)
    d_inter_raw = net_d.forward(interpolates*2-1)
    if(type(d_inter_raw) == list):
        d_inter = [item[...,proj_id] for item in d_inter_raw]
    else:
        d_inter = d_inter_raw[...,proj_id]
    return interpolates, d_inter