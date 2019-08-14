import tensorflow as tf

class featuremap_classifier_face:
    inited = False
    c_id = 0

    def __init__(self, id = 0, n_cluster = 20):
        self.inited = False
        self.c_id = id      
        self.n_cluster = n_cluster    
    
    def forward(self, input, training = True):
        with tf.variable_scope('classifier_{}'.format(self.c_id), reuse = self.inited):
            self.inited = True

            with tf.variable_scope('fc0'):
                output = tf.layers.dense(input, 512, kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), activation = tf.nn.leaky_relu)

            with tf.variable_scope('fc1'):
                output = tf.layers.dense(output, 50, kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), activation = tf.nn.leaky_relu)

            with tf.variable_scope('fc2'):
                output = tf.layers.dense(output, self.n_cluster, kernel_initializer = tf.contrib.layers.variance_scaling_initializer())
            return output



class featuremap_classifier_bedroom:
    inited = False
    c_id = 0

    def __init__(self, id = 0, n_cluster = 20):
        self.inited = False
        self.c_id = id      
        self.n_cluster = n_cluster    
    
    def forward(self, input, training = True):
        with tf.variable_scope('classifier_{}'.format(self.c_id), reuse = self.inited):
            self.inited = True

            #with tf.variable_scope('fc0'):
            #    output = tf.layers.dense(input, 512, kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), activation = tf.nn.leaky_relu)

            with tf.variable_scope('fc1'):
                output = tf.layers.dense(input, 50, kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), activation = tf.nn.leaky_relu)

            with tf.variable_scope('fc2'):
                output = tf.layers.dense(output, self.n_cluster, kernel_initializer = tf.contrib.layers.variance_scaling_initializer())
            return output