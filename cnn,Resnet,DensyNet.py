import tensorflow as tf
import numpy as np
import os
import struct
from read_Intel_Image_Classification import get_data
def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path+'/mnist/','%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path+'/mnist/','%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath,dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",imgpath.read(16))
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)
        images = ((images / 255.) - .5) * 2
    return images, labels
X_train,y_train = load_mnist(os.getcwd(), kind='train')
X_train.shape
type(y_train)
X_test , y_test = load_mnist(os.getcwd(),kind='t10k')
mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)
X_train_centered = (X_train - mean_vals)/std_val
X_test_centered = (X_test - mean_vals)/std_val

#############  Image data  ###############
############# train test valid 14000/3000/7000 ####################


X_train,y_train = get_data('train' ,10000)
X_test,y_test = get_data('test' ,2000)
X_train.shape
y_train.shape
mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)
X_train_centered = (X_train - mean_vals)/std_val
X_test_centered = (X_test - mean_vals)/std_val
del X_train,X_test





def init_weight( shape ,name):
    return tf.Variable(tf.random_normal(shape, stddev=0.01) ,name = name)


def init_biases( shape ,name):
    return tf.Variable(tf.zeros(shape) ,name = name)


class CNN(object):
    def __init__(self,X_train,y_train,X_test,y_test,random_seed=None,epochs = 20,batch_size = 128,learning_rate= 0.0001):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.random_seed = random_seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        np.random.seed(random_seed)
        g = tf.Graph()
        with g.as_default():
            self.Weight = {
                'wconv1': init_weight([5, 5, 1, 32]),  # [filter_shape[0], filter_shape[1], filter_shape[2], chanel_num]
                'bconv1': init_biases([32]),
                'wconv2': init_weight([5, 5, 32, 64]),
                'bconv2': init_biases([64]),
                #'wconv3': init_weight([3, 3, 64, 128]),
                #'bconv3': init_biases([128]),
                'fc_1': 1024,
                'fc_2': 2048,
                'output': 10
            }
            ## set random-seed:
            tf.set_random_seed(random_seed)
            ## build the network:
            self.build()
            ## initializer
            self.init_op = tf.global_variables_initializer()
        ## create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config ,graph = g)



    def build(self):
                #####input######
        tf_x = tf.placeholder(tf.float32 ,shape = [None,784] ,name = 'input_x')
        tf_y = tf.placeholder(tf.int32 ,shape = [None] ,name = 'input_y')
        x_image = tf.reshape(tf_x ,[-1 ,28 ,28 ,1] ,name = 'x_image')
        y_onehot = tf.one_hot(indices=tf_y, depth=10,dtype=tf.float32)

               ######build model#####

        conv1 = tf.add(tf.nn.conv2d(x_image ,self.Weight["wconv1"] ,strides = [1 ,1 ,1 ,1] ,padding = "SAME") ,self.Weight['bconv1'])
        conv1 = tf.nn.relu(conv1)

        conv2 = tf.add(tf.nn.conv2d(conv1 ,self.Weight["wconv2"] ,strides = [1 ,1 ,1 ,1] ,padding = "SAME") ,self.Weight['bconv2'])
        conv2 = tf.nn.relu(conv2)
        conv2_pool = tf.nn.max_pool(conv2 ,ksize = [1 ,2 ,2 ,1] ,strides = [1 ,2 ,2 ,1] ,padding = 'VALID')

        #conv3 = tf.add(tf.nn.conv2d(conv2_pool ,self.Weight["wconv3"] ,strides = [1, 1, 1, 1] ,padding = "SAME") ,self.Weight['bconv3'])
        #conv3 = tf.nn.relu(conv3)
        #conv3_pool = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


        #flat = tf.contrib.layers.flatten(conv3_pool)
        flat = tf.contrib.layers.flatten(conv2_pool)

        fc3 = tf.layers.dense(flat ,units = self.Weight['fc_1'] ,activation = tf.nn.relu)

        fc4 = tf.layers.dense(fc3 ,units = self.Weight["fc_2"] ,activation = tf.nn.relu)

        output = tf.layers.dense(fc4 ,units = self.Weight['output'] ,activation =None)
        classes = tf.cast(tf.argmax(output ,axis = 1) ,tf.int32 ,name = 'prediction_class'),
        probability = tf.nn.softmax(output ,name = 'softmax_tensor')

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_onehot ,logits = output) ,name = 'cost')
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost ,name = 'train_op')
        accuracy = tf.reduce_mean(tf.cast(tf.equal(classes ,tf_y) ,tf.float32) ,name = 'accuracy')

    def batch_generator(self ,X ,y ,batc_size):
        X_copy = np.array(X)
        y_copy = np.array(y)
        for i in range(0 ,X_copy.shape[0] ,batc_size):
            yield (X_copy[i : i+batc_size ,:],y_copy[i : i+batc_size])

    def train(self):
        self.sess.run(self.init_op)
        self.train_cost = []
        for epoch in range(self.epochs):
            batch = self.batch_generator(self.X_train ,self.y_train ,batc_size = self.batch_size)
            for batch_X ,batch_y in batch:
                feed = {"input_x:0" : batch_X ,"input_y:0" : batch_y}
                _,cost = self.sess.run(["train_op" ,"cost:0"] ,feed_dict = feed)
            print("Epochs: %2d , train cost : %5f"%(epoch+1,cost))
            self.train_cost.append(cost)
        pred_test = []
        for i in range(10):
            batch_test_x = self.X_test[i*1000:(i+1)*1000 ,:]
            batch_test_y = self.y_test[i * 1000:(i + 1) * 1000]
            feed_test = {'input_x:0' : batch_test_x ,'input_y:0' : batch_test_y}
            pred = self.sess.run('accuracy:0' ,feed_dict = feed_test)
            pred_test.append(pred)
        print('Test Accuracy is : %2f'%(np.mean(pred_test)))

    def predict(self ,data ,pred_prob = False):
        feed = {'input_x:0' : data}
        if pred_prob:
            return self.sess.run('softmax_tensor:0' ,feed_dict = feed)
        else:
            return self.sess.run('prediction_class:0' ,feed_dict = feed)

##class ResNet(object):
    def __init__(self,X_train,y_train,X_test,y_test,random_seed=None,epochs = 20,batch_size = 128,learning_rate= 0.0001):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.random_seed = random_seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        np.random.seed(random_seed)
        g = tf.Graph()
        with g.as_default():
            self.Weight = {
                'wconv1': init_weight([5, 5, 1, 32]),  # [filter_shape[0], filter_shape[1], filter_shape[2], chanel_num]
                'bconv1': init_biases([32]),
                'wconv2': init_weight([5, 5, 32, 64]),
                'bconv2': init_biases([64]),
                'wconv3': init_weight([3, 3, 64, 128]),
                'bconv3': init_biases([128]),
                'Residualway': init_weight([1,1,64,32]),
                'fc_1': 1024,
                'fc_2': 2048,
                'output': 10
            }
            ## set random-seed:
            tf.set_random_seed(random_seed)
            ## build the network:
            self.build()
            ## initializer
            self.init_op = tf.global_variables_initializer()
        ## create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config ,graph = g)

    def build(self):
        #####input######
        tf_x = tf.placeholder(tf.float32, shape=[None, 784], name='input_x')
        tf_y = tf.placeholder(tf.int32, shape=[None], name='input_y')
        x_image = tf.reshape(tf_x, [-1, 28, 28, 1], name='x_image')
        y_onehot = tf.one_hot(indices=tf_y, depth=10, dtype=tf.float32)

        ######build model#####

        conv1 = tf.add(tf.nn.conv2d(x_image, self.Weight["wconv1"], strides=[1, 1, 1, 1], padding="SAME"),self.Weight['bconv1'])
        conv1 = tf.nn.relu(conv1)

        conv2 = tf.add(tf.nn.conv2d(conv1, self.Weight["wconv2"], strides=[1, 1, 1, 1], padding="SAME"),self.Weight['bconv2'])
        high_way_gate = tf.nn.conv2d(conv2, self.Weight['Residualway'], strides=[1, 1, 1, 1], padding='SAME')
        conv2 = tf.nn.relu(tf.add(high_way_gate ,conv1))
        conv2_pool = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        ##conv3 = tf.add(tf.nn.conv2d(conv2, self.Weight["wconv3"], strides=[1, 1, 1, 1], padding="SAME"),self.Weight['bconv3'])
        ##conv3 = tf.add(tf.nn.relu(conv3) ,high_way)
        ##conv3_pool = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


        flat = tf.contrib.layers.flatten(conv2_pool)

        fc3 = tf.layers.dense(flat, units=self.Weight['fc_1'], activation=tf.nn.relu)

        fc4 = tf.layers.dense(fc3, units=self.Weight["fc_2"], activation=tf.nn.relu)

        output = tf.layers.dense(fc4, units=self.Weight['output'], activation=None)
        classes = tf.cast(tf.argmax(output, axis=1), tf.int32, name='prediction_class'),
        probability = tf.nn.softmax(output, name='softmax_tensor')

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_onehot, logits=output), name='cost')
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost, name='train_op')
        accuracy = tf.reduce_mean(tf.cast(tf.equal(classes, tf_y), tf.float32), name='accuracy')

    def batch_generator(self ,X ,y ,batc_size):
        X_copy = np.array(X)
        y_copy = np.array(y)
        for i in range(0 ,X_copy.shape[0] ,batc_size):
            yield (X_copy[i : i+batc_size ,:],y_copy[i : i+batc_size])

    def train(self):
        self.sess.run(self.init_op)
        self.train_cost = []
        for epoch in range(self.epochs):
            batch = self.batch_generator(self.X_train ,self.y_train ,batc_size = self.batch_size)
            for batch_X ,batch_y in batch:
                feed = {"input_x:0" : batch_X ,"input_y:0" : batch_y}
                _,cost = self.sess.run(["train_op" ,"cost:0"] ,feed_dict = feed)
            print("Epochs: %2d , train cost : %5f"%(epoch+1,cost))
            self.train_cost.append(cost)
        pred_test = []
        for i in range(10):
            batch_test_x = self.X_test[i*1000:(i+1)*1000 ,:]
            batch_test_y = self.y_test[i * 1000:(i + 1) * 1000]
            feed_test = {'input_x:0' : batch_test_x ,'input_y:0' : batch_test_y}
            pred = self.sess.run('accuracy:0' ,feed_dict = feed_test)
            pred_test.append(pred)
        print('Test Accuracy is : %2f'%(np.mean(pred_test)))

    def predict(self ,data ,pred_prob = False):
        feed = {'input_x:0' : data}
        if pred_prob:
            return self.sess.run('softmax_tensor:0' ,feed_dict = feed)
        else:
            return self.sess.run('prediction_class:0' ,feed_dict = feed)



tf.reset_default_graph()


class CNN(object):
    def __init__(self,X_train,y_train,X_test,y_test,random_seed=None,epochs = 50,batch_size = 128 ,dropout_rate =0.5 ,leraning_rate = 0.005):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.random_seed = random_seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.learning_rate = leraning_rate
        np.random.seed(random_seed)
        g = tf.Graph()
        with g.as_default():
            self.Weight = {
                'wconv1': [6, 6, 1, 32] ,  # [filter_shape[0], filter_shape[1], filter_shape[2], chanel_num]
                'bconv1': [32] ,
                'wconv2': [6, 6, 32, 32] ,
                'bconv2': [32] ,
                'wconv3': [6, 6, 32, 64] ,
                'bconv3': [64] ,
                'wconv4': [6, 6, 64, 64] ,
                'bconv4': [64],
                'fc_1': 512,
                'fc_2': 1024,
                'output': 6
                }
            ## set random-seed:
            tf.set_random_seed(random_seed)
            ## build the network:
            self.build()
            ## initializer
            self.init_op = tf.global_variables_initializer()
        ## create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config ,graph = g)
        writer = tf.summary.FileWriter("TensorBoard/CNN/4Layers", graph = self.sess.graph)



    def build(self):
                #####input######
        tf_x = tf.placeholder(tf.float32 ,shape = [None,self.X_train.shape[1],self.X_train.shape[2]] ,name = 'input_x')
        x_image = tf.reshape(tf_x, [-1 ,self.X_train.shape[1] ,self.X_train.shape[2] ,1], name='x_image')
        tf_y = tf.placeholder(tf.int32 ,shape = [None] ,name = 'input_y')
        y_onehot = tf.one_hot(indices=tf_y, depth=6,dtype=tf.float32)
        is_train = tf.placeholder(tf.bool, shape=(), name='is_train')

               ######build model#####


        with tf.name_scope('conv1'):
            w1 = init_weight(self.Weight['wconv1'] ,'wconv1')
            b1 = init_biases(self.Weight['bconv1'] ,'bconv1')
            conv1 = tf.add(tf.nn.conv2d(x_image ,w1,strides = [1 ,1 ,1 ,1] ,padding = "SAME") ,b1)
            conv1 = tf.layers.batch_normalization(conv1)
            conv1 = tf.nn.relu(conv1)

        with tf.name_scope('conv2'):
            w2 = init_weight(self.Weight['wconv2'] ,'wconv2')
            b2 = init_biases(self.Weight['bconv2'] ,'bconv2')
            conv2 = tf.add(tf.nn.conv2d(conv1 ,w2 ,strides = [1 ,1 ,1 ,1] ,padding = "SAME") ,b2)
            conv2 = tf.layers.batch_normalization(conv2)
            conv2 = tf.nn.relu(conv2)
            conv2_pool = tf.nn.max_pool(conv2 ,ksize = [1 ,2 ,2 ,1] ,strides = [1 ,2 ,2 ,1] ,padding = 'VALID')

        with tf.name_scope('conv3'):
            w3 = init_weight(self.Weight['wconv3'],'wconv3')
            b3 = init_biases(self.Weight['bconv3'],'bconv3')
            conv3 = tf.add(tf.nn.conv2d(conv2_pool ,w3 ,strides = [1, 1, 1, 1] ,padding = "SAME") ,b3)
            conv3 = tf.layers.batch_normalization(conv3)
            conv3 = tf.nn.relu(conv3)

        with tf.name_scope('conv4'):
            w4 = init_weight(self.Weight['wconv4'] ,'wconv4')
            b4 = init_biases(self.Weight['bconv4'] ,'bconv4')
            conv4 = tf.add(tf.nn.conv2d(conv3, w4, strides=[1, 1, 1, 1], padding="SAME"),b4)
            conv4 = tf.layers.batch_normalization(conv4)
            conv4 = tf.nn.relu(conv4)
            conv4_pool = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')



        with tf.name_scope('flatten'):
            #flat = tf.contrib.layers.flatten(conv3_pool)
            #flat = tf.contrib.layers.flatten(conv2_pool)
            flat = tf.contrib.layers.flatten(conv4_pool)

        with tf.name_scope('fc3'):
            fc3 = tf.layers.dense(flat ,units = self.Weight['fc_1'] ,activation = tf.nn.relu)
            fc3_drop = tf.layers.dropout(fc3, rate=self.dropout_rate, training=is_train)

        with tf.name_scope('fc4'):
            fc4 = tf.layers.dense(fc3_drop ,units = self.Weight["fc_2"] ,activation = tf.nn.relu)
            fc4_drop = tf.layers.dropout(fc4, rate=self.dropout_rate, training=is_train)

        with tf.name_scope('output'):
            output = tf.layers.dense(fc4_drop ,units = self.Weight['output'] ,activation =None)

        #with tf.name_scope('eval'):
        classes = tf.cast(tf.argmax(output ,axis = 1) ,tf.int32 ,name = 'prediction_class'),
        probability = tf.nn.softmax(output ,name = 'softmax_tensor')
        accuracy = tf.reduce_mean(tf.cast(tf.equal(classes, tf_y), tf.float32), name='accuracy')

        #with tf.name_scope('cost_function'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_onehot ,logits = output) ,name = 'cost')

        #with tf.name_scope('training'):
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        train_op = optimizer.minimize(cost ,name = 'train_op')


    def batch_generator(self ,X ,y ,batc_size ,shuffle = True):
        X_copy = np.array(X)
        y_copy = np.array(y)
        if shuffle:
            X_copy = X_copy.reshape(X.shape[0],X.shape[1]*X.shape[2])
            data = np.column_stack((X_copy, y_copy))  # column bind
            np.random.shuffle(data)
            X_copy = data[:, :-1].reshape(X.shape[0],X.shape[1],X.shape[2])
            y_copy = data[:, -1].astype(int)

        for i in range(0 ,X_copy.shape[0] ,batc_size):
            yield (X_copy[i : i+batc_size ,:],y_copy[i : i+batc_size])

    def train(self):
        self.sess.run(self.init_op)
        self.train_cost = []
        self.test_acc = []
        for epoch in range(self.epochs):
            batch = self.batch_generator(self.X_train ,self.y_train ,batc_size = self.batch_size)
            mean_cost = 0
            n = 0
            for batch_X ,batch_y in batch:
                feed = {"input_x:0" : batch_X ,"input_y:0" : batch_y,'is_train:0': True}
                _,cost = self.sess.run(["train_op" ,"cost:0"] ,feed_dict = feed)
                mean_cost += cost
                n += 1
            mean_cost = mean_cost/n
            print("Epochs: %2d , train cost : %5f"%(epoch+1,mean_cost))
            self.train_cost.append(mean_cost)
            pred_test = []
            for i in range(40):
                batch_test_x = self.X_test[i*50:(i+1)*50 ,:]
                batch_test_y = self.y_test[i * 50:(i + 1) * 50]
                feed_test = {'input_x:0' : batch_test_x ,'input_y:0' : batch_test_y,'is_train:0': False}
                pred = self.sess.run('accuracy:0' ,feed_dict = feed_test)
                pred_test.append(pred)
            print('Test Accuracy is : %2f'%(np.mean(pred_test)))
            self.test_acc.append(np.mean(pred_test))

    def predict(self ,data ,pred_prob = False):
        feed = {'input_x:0' : data ,'is_train:0': False}
        if pred_prob:
            return self.sess.run('softmax_tensor:0' ,feed_dict = feed)
        else:
            return self.sess.run('prediction_class:0' ,feed_dict = feed)

class CNN_2(object):
    def __init__(self,X_train,y_train,X_test,y_test,random_seed=None,epochs = 50,batch_size = 128 ,dropout_rate =0.5,leraning_rate = 0.005):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.random_seed = random_seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.learning_rate = leraning_rate
        np.random.seed(random_seed)
        g = tf.Graph()
        with g.as_default():
            self.Weight = {
                'wconv1': [6, 6, 1, 32] ,  # [filter_shape[0], filter_shape[1], filter_shape[2], chanel_num]
                'bconv1': [32] ,
                'wconv2': [6, 6, 32, 32] ,
                'bconv2': [32] ,
                'wconv3': [6, 6, 32, 32] ,
                'bconv3': [32] ,
                'wconv4': [6, 6, 32, 64] ,
                'bconv4': [64],
                'wconv5': [6, 6, 64, 64],
                'bconv5': [64],
                'wconv6': [6, 6, 64, 64],
                'bconv6': [64],
                'fc_1': 512,
                'fc_2': 1024,
                'output': 6
                }
            ## set random-seed:
            tf.set_random_seed(random_seed)
            ## build the network:
            self.build()
            ## initializer
            self.init_op = tf.global_variables_initializer()
        ## create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config ,graph = g)
        writer = tf.summary.FileWriter("TensorBoard/CNN/6Layers", graph = self.sess.graph)



    def build(self):
                #####input######
        tf_x = tf.placeholder(tf.float32 ,shape = [None,self.X_train.shape[1],self.X_train.shape[2]] ,name = 'input_x')
        x_image = tf.reshape(tf_x, [-1 ,self.X_train.shape[1] ,self.X_train.shape[2] ,1], name='x_image')
        tf_y = tf.placeholder(tf.int32 ,shape = [None] ,name = 'input_y')
        y_onehot = tf.one_hot(indices=tf_y, depth=6,dtype=tf.float32)
        is_train = tf.placeholder(tf.bool, shape=(), name='is_train')

               ######build model#####


        with tf.name_scope('conv1'):
            w1 = init_weight(self.Weight['wconv1'] ,'wconv1')
            b1 = init_biases(self.Weight['bconv1'] ,'bconv1')
            conv1 = tf.add(tf.nn.conv2d(x_image ,w1,strides = [1 ,1 ,1 ,1] ,padding = "SAME") ,b1)
            conv1 = tf.layers.batch_normalization(conv1)
            conv1 = tf.nn.relu(conv1)

        with tf.name_scope('conv2'):
            w2 = init_weight(self.Weight['wconv2'] ,'wconv2')
            b2 = init_biases(self.Weight['bconv2'] ,'bconv2')
            conv2 = tf.add(tf.nn.conv2d(conv1 ,w2 ,strides = [1 ,1 ,1 ,1] ,padding = "SAME") ,b2)
            conv2 = tf.layers.batch_normalization(conv2)
            conv2 = tf.nn.relu(conv2)
            conv2_pool = tf.nn.max_pool(conv2 ,ksize = [1 ,2 ,2 ,1] ,strides = [1 ,2 ,2 ,1] ,padding = 'VALID')

        with tf.name_scope('conv3'):
            w3 = init_weight(self.Weight['wconv3'],'wconv3')
            b3 = init_biases(self.Weight['bconv3'],'bconv3')
            conv3 = tf.add(tf.nn.conv2d(conv2_pool ,w3 ,strides = [1, 1, 1, 1] ,padding = "SAME") ,b3)
            conv3 = tf.layers.batch_normalization(conv3)
            conv3 = tf.nn.relu(conv3)

        with tf.name_scope('conv4'):
            w4 = init_weight(self.Weight['wconv4'] ,'wconv4')
            b4 = init_biases(self.Weight['bconv4'] ,'bconv4')
            conv4 = tf.add(tf.nn.conv2d(conv3, w4, strides=[1, 1, 1, 1], padding="SAME"),b4)
            conv4 = tf.layers.batch_normalization(conv4)
            conv4 = tf.nn.relu(conv4)
            conv4_pool = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        with tf.name_scope('conv5'):
            w5 = init_weight(self.Weight['wconv5'], 'wconv5')
            b5 = init_biases(self.Weight['bconv5'], 'bconv5')
            conv5 = tf.add(tf.nn.conv2d(conv4_pool, w5, strides=[1, 1, 1, 1], padding="SAME"), b5)
            conv5 = tf.layers.batch_normalization(conv5)
            conv5 = tf.nn.relu(conv5)

        with tf.name_scope('conv6'):
            w6 = init_weight(self.Weight['wconv6'], 'wconv6')
            b6 = init_biases(self.Weight['bconv6'], 'bconv6')
            conv6 = tf.add(tf.nn.conv2d(conv5, w6, strides=[1, 1, 1, 1], padding="SAME"), b6)
            conv6 = tf.layers.batch_normalization(conv6)
            conv6 = tf.nn.relu(conv6)
            conv6_pool = tf.nn.max_pool(conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        with tf.name_scope('flatten'):
            #flat = tf.contrib.layers.flatten(conv3_pool)
            #flat = tf.contrib.layers.flatten(conv2_pool)
            flat = tf.contrib.layers.flatten(conv6_pool)

        with tf.name_scope('fc3'):
            fc3 = tf.layers.dense(flat ,units = self.Weight['fc_1'] ,activation = tf.nn.relu)
            fc3_drop = tf.layers.dropout(fc3, rate=self.dropout_rate, training=is_train)

        with tf.name_scope('fc4'):
            fc4 = tf.layers.dense(fc3_drop ,units = self.Weight["fc_2"] ,activation = tf.nn.relu)
            fc4_drop = tf.layers.dropout(fc4, rate=self.dropout_rate, training=is_train)

        with tf.name_scope('output'):
            output = tf.layers.dense(fc4_drop ,units = self.Weight['output'] ,activation =None)

        #with tf.name_scope('eval'):
        classes = tf.cast(tf.argmax(output ,axis = 1) ,tf.int32 ,name = 'prediction_class')
        probability = tf.nn.softmax(output ,name = 'softmax_tensor')
        accuracy = tf.reduce_mean(tf.cast(tf.equal(classes, tf_y), tf.float32), name='accuracy')

        #with tf.name_scope('cost_function'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_onehot ,logits = output) ,name = 'cost')

        #with tf.name_scope('training'):
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        train_op = optimizer.minimize(cost ,name = 'train_op')


    def batch_generator(self ,X ,y ,batc_size ,shuffle = True):
        X_copy = np.array(X)
        y_copy = np.array(y)
        if shuffle:
            X_copy = X_copy.reshape(X.shape[0],X.shape[1]*X.shape[2])
            data = np.column_stack((X_copy, y_copy))  # column bind
            np.random.shuffle(data)
            X_copy = data[:, :-1].reshape(X.shape[0],X.shape[1],X.shape[2])
            y_copy = data[:, -1].astype(int)

        for i in range(0 ,X_copy.shape[0] ,batc_size):
            yield (X_copy[i : i+batc_size ,:],y_copy[i : i+batc_size])

    def train(self):
        self.sess.run(self.init_op)
        self.train_cost = []
        self.test_acc = []
        for epoch in range(self.epochs):
            batch = self.batch_generator(self.X_train ,self.y_train ,batc_size = self.batch_size)
            mean_cost = 0
            n = 0
            for batch_X ,batch_y in batch:
                feed = {"input_x:0" : batch_X ,"input_y:0" : batch_y,'is_train:0': True}
                _,cost = self.sess.run(["train_op" ,"cost:0"] ,feed_dict = feed)
                mean_cost += cost
                n += 1
            mean_cost = mean_cost/n
            print("Epochs: %2d , train cost : %5f"%(epoch+1,mean_cost))
            self.train_cost.append(mean_cost)
            pred_test = []
            for i in range(40):
                batch_test_x = self.X_test[i*50:(i+1)*50 ,:]
                batch_test_y = self.y_test[i * 50:(i + 1) * 50]
                feed_test = {'input_x:0' : batch_test_x ,'input_y:0' : batch_test_y,'is_train:0': False}
                pred = self.sess.run('accuracy:0' ,feed_dict = feed_test)
                pred_test.append(pred)
            print('Test Accuracy is : %2f'%(np.mean(pred_test)))
            self.test_acc.append(np.mean(pred_test))

    def predict(self ,data ,pred_prob = False):
        feed = {'input_x:0' : data ,'is_train:0': False}
        if pred_prob:
            return self.sess.run('softmax_tensor:0' ,feed_dict = feed)
        else:
            return self.sess.run('prediction_class:0' ,feed_dict = feed)

model_1 = CNN(X_train = X_train_centered ,y_train = y_train ,X_test =X_test_centered ,y_test = y_test)
model_1.train()

model_12 = CNN_2(X_train = X_train_centered ,y_train = y_train ,X_test =X_test_centered ,y_test = y_test)
model_12.train()


class ResNet(object):
    def __init__(self,X_train,y_train,X_test,y_test,random_seed=None,epochs = 50,batch_size = 128 ,dropout_rate =0.5,leraning_rate = 0.005):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.random_seed = random_seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.learning_rate = leraning_rate
        np.random.seed(random_seed)
        g = tf.Graph()
        with g.as_default():
            self.Weight = {
                'wconv1': [3, 3, 1, 32] ,  # [filter_shape[0], filter_shape[1], filter_shape[2], chanel_num]
                'bconv1': [32],
                'wconv2': [3, 3, 32, 32] ,
                'bconv2': [32] ,
                ##  POOL　##
                'wconv3': [6, 6, 32, 64] ,
                'bconv3': [64] ,
                'wconv4': [6, 6, 64, 64] ,
                'bconv4': [64],
                'Identitymap_1': [1, 1, 1, 32],
                'Identitymap_2': [1, 1, 32, 64],
                'fc_1': 512,
                'fc_2': 1024,
                'output': 6
            }
            ## set random-seed:
            tf.set_random_seed(random_seed)
            ## build the network:
            self.build()
            ## initializer
            self.init_op = tf.global_variables_initializer()
        ## create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config ,graph = g)
        tf.summary.merge_all()
        writer = tf.summary.FileWriter("TensorBoard/ResNet/4Layers", graph = self.sess.graph)



    def build(self):
                #####input######
        tf_x = tf.placeholder(tf.float32 ,shape = [None,self.X_train.shape[1],self.X_train.shape[2]] ,name = 'input_x')
        x_image = tf.reshape(tf_x, [-1 ,self.X_train.shape[1] ,self.X_train.shape[2] ,1], name='x_image')
        tf_y = tf.placeholder(tf.int32 ,shape = [None] ,name = 'input_y')
        y_onehot = tf.one_hot(indices=tf_y, depth=6,dtype=tf.float32)
        is_train = tf.placeholder(tf.bool, shape=(), name='is_train')

               ######build model#####
        with tf.name_scope('conv1'):
            w1 = init_weight(self.Weight['wconv1'], 'wconv1')
            b1 = init_biases(self.Weight['bconv1'], 'bconv1')
            conv1 = tf.add(tf.nn.conv2d(x_image ,w1 ,strides = [1 ,1 ,1 ,1] ,padding = "SAME") ,b1)
            bn1 = tf.layers.batch_normalization(conv1)

        with tf.name_scope('resnet1'):
            r1 = init_weight(self.Weight['Identitymap_1'], 'Identitymap_1')
            map_1 = tf.nn.conv2d(x_image, r1, strides=[1, 1, 1, 1], padding='SAME')
            conv1_re = tf.nn.relu(tf.add(map_1, bn1))  # Residual add

        with tf.name_scope('conv2'):
            w2 = init_weight(self.Weight['wconv2'], 'wconv2')
            b2 = init_biases(self.Weight['bconv2'], 'bconv2')
            conv2 = tf.add(tf.nn.conv2d(conv1_re ,w2 ,strides = [1 ,1 ,1 ,1] ,padding = "SAME") ,b2)
            bn2 = tf.layers.batch_normalization(conv2)

        with tf.name_scope('resnet2'):
            conv2_re = tf.nn.relu(tf.add(bn2 ,conv1_re)) #Residual add

        with tf.name_scope('pool1'):
            conv2_pool = tf.nn.max_pool(conv2_re ,ksize = [1 ,2 ,2 ,1] ,strides = [1 ,2 ,2 ,1] ,padding = 'VALID')

        with tf.name_scope('conv3'):
            w3 = init_weight(self.Weight['wconv3'], 'wconv3')
            b3 = init_biases(self.Weight['bconv3'], 'bconv3')
            conv3 = tf.add(tf.nn.conv2d(conv2_pool ,w3,strides = [1, 1, 1, 1] ,padding = "SAME") ,b3)
            bn3 = tf.layers.batch_normalization(conv3)

        with tf.name_scope('resnet3'):
            r3 = init_weight(self.Weight['Identitymap_2'], 'Identitymap_2')
            map_3 = tf.nn.conv2d(conv2_pool, r3, strides=[1, 1, 1, 1], padding='SAME')
            conv3_re = tf.nn.relu(tf.add(map_3, bn3))  # Residual add

        with tf.name_scope('conv4'):
            w4 = init_weight(self.Weight['wconv4'], 'wconv4')
            b4 = init_biases(self.Weight['bconv4'], 'bconv4')
            conv4 = tf.add(tf.nn.conv2d(conv3_re, w4, strides=[1, 1, 1, 1], padding="SAME"),b4)
            bn4 = tf.layers.batch_normalization(conv4)
        with tf.name_scope('resnet4'):
            conv4_re = tf.nn.relu(tf.add(bn4, conv3_re))

        with tf.name_scope('pool2'):
            conv4_pool = tf.nn.max_pool(conv4_re, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')



        with tf.name_scope('flatten'):
            #flat = tf.contrib.layers.flatten(conv3_pool)
            #flat = tf.contrib.layers.flatten(conv2_pool)
            flat = tf.contrib.layers.flatten(conv4_pool)

        with tf.name_scope('fc3'):
            fc3 = tf.layers.dense(flat ,units = self.Weight['fc_1'] ,activation = tf.nn.relu)
            fc3_drop = tf.layers.dropout(fc3, rate=self.dropout_rate, training=is_train)

        with tf.name_scope('fc4'):
            fc4 = tf.layers.dense(fc3_drop ,units = self.Weight["fc_2"] ,activation = tf.nn.relu)
            fc4_drop = tf.layers.dropout(fc4, rate=self.dropout_rate, training=is_train)

        with tf.name_scope('output'):
            output = tf.layers.dense(fc4_drop ,units = self.Weight['output'] ,activation =None)


        classes = tf.cast(tf.argmax(output ,axis = 1) ,tf.int32 ,name = 'prediction_class'),
        probability = tf.nn.softmax(output ,name = 'softmax_tensor')
        accuracy = tf.reduce_mean(tf.cast(tf.equal(classes, tf_y), tf.float32), name='accuracy')

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_onehot ,logits = output) ,name = 'cost')

        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost ,name = 'train_op')

    def batch_generator(self ,X ,y ,batc_size ,shuffle = True):
        X_copy = np.array(X)
        y_copy = np.array(y)
        if shuffle:
            X_copy = X_copy.reshape(X.shape[0],X.shape[1]*X.shape[2])
            data = np.column_stack((X_copy, y_copy))  # column bind
            np.random.shuffle(data)
            X_copy = data[:, :-1].reshape(X.shape[0],X.shape[1],X.shape[2])
            y_copy = data[:, -1].astype(int)

        for i in range(0 ,X_copy.shape[0] ,batc_size):
            yield (X_copy[i : i+batc_size ,:],y_copy[i : i+batc_size])

    def train(self):
        self.sess.run(self.init_op)
        self.train_cost = []
        self.test_acc = []
        for epoch in range(self.epochs):
            batch = self.batch_generator(self.X_train ,self.y_train ,batc_size = self.batch_size)
            mean_cost = 0
            n = 0
            for batch_X ,batch_y in batch:
                feed = {"input_x:0" : batch_X ,"input_y:0" : batch_y,'is_train:0': True}
                _,cost = self.sess.run(["train_op" ,"cost:0"] ,feed_dict = feed)
                mean_cost += cost
                n += 1
            mean_cost = mean_cost/n
            print("Epochs: %2d , train cost : %5f"%(epoch+1,mean_cost))
            self.train_cost.append(mean_cost)
            pred_test = []
            for i in range(40):
                batch_test_x = self.X_test[i*50:(i+1)*50 ,:]
                batch_test_y = self.y_test[i * 50:(i + 1) * 50]
                feed_test = {'input_x:0' : batch_test_x ,'input_y:0' : batch_test_y,'is_train:0': False}
                pred = self.sess.run('accuracy:0' ,feed_dict = feed_test)
                pred_test.append(pred)
            print('Test Accuracy is : %2f'%(np.mean(pred_test)))
            self.test_acc.append(np.mean(pred_test))

    def predict(self ,data ,pred_prob = False):
        feed = {'input_x:0' : data,'is_train:0': False}
        if pred_prob:
            return self.sess.run('softmax_tensor:0' ,feed_dict = feed)
        else:
            return self.sess.run('prediction_class:0' ,feed_dict = feed)

class ResNet_2(object):
    def __init__(self,X_train,y_train,X_test,y_test,random_seed=None,epochs = 50,batch_size = 128 ,dropout_rate =0.5,leraning_rate = 0.005):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.random_seed = random_seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.learning_rate = leraning_rate
        np.random.seed(random_seed)
        g = tf.Graph()
        with g.as_default():
            self.Weight = {
                'wconv1': [3, 3, 1, 32] ,  # [filter_shape[0], filter_shape[1], filter_shape[2], chanel_num]
                'bconv1': [32] ,
                'wconv2': [3, 3, 32, 32] ,
                'bconv2': [32] ,
                'wconv3': [3, 3, 32, 32] ,
                'bconv3': [32] ,
                ########## Pooling #########
                'wconv4': [6, 6, 32, 64] ,
                'bconv4': [64],
                'wconv5': [6, 6, 64, 64],
                'bconv5': [64],
                'wconv6': [6, 6, 64, 64],
                'bconv6': [64],
                'Identitymap_1': [1, 1, 1, 32],
                'Identitymap_2': [1, 1, 32, 64],
                'fc_1': 512,
                'fc_2': 1024,
                'output': 6
            }
            ## set random-seed:
            tf.set_random_seed(random_seed)
            ## build the network:
            self.build()
            ## initializer
            self.init_op = tf.global_variables_initializer()
        ## create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config ,graph = g)
        tf.summary.merge_all()
        writer = tf.summary.FileWriter("TensorBoard/ResNet/6Layers", graph = self.sess.graph)



    def build(self):
                #####input######
        tf_x = tf.placeholder(tf.float32 ,shape = [None,self.X_train.shape[1],self.X_train.shape[2]] ,name = 'input_x')
        x_image = tf.reshape(tf_x, [-1 ,self.X_train.shape[1] ,self.X_train.shape[2] ,1], name='x_image')
        tf_y = tf.placeholder(tf.int32 ,shape = [None] ,name = 'input_y')
        y_onehot = tf.one_hot(indices=tf_y, depth=6,dtype=tf.float32)
        is_train = tf.placeholder(tf.bool, shape=(), name='is_train')

               ######build model#####
        with tf.name_scope('conv1'):
            w1 = init_weight(self.Weight['wconv1'], 'wconv1')
            b1 = init_biases(self.Weight['bconv1'], 'bconv1')
            conv1 = tf.add(tf.nn.conv2d(x_image ,w1 ,strides = [1 ,1 ,1 ,1] ,padding = "SAME") ,b1)
            bn1 = tf.layers.batch_normalization(conv1)


        with tf.name_scope('resnet1'):
            r1 = init_weight(self.Weight['Identitymap_1'], 'Identitymap_1')
            map_1 = tf.nn.conv2d(x_image, r1, strides=[1, 1, 1, 1], padding='SAME')
            conv1_re = tf.nn.relu(tf.add(map_1, bn1))  # Residual add

        with tf.name_scope('conv2'):
            w2 = init_weight(self.Weight['wconv2'], 'wconv2')
            b2 = init_biases(self.Weight['bconv2'], 'bconv2')
            conv2 = tf.add(tf.nn.conv2d(conv1_re ,w2 ,strides = [1 ,1 ,1 ,1] ,padding = "SAME") ,b2)
            bn2 = tf.layers.batch_normalization(conv2)

        with tf.name_scope('resnet2'):
            conv2_re = tf.nn.relu(tf.add(conv1_re, bn2))  # Residual add

        with tf.name_scope('conv3'):
            w3 = init_weight(self.Weight['wconv3'], 'wconv3')
            b3 = init_biases(self.Weight['bconv3'], 'bconv3')
            conv3 = tf.add(tf.nn.conv2d(conv2_re ,w3,strides = [1, 1, 1, 1] ,padding = "SAME") ,b3)
            bn3 = tf.layers.batch_normalization(conv3)

        with tf.name_scope('resnet3'):
            conv3_re = tf.nn.relu(tf.add(conv2_re, bn3))  # Residual add

        with tf.name_scope('pool1'):
            conv3_pool = tf.nn.max_pool(conv3_re, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        with tf.name_scope('conv4'):
            w4 = init_weight(self.Weight['wconv4'], 'wconv4')
            b4 = init_biases(self.Weight['bconv4'], 'bconv4')
            conv4 = tf.add(tf.nn.conv2d(conv3_pool, w4, strides=[1, 1, 1, 1], padding="SAME"),b4)
            bn4 = tf.layers.batch_normalization(conv4)

        with tf.name_scope('resnet4'):
            r4 = init_weight(self.Weight['Identitymap_2'], 'Identitymap_2')
            map_4 = tf.nn.conv2d(conv3_pool, r4, strides=[1, 1, 1, 1], padding='SAME')
            conv4_re = tf.nn.relu(tf.add(map_4, bn4))  # Residual add

        with tf.name_scope('conv5'):
            w5 = init_weight(self.Weight['wconv5'], 'wconv5')
            b5 = init_biases(self.Weight['bconv5'], 'bconv5')
            conv5 = tf.add(tf.nn.conv2d(conv4_re, w5, strides=[1, 1, 1, 1], padding="SAME"), b5)
            bn5 = tf.layers.batch_normalization(conv5)

        with tf.name_scope('resnet5'):
            conv5_re = tf.nn.relu(tf.add(conv4_re, bn5))  # Residual add

        with tf.name_scope('conv6'):
            w6 = init_weight(self.Weight['wconv6'], 'wconv6')
            b6 = init_biases(self.Weight['bconv6'], 'bconv6')
            conv6 = tf.add(tf.nn.conv2d(conv5_re, w6, strides=[1, 1, 1, 1], padding="SAME"), b6)
            bn6 = tf.layers.batch_normalization(conv6)

        with tf.name_scope('resnet6'):
            conv6_re = tf.nn.relu(tf.add(conv5_re, bn6))

        with tf.name_scope('pool2'):
            conv6_pool = tf.nn.max_pool(conv6_re, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


        with tf.name_scope('flatten'):
            #flat = tf.contrib.layers.flatten(conv3_pool)
            #flat = tf.contrib.layers.flatten(conv2_pool)
            flat = tf.contrib.layers.flatten(conv6_pool)

        with tf.name_scope('fc3'):
            fc3 = tf.layers.dense(flat ,units = self.Weight['fc_1'] ,activation = tf.nn.relu)
            fc3_drop = tf.layers.dropout(fc3, rate=self.dropout_rate, training=is_train)

        with tf.name_scope('fc4'):
            fc4 = tf.layers.dense(fc3_drop ,units = self.Weight["fc_2"] ,activation = tf.nn.relu)
            fc4_drop = tf.layers.dropout(fc4, rate=self.dropout_rate, training=is_train)

        with tf.name_scope('output'):
            output = tf.layers.dense(fc4_drop ,units = self.Weight['output'] ,activation =None)


        classes = tf.cast(tf.argmax(output ,axis = 1) ,tf.int32 ,name = 'prediction_class'),
        probability = tf.nn.softmax(output ,name = 'softmax_tensor')
        accuracy = tf.reduce_mean(tf.cast(tf.equal(classes, tf_y), tf.float32), name='accuracy')

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_onehot ,logits = output) ,name = 'cost')

        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost ,name = 'train_op')

    def batch_generator(self ,X ,y ,batc_size ,shuffle = True):
        X_copy = np.array(X)
        y_copy = np.array(y)
        if shuffle:
            X_copy = X_copy.reshape(X.shape[0],X.shape[1]*X.shape[2])
            data = np.column_stack((X_copy, y_copy))  # column bind
            np.random.shuffle(data)
            X_copy = data[:, :-1].reshape(X.shape[0],X.shape[1],X.shape[2])
            y_copy = data[:, -1].astype(int)

        for i in range(0 ,X_copy.shape[0] ,batc_size):
            yield (X_copy[i : i+batc_size ,:],y_copy[i : i+batc_size])

    def train(self):
        self.sess.run(self.init_op)
        self.train_cost = []
        self.test_acc = []
        for epoch in range(self.epochs):
            batch = self.batch_generator(self.X_train ,self.y_train ,batc_size = self.batch_size)
            mean_cost = 0
            n = 0
            for batch_X ,batch_y in batch:
                feed = {"input_x:0" : batch_X ,"input_y:0" : batch_y,'is_train:0': True}
                _,cost = self.sess.run(["train_op" ,"cost:0"] ,feed_dict = feed)
                mean_cost += cost
                n += 1
            mean_cost = mean_cost/n
            print("Epochs: %2d , train cost : %5f"%(epoch+1,mean_cost))
            self.train_cost.append(mean_cost)
            pred_test = []
            for i in range(40):
                batch_test_x = self.X_test[i*50:(i+1)*50 ,:]
                batch_test_y = self.y_test[i * 50:(i + 1) * 50]
                feed_test = {'input_x:0' : batch_test_x ,'input_y:0' : batch_test_y,'is_train:0': False}
                pred = self.sess.run('accuracy:0' ,feed_dict = feed_test)
                pred_test.append(pred)
            print('Test Accuracy is : %2f'%(np.mean(pred_test)))
            self.test_acc.append(np.mean(pred_test))

    def predict(self ,data ,pred_prob = False):
        feed = {'input_x:0' : data,'is_train:0': False}
        if pred_prob:
            return self.sess.run('softmax_tensor:0' ,feed_dict = feed)
        else:
            return self.sess.run('prediction_class:0' ,feed_dict = feed)

model_2 = ResNet(X_train = X_train_centered ,y_train = y_train ,X_test =X_test_centered ,y_test = y_test)
model_2.train()

model_22 = ResNet_2(X_train = X_train_centered ,y_train = y_train ,X_test =X_test_centered ,y_test = y_test)
model_22.train()

class DenseNet(object):
    def __init__(self,X_train,y_train,X_test,y_test,random_seed=None,epochs = 50,batch_size = 128,dropout_rate =0.5,leraning_rate = 0.005):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.random_seed = random_seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.learning_rate = leraning_rate
        np.random.seed(random_seed)
        g = tf.Graph()
        with g.as_default():
            self.Weight = {
                'wconv1': [3, 3, 1, 32] ,  # [filter_shape[0], filter_shape[1], filter_shape[2], chanel_num]
                'bconv1': [32] ,
                'wconv2': [3, 3, 33, 32] ,
                'bconv2': [32] ,
                #### Pool #####
                'wconv3': [6, 6, 32, 64] ,
                'bconv3': [64] ,
                'wconv4': [6, 6, 96, 64] ,
                'bconv4': [64],
                'fc_1': 512,
                'fc_2': 1024,
                'output': 6
                }
            ## set random-seed:
            tf.set_random_seed(random_seed)
            ## build the network:
            self.build()
            ## initializer
            self.init_op = tf.global_variables_initializer()
        ## create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config ,graph = g)
        writer = tf.summary.FileWriter("TensorBoard/Densenet/4Layers", graph = self.sess.graph)



    def build(self):
                #####input######
        tf_x = tf.placeholder(tf.float32 ,shape = [None,self.X_train.shape[1],self.X_train.shape[2]] ,name = 'input_x')
        x_image = tf.reshape(tf_x, [-1 ,self.X_train.shape[1] ,self.X_train.shape[2] ,1], name='x_image')
        tf_y = tf.placeholder(tf.int32 ,shape = [None] ,name = 'input_y')
        y_onehot = tf.one_hot(indices=tf_y, depth=6,dtype=tf.float32)
        is_train = tf.placeholder(tf.bool, shape=(), name='is_train')

               ######build model#####
        with tf.name_scope('conv1'):
            w1 = init_weight(self.Weight['wconv1'], 'wconv1')
            b1 = init_biases(self.Weight['bconv1'], 'bconv1')
            conv1 = tf.add(tf.nn.conv2d(x_image ,w1 ,strides = [1 ,1 ,1 ,1] ,padding = "SAME") ,b1)
            bn1 = tf.layers.batch_normalization(conv1)
            conv1_relu = tf.nn.relu(bn1)

        with tf.name_scope('concat1'):
            conv1_dense = tf.concat([conv1_relu, x_image], 3)

        with tf.name_scope('conv2'):
            w2 = init_weight(self.Weight['wconv2'], 'wconv2')
            b2 = init_biases(self.Weight['bconv2'], 'bconv2')
            conv2 = tf.add(tf.nn.conv2d(conv1_dense ,w2 ,strides = [1 ,1 ,1 ,1] ,padding = "SAME") ,b2)
            bn2 = tf.layers.batch_normalization(conv2)
            conv2_relu = tf.nn.relu(bn2)


        with tf.name_scope('pool1'):
            conv2_pool = tf.nn.max_pool(conv2_relu ,ksize = [1 ,2 ,2 ,1] ,strides = [1 ,2 ,2 ,1] ,padding = 'VALID')

        with tf.name_scope('conv3'):
            w3 = init_weight(self.Weight['wconv3'], 'wconv3')
            b3 = init_biases(self.Weight['bconv3'], 'bconv3')
            conv3 = tf.add(tf.nn.conv2d(conv2_pool ,w3 ,strides = [1, 1, 1, 1] ,padding = "SAME") ,b3)
            bn3 = tf.layers.batch_normalization(conv3)
            conv3_relu = tf.nn.relu(bn3)

        with tf.name_scope('concat3'):
            conv3_dense = tf.concat([conv3_relu ,conv2_pool],3)

        with tf.name_scope('conv4'):
            w4 = init_weight(self.Weight['wconv4'], 'wconv4')
            b4 = init_biases(self.Weight['bconv4'], 'bconv4')
            conv4 = tf.add(tf.nn.conv2d(conv3_dense, w4, strides=[1, 1, 1, 1], padding="SAME"),b4)
            bn4 = tf.layers.batch_normalization(conv4)
            conv4_relu = tf.nn.relu(bn4)


        with tf.name_scope('pool2'):
            conv4_pool = tf.nn.max_pool(conv4_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


        with tf.name_scope('flatten'):
            #flat = tf.contrib.layers.flatten(conv3_pool)
            #flat = tf.contrib.layers.flatten(conv2_pool)
            flat = tf.contrib.layers.flatten(conv4_pool)

        with tf.name_scope('fc3'):
            fc3 = tf.layers.dense(flat ,units = self.Weight['fc_1'] ,activation = tf.nn.relu)
            fc3_drop = tf.layers.dropout(fc3, rate=self.dropout_rate, training=is_train)

        with tf.name_scope('fc4'):
            fc4 = tf.layers.dense(fc3_drop ,units = self.Weight["fc_2"] ,activation = tf.nn.relu)
            fc4_drop = tf.layers.dropout(fc4, rate=self.dropout_rate, training=is_train)

        with tf.name_scope('output'):
            output = tf.layers.dense(fc4_drop ,units = self.Weight['output'] ,activation =None)


        classes = tf.cast(tf.argmax(output ,axis = 1) ,tf.int32 ,name = 'prediction_class'),
        probability = tf.nn.softmax(output ,name = 'softmax_tensor')
        accuracy = tf.reduce_mean(tf.cast(tf.equal(classes, tf_y), tf.float32), name='accuracy')


        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_onehot ,logits = output) ,name = 'cost')


        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost ,name = 'train_op')


    def batch_generator(self ,X ,y ,batc_size ,shuffle = True):
        X_copy = np.array(X)
        y_copy = np.array(y)
        if shuffle:
            X_copy = X_copy.reshape(X.shape[0],X.shape[1]*X.shape[2])
            data = np.column_stack((X_copy, y_copy))  # column bind
            np.random.shuffle(data)
            X_copy = data[:, :-1].reshape(X.shape[0],X.shape[1],X.shape[2])
            y_copy = data[:, -1].astype(int)

        for i in range(0 ,X_copy.shape[0] ,batc_size):
            yield (X_copy[i : i+batc_size ,:],y_copy[i : i+batc_size])

    def train(self):
        self.sess.run(self.init_op)
        self.train_cost = []
        self.test_acc = []
        for epoch in range(self.epochs):
            batch = self.batch_generator(self.X_train, self.y_train, batc_size=self.batch_size)
            mean_cost = 0
            n = 0
            for batch_X, batch_y in batch:
                feed = {"input_x:0": batch_X, "input_y:0": batch_y, 'is_train:0': True}
                _, cost = self.sess.run(["train_op", "cost:0"], feed_dict=feed)
                mean_cost += cost
                n += 1
            mean_cost = mean_cost / n
            print("Epochs: %2d , train cost : %5f" % (epoch + 1, mean_cost))
            self.train_cost.append(mean_cost)
            pred_test = []
            for i in range(40):
                batch_test_x = self.X_test[i * 50:(i + 1) * 50, :]
                batch_test_y = self.y_test[i * 50:(i + 1) * 50]
                feed_test = {'input_x:0': batch_test_x, 'input_y:0': batch_test_y, 'is_train:0': False}
                pred = self.sess.run('accuracy:0', feed_dict=feed_test)
                pred_test.append(pred)
            print('Test Accuracy is : %2f' % (np.mean(pred_test)))
            self.test_acc.append(np.mean(pred_test))

    def predict(self ,data ,pred_prob = False):
        feed = {'input_x:0' : data,'is_train:0': False}
        if pred_prob:
            return self.sess.run('softmax_tensor:0' ,feed_dict = feed)
        else:
            return self.sess.run('prediction_class:0' ,feed_dict = feed)

class DenseNet_2(object):
    def __init__(self,X_train,y_train,X_test,y_test,random_seed=None,epochs = 50,batch_size = 128,dropout_rate =0.5,leraning_rate = 0.005):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.random_seed = random_seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.learning_rate = leraning_rate
        np.random.seed(random_seed)
        g = tf.Graph()
        with g.as_default():
            self.Weight = {
                'wconv1': [3, 3, 1, 32] ,  # [filter_shape[0], filter_shape[1], filter_shape[2], chanel_num]
                'bconv1': [32] ,
                'wconv2': [3, 3, 33, 32] ,
                'bconv2': [32] ,
                'wconv3': [3, 3, 65, 32] ,
                'bconv3': [32] ,
                'wconv4': [6, 6, 32, 64] ,
                'bconv4': [64],
                'wconv5': [5, 5, 96, 64],
                'bconv5': [64],
                'wconv6': [5, 5, 160, 64],
                'bconv6': [64],
                'fc_1': 512,
                'fc_2': 1024,
                'output': 6
                }
            ## set random-seed:
            tf.set_random_seed(random_seed)
            ## build the network:
            self.build()
            ## initializer
            self.init_op = tf.global_variables_initializer()
        ## create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config ,graph = g)
        writer = tf.summary.FileWriter("TensorBoard/Densenet/6Layers", graph = self.sess.graph)



    def build(self):
                #####input######
        tf_x = tf.placeholder(tf.float32 ,shape = [None,self.X_train.shape[1],self.X_train.shape[2]] ,name = 'input_x')
        x_image = tf.reshape(tf_x, [-1 ,self.X_train.shape[1] ,self.X_train.shape[2] ,1], name='x_image')
        tf_y = tf.placeholder(tf.int32 ,shape = [None] ,name = 'input_y')
        y_onehot = tf.one_hot(indices=tf_y, depth=6,dtype=tf.float32)
        is_train = tf.placeholder(tf.bool, shape=(), name='is_train')

               ######build model#####
        with tf.name_scope('conv1'):
            w1 = init_weight(self.Weight['wconv1'], 'wconv1')
            b1 = init_biases(self.Weight['bconv1'], 'bconv1')
            conv1 = tf.add(tf.nn.conv2d(x_image ,w1 ,strides = [1 ,1 ,1 ,1] ,padding = "SAME") ,b1)
            bn1 = tf.layers.batch_normalization(conv1)
            conv1_relu = tf.nn.relu(bn1)

        with tf.name_scope('concat1'):
            conv1_dense = tf.concat([conv1_relu, x_image], 3)

        with tf.name_scope('conv2'):
            w2 = init_weight(self.Weight['wconv2'], 'wconv2')
            b2 = init_biases(self.Weight['bconv2'], 'bconv2')
            conv2 = tf.add(tf.nn.conv2d(conv1_dense ,w2 ,strides = [1 ,1 ,1 ,1] ,padding = "SAME") ,b2)
            bn2 = tf.layers.batch_normalization(conv2)
            conv2_relu = tf.nn.relu(bn2)

        with tf.name_scope('concat2'):
            conv2_dense = tf.concat([conv2_relu, conv1, x_image], 3)


        with tf.name_scope('conv3'):
            w3 = init_weight(self.Weight['wconv3'], 'wconv3')
            b3 = init_biases(self.Weight['bconv3'], 'bconv3')
            conv3 = tf.add(tf.nn.conv2d(conv2_dense ,w3 ,strides = [1, 1, 1, 1] ,padding = "SAME") ,b3)
            bn3 = tf.layers.batch_normalization(conv3)
            conv3_relu = tf.nn.relu(bn3)

        with tf.name_scope('pool1'):
            conv3_pool = tf.nn.max_pool(conv3_relu ,ksize = [1 ,2 ,2 ,1] ,strides = [1 ,2 ,2 ,1] ,padding = 'VALID')

        with tf.name_scope('conv4'):
            w4 = init_weight(self.Weight['wconv4'], 'wconv4')
            b4 = init_biases(self.Weight['bconv4'], 'bconv4')
            conv4 = tf.add(tf.nn.conv2d(conv3_pool, w4, strides=[1, 1, 1, 1], padding="SAME"),b4)
            bn4 = tf.layers.batch_normalization(conv4)
            conv4_relu = tf.nn.relu(bn4)

        with tf.name_scope('concat4'):
            conv4_dense = tf.concat([conv4_relu ,conv3_pool] ,3)



        with tf.name_scope('conv5'):
            w5 = init_weight(self.Weight['wconv5'], 'wconv5')
            b5 = init_biases(self.Weight['bconv5'], 'bconv5')
            conv5 = tf.add(tf.nn.conv2d(conv4_dense, w5, strides=[1, 1, 1, 1], padding="SAME"), b5)
            bn5 = tf.layers.batch_normalization(conv5)
            conv5_relu = tf.nn.relu(bn5)

        with tf.name_scope('concat5'):
            conv5_dense = tf.concat([conv5_relu ,conv4_relu ,conv3_pool], 3)

        with tf.name_scope('conv6'):
            w6 = init_weight(self.Weight['wconv6'], 'wconv6')
            b6 = init_biases(self.Weight['bconv6'], 'bconv6')
            conv6 = tf.add(tf.nn.conv2d(conv5_dense, w6, strides=[1, 1, 1, 1], padding="SAME"), b6)
            bn6 = tf.layers.batch_normalization(conv6)
            conv6_relu = tf.nn.relu(bn6)


        with tf.name_scope('pool3'):
            conv6_pool = tf.nn.max_pool(conv6_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


        with tf.name_scope('flatten'):
            #flat = tf.contrib.layers.flatten(conv3_pool)
            #flat = tf.contrib.layers.flatten(conv2_pool)
            flat = tf.contrib.layers.flatten(conv6_pool)

        with tf.name_scope('fc3'):
            fc3 = tf.layers.dense(flat ,units = self.Weight['fc_1'] ,activation = tf.nn.relu)
            fc3_drop = tf.layers.dropout(fc3, rate=self.dropout_rate, training=is_train)

        with tf.name_scope('fc4'):
            fc4 = tf.layers.dense(fc3_drop ,units = self.Weight["fc_2"] ,activation = tf.nn.relu)
            fc4_drop = tf.layers.dropout(fc4, rate=self.dropout_rate, training=is_train)

        with tf.name_scope('output'):
            output = tf.layers.dense(fc4_drop ,units = self.Weight['output'] ,activation =None)


        classes = tf.cast(tf.argmax(output ,axis = 1) ,tf.int32 ,name = 'prediction_class'),
        probability = tf.nn.softmax(output ,name = 'softmax_tensor')
        accuracy = tf.reduce_mean(tf.cast(tf.equal(classes, tf_y), tf.float32), name='accuracy')


        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_onehot ,logits = output) ,name = 'cost')


        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost ,name = 'train_op')


    def batch_generator(self ,X ,y ,batc_size ,shuffle = True):
        X_copy = np.array(X)
        y_copy = np.array(y)
        if shuffle:
            X_copy = X_copy.reshape(X.shape[0],X.shape[1]*X.shape[2])
            data = np.column_stack((X_copy, y_copy))  # column bind
            np.random.shuffle(data)
            X_copy = data[:, :-1].reshape(X.shape[0],X.shape[1],X.shape[2])
            y_copy = data[:, -1].astype(int)

        for i in range(0 ,X_copy.shape[0] ,batc_size):
            yield (X_copy[i : i+batc_size ,:],y_copy[i : i+batc_size])

    def train(self):
        self.sess.run(self.init_op)
        self.train_cost = []
        self.test_acc = []
        for epoch in range(self.epochs):
            batch = self.batch_generator(self.X_train, self.y_train, batc_size=self.batch_size)
            mean_cost = 0
            n = 0
            for batch_X, batch_y in batch:
                feed = {"input_x:0": batch_X, "input_y:0": batch_y, 'is_train:0': True}
                _, cost = self.sess.run(["train_op", "cost:0"], feed_dict=feed)
                mean_cost += cost
                n += 1
            mean_cost = mean_cost / n
            print("Epochs: %2d , train cost : %5f" % (epoch + 1, mean_cost))
            self.train_cost.append(mean_cost)
            pred_test = []
            for i in range(40):
                batch_test_x = self.X_test[i * 50:(i + 1) * 50, :]
                batch_test_y = self.y_test[i * 50:(i + 1) * 50]
                feed_test = {'input_x:0': batch_test_x, 'input_y:0': batch_test_y, 'is_train:0': False}
                pred = self.sess.run('accuracy:0', feed_dict=feed_test)
                pred_test.append(pred)
            print('Test Accuracy is : %2f' % (np.mean(pred_test)))
            self.test_acc.append(np.mean(pred_test))

    def predict(self ,data ,pred_prob = False):
        feed = {'input_x:0' : data,'is_train:0': False}
        if pred_prob:
            return self.sess.run('softmax_tensor:0' ,feed_dict = feed)
        else:
            return self.sess.run('prediction_class:0' ,feed_dict = feed)



model_3 = DenseNet(X_train = X_train_centered ,y_train = y_train ,X_test =X_test_centered ,y_test = y_test)
model_3.train()

model_32 = DenseNet_2(X_train = X_train_centered ,y_train = y_train ,X_test =X_test_centered ,y_test = y_test)
model_32.train()






##########  繪圖  ###############

import matplotlib.pyplot as plt
plt.plot(range(1,len(model_1.train_cost)+1) ,model_1.train_cost ,label = 'CNN-4Layers')
plt.plot(range(1,len(model_2.train_cost)+1) ,model_2.train_cost ,label = 'ResNet-4Layers')
plt.plot(range(1,len(model_3.train_cost)+1) ,model_3.train_cost ,label = 'DenseNet-4Layers')
plt.tight_layout()
plt.xlabel('Epochs')
plt.ylabel('Training_cost')
plt.legend()
plt.savefig('compare 1')
plt.show()



plt.plot(range(1,len(model_12.train_cost)+1) ,model_12.train_cost ,label = 'CNN-6Layers')
plt.plot(range(1,len(model_22.train_cost)+1) ,model_22.train_cost ,label = 'ResNet-6Layers')
plt.plot(range(1,len(model_32.train_cost)+1) ,model_32.train_cost ,label = 'DenseNet-6Layers')
plt.tight_layout()
plt.xlabel('Epochs')
plt.ylabel('Training_cost')
plt.legend()
plt.savefig('train_cost-4Layers')
plt.show()


plt.plot(range(1,len(model_1.train_cost)+1) ,model_1.train_cost ,label = 'CNN-4Layers')
plt.plot(range(1,len(model_12.train_cost)+1) ,model_12.train_cost ,label = 'CNN-6Layers')
plt.tight_layout()
plt.xlabel('Epochs')
plt.ylabel('Training_cost')
plt.legend()
plt.savefig('train_cost-CNN')
plt.show()

plt.plot(range(1,len(model_2.train_cost)+1) ,model_2.train_cost ,label = 'ResNet-4Layers')
plt.plot(range(1,len(model_22.train_cost)+1) ,model_22.train_cost ,label = 'ResNet-6Layers')
plt.tight_layout()
plt.xlabel('Epochs')
plt.ylabel('Training_cost')
plt.legend()
plt.savefig('train_cost-ResNet')
plt.show()

plt.plot(range(1,len(model_3.train_cost)+1) ,model_3.train_cost ,label = 'DenseNet-4Layers')
plt.plot(range(1,len(model_32.train_cost)+1) ,model_32.train_cost ,label = 'DenseNet-6Layers')
plt.tight_layout()
plt.xlabel('Epochs')
plt.ylabel('Training_cost')
plt.legend()
plt.savefig('train_cost-DenseNet')
plt.show()




plt.plot(range(1,len(model_1.test_acc)+1) ,model_1.test_acc ,label = 'CNN-4Layers_test_acc')
plt.plot(range(1,len(model_2.test_acc)+1) ,model_2.test_acc ,label = 'Resnet-4Layers_test_acc')
plt.plot(range(1,len(model_3.test_acc)+1) ,model_3.test_acc ,label = 'DenseNet-4Layers_test_acc')
plt.tight_layout()
plt.xlabel('Epochs')
plt.ylabel('Test_acc')
plt.legend()
plt.savefig('ACC_compare_4Layers')
plt.show()


plt.plot(range(1,len(model_12.test_acc)+1) ,model_12.test_acc ,label = 'CNN-6Layers_test_acc')
plt.plot(range(1,len(model_22.test_acc)+1) ,model_22.test_acc ,label = 'Resnet-6Layers_test_acc')
plt.plot(range(1,len(model_32.test_acc)+1) ,model_32.test_acc ,label = 'DenseNet-6Layers_test_acc')
plt.tight_layout()
plt.xlabel('Epochs')
plt.ylabel('Test_acc')
plt.legend()
plt.savefig('ACC_compare_6Layers')
plt.show()
####Weight必須在徒裡面設定
###呼叫tensor的name的時候必須+":0'