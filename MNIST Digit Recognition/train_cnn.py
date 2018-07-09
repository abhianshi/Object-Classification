import time
import sys
import tensorflow as tf

# Helper functions
# Assigning the weights and bias
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# Convolution Layer
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Pooling Layer
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


########### Convolutional neural network class ############
class ConvNet(object):
    def __init__(self, mode):
        self.mode = mode

    # Read train, valid and test data.
    def read_data(self, train_set, test_set):
        # Load train set.
        trainX = train_set.images
        trainY = train_set.labels

        # Load test set.
        testX = test_set.images
        testY = test_set.labels

        return trainX, trainY, testX, testY

    # Baseline model. step 1
    def model_1(self, X, hidden_size):
        # One fully connected layer.
        # 784 * 100
        W_fc1 = weight_variable([28 * 28,hidden_size])
        b_fc1 = bias_variable([hidden_size])

        # Changing the shape of input X for making the calculations simple and multiplication with weight_variable
        h_flat = tf.reshape(X, [-1, 28 * 28])

        # Applying the sigmoid on the fully connected layer
        h_fc1 = tf.nn.sigmoid(tf.matmul(h_flat, W_fc1) + b_fc1)

        return h_fc1

    # Use two convolutional layers.
    def model_2(self, X, hidden_size):
        # Two convolutional layers + one fully connnected layer.

        # 1st Convolution Layer and Pooling Layer
        # Filter size is 5 * 5 and depth is 40, so taking this weight shape
        W_conv1 = weight_variable([5, 5, 1, 40])
        b_conv1 = bias_variable([40])
        x_image = tf.reshape(X, [-1, 28, 28, 1])

        # Produces an output of size 24,24,40
        h_conv1 = tf.nn.sigmoid(conv2d(x_image, W_conv1) + b_conv1)

        # Pooling divides width and height by 2
        h_pool1 = max_pool_2x2(h_conv1)

        # 2nd Convolution Layer and Pooling Layer
        W_conv2 = weight_variable([5, 5, 40, 40])
        b_conv2 = bias_variable([40])

        h_conv2 = tf.nn.sigmoid(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        W_fc1 = weight_variable([7 * 7 * 40,hidden_size])
        b_fc1 = bias_variable([hidden_size])
        h_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 40])
        h_fc1 = tf.nn.sigmoid(tf.matmul(h_flat, W_fc1) + b_fc1)

        return h_fc1

    # Replace sigmoid with ReLU.
    def model_3(self, X, hidden_size):
        # Two convolutional layers + one fully connected layer, with ReLU.

        # 1st Convolution and Pooling Layer with ReLU
        W_conv1 = weight_variable([5, 5, 1, 40])
        b_conv1 = bias_variable([40])
        x_image = tf.reshape(X, [-1, 28, 28, 1])

        # Produces an output of size 24,24,40
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

        # Pooling divides width and height by 2
        h_pool1 = max_pool_2x2(h_conv1)

        # 2nd Convolution and Pooling Layer with ReLU
        W_conv2 = weight_variable([5, 5, 40, 40])
        b_conv2 = bias_variable([40])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        W_fc1 = weight_variable([7 * 7 * 40,hidden_size])
        b_fc1 = bias_variable([hidden_size])
        h_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 40])
        h_fc1 = tf.nn.sigmoid(tf.matmul(h_flat, W_fc1) + b_fc1)

        return h_fc1

    # Add one extra fully connected layer.
    def model_4(self, X, hidden_size, decay):
        # Two convolutional layers + two fully connected layers, with ReLU.

        W_conv1 = weight_variable([5, 5, 1, 40])
        b_conv1 = bias_variable([40])
        x_image = tf.reshape(X, [-1, 28, 28, 1])

        # Produces an output of size 24,24,40
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

        # Pooling divides width and height by 2
        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable([5, 5, 40, 40])
        b_conv2 = bias_variable([40])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        # 1st Fully Connected Layer
        W_fc1 = weight_variable([7 * 7 * 40,hidden_size])
        b_fc1 = bias_variable([hidden_size])
        h_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 40])
        h_fc1 = tf.nn.sigmoid(tf.matmul(h_flat, W_fc1) + b_fc1)

        # 2nd Fully Connected Layer
        W_fc2 = weight_variable([hidden_size,hidden_size])
        b_fc2 = bias_variable([hidden_size])
        h_fc2 = tf.nn.sigmoid(tf.matmul(h_fc1, W_fc2) + b_fc2)

        # Passing the fully connected layer parameters for L2 Regularization
        return W_fc1,b_fc1,h_fc1,W_fc2,b_fc2,h_fc2

    # Use Dropout now.
    def model_5(self, X, hidden_size, is_train):
        # Two convolutional layers + two fully connected layers, with ReLU.
        # and  + Dropout.
        W_conv1 = weight_variable([5, 5, 1, 40])
        b_conv1 = bias_variable([40])
        x_image = tf.reshape(X, [-1, 28, 28, 1])

        # Produces an output of size 24,24,40
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

        # Pooling divides width and height by 2
        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable([5, 5, 40, 40])
        b_conv2 = bias_variable([40])

        # 2nd convolutional and Pooling Layer
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        dropout = 0.5
        h_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 40])

        # 1st fully connected layer
        W_fc1 = weight_variable([7 * 7 * 40,hidden_size])
        b_fc1 = bias_variable([hidden_size])

        # Using droupout on 1st fully connected layer
        h_fc1_drop = tf.nn.dropout(h_flat, dropout)
        h_fc1 = tf.nn.sigmoid(tf.matmul(h_fc1_drop, W_fc1) + b_fc1)

        # 2nd Fully Connected Layer
        W_fc2 = weight_variable([hidden_size,hidden_size])
        b_fc2 = bias_variable([hidden_size])

        # Using droupout on 2nd fully connected layer
        h_fc2_drop = tf.nn.dropout(h_fc1, dropout)
        h_fc2 = tf.nn.sigmoid(tf.matmul(h_fc2_drop, W_fc2) + b_fc2)

        return h_fc2

    # Entry point for training and evaluation.
    def train_and_evaluate(self, FLAGS, train_set, test_set):
        class_num = 10
        num_epochs = FLAGS.num_epochs
        batch_size = FLAGS.batch_size
        learning_rate = FLAGS.learning_rate
        hidden_size = FLAGS.hiddenSize
        decay = FLAGS.decay

        trainX, trainY, testX, testY = self.read_data(train_set, test_set)
        input_size = trainX.shape[1]
        train_size = trainX.shape[0]
        test_size = testX.shape[0]

        # Change the shape of trainX, trainY, testX and testY for easy computation
        trainX = trainX.reshape((-1, 28, 28, 1))
        testX = testX.reshape((-1, 28, 28, 1))

        with tf.Graph().as_default():
            # Input data
            X = tf.placeholder(tf.float32, [None, 28, 28, 1])
            #Y = tf.placeholder(tf.int32, [None])
            # Change the labels type from [7] to [0,0,0,0,0,0,0,1,0,0] while reading
            Y = tf.placeholder(tf.int32, [None, 10])

            is_train = tf.placeholder(tf.bool)

            # model 1: base line
            if self.mode == 1:
                features = self.model_1(X, hidden_size)

            # model 2: use two convolutional layer
            elif self.mode == 2:
                features = self.model_2(X, hidden_size)

            # model 3: replace sigmoid with relu
            elif self.mode == 3:
                features = self.model_3(X, hidden_size)

            # model 4: add one extral fully connected layer
            elif self.mode == 4:
                W_fc1,b_fc1,h_fc1,W_fc2,b_fc2,h_fc2 = self.model_4(X, hidden_size, decay)
                features = h_fc2

            # model 5: utilize dropout
            elif self.mode == 5:
                features = self.model_5(X, hidden_size, is_train)

            # 100 * 10
            # Output Layer
            W_out = weight_variable([hidden_size, 10])
            b_out = bias_variable([10])
            y = tf.matmul(features, W_out) + b_out

            # Define softmax layer, use the features.
            # Softmax layer is combined with loss function, we took the softmax_cross_entropy_with_logits
            # softmax_cross_entropy_with_logits first computes softmax on the logits and then do the cross_entropy
            logits = y

            # Define loss function, use the logits.
            if(self.mode == 4):
                beta = 0.01
                regularization_fc1 = beta * (tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) + tf.nn.l2_loss(h_fc1))
                regularization_fc2 = beta * (tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc2) + tf.nn.l2_loss(h_fc2))
                regularization = regularization_fc1 + regularization_fc2
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = y) + regularization)
            else:
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = y))

            # Define training op, use the loss.
            train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

            # Define accuracy op.
            correct_prediction = tf.equal(tf.argmax(Y,1),tf.argmax(y,1))

            # Taking the mean accuracy, so no need to do the division by total at the end of the code while printing the accuracy.
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            # ======================================================================
            # Allocate percentage of GPU memory to the session.
            # If you system does not have GPU, set has_GPU = False
            #
            has_GPU = False
            if has_GPU:
                gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
                config = tf.ConfigProto(gpu_options=gpu_option)
            else:
                config = tf.ConfigProto()

            # Create TensorFlow session with GPU setting.
            with tf.Session(config=config) as sess:
                tf.global_variables_initializer().run()

                for i in range(num_epochs):
                    print(20 * '*', 'epoch', i + 1, 20 * '*')
                    start_time = time.time()
                    s = 0

                    while s < train_size:
                        e = min(s + batch_size, train_size)
                        batch_x = trainX[s: e]
                        batch_y = trainY[s: e]

                        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, is_train: True})
                        s = e
                    end_time = time.time()
                    print ('the training took: %d(s)' % (end_time - start_time))

                    total_correct = sess.run(accuracy, feed_dict={X: testX, Y: testY, is_train: False})
                    print ('accuracy of the trained model %f' % (total_correct))
                    print ()

                return sess.run(accuracy, feed_dict={X: testX, Y: testY, is_train: False})
