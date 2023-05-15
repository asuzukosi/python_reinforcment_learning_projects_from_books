import logging
logging.basicConfig(level = logging.INFO)
import os
import numpy as np
import sys
from PIL import Image

# setup logger
logger = logging.getLogger(__name__)

import tensorflow as tf
from keras.datasets import fashion_mnist
from keras import utils as np_utils
import keras


# disable all v2 behavior as we would be working with v1 API
tf.compat.v1.disable_v2_behavior()


# implementation of our simple cnn class
class SimpleCNN:
    def __init__(self, learning_rate, num_epochs, beta, batch_size, dropout=1e-5):
        # we will initialize our learning hyperparameter for the simple cnn class
        self.learning_rate = learning_rate # specify the learning rate for the optimizer
        self.num_epochs = num_epochs # specify the number of training epochs
        self.beta = beta # specify the parameter for the l2 normalization
        self.batch_size = batch_size # batch size for training a neural network
        self.dropout = dropout # percentage of neural network that would be dropped off during training
        
        # utility attributes
        self.save_dir = "saves"
        self.logs_dir = "logs"
        
        # create saves and logs directory
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # create the saves and logs path for our simple cnn
        self.save_path = os.path.join(self.save_dir, "simple_cnn")
        self.logs_path = os.path.join(self.logs_dir, "simple_cnn")
        
        
    
    def build(self, input_tensor:int, num_classes:int):
        """
        This methods is responsible for building the architecture of our neural network
        Args:
            input_tensor (int): input tensor size
            num_classes (int): number of output classes
        Returns: returns output logits before softmax
        """
        with tf.name_scope("input_placeholders"): # allows us the name sections of our network operation properly
            self.is_training = tf.compat.v1.placeholder_with_default(True, shape=(), name="is_training")
            
        with tf.name_scope("convolutional_neural_network"): # convolutional neural network section
            # first convolutional layer
            conv_1 = tf.compat.v1.layers.conv2d(input_tensor,
                                                filters=16, 
                                                kernel_size=(5, 5),
                                                strides=(1, 1), 
                                                padding="SAME", 
                                                activation=tf.nn.relu,
                                                kernel_regularizer=tf.keras.regularizers.L2(l2=self.beta),
                                                name="conv_1")
            
            # second convolutional layer
            conv_2 = tf.compat.v1.layers.conv2d(conv_1,
                                                filters=32,
                                                kernel_size=(3, 3),
                                                strides=(1, 1),
                                                padding="SAME",
                                                activation=tf.nn.relu,
                                                kernel_regularizer=tf.keras.regularizers.L2(l2=self.beta),
                                                name="conv_2")
            
            # first max pooling layer
            pool_3 = tf.compat.v1.layers.max_pooling2d(conv_2, 
                                               pool_size=(2, 2),
                                               strides=1,
                                               padding="SAME",
                                               name="pool_3")
            
            # first dropout layer
            drop_4 = tf.compat.v1.layers.dropout(pool_3, 
                                                 training=self.is_training, 
                                                 name="drop_4")  # set the training parameter to determine if the 
                                                                                                          # model is training or not
            # fifth convolutional layer                                                                             
            conv_5 = tf.compat.v1.layers.conv2d(drop_4,
                                                filters=64,
                                                kernel_size=(3, 3),
                                                strides=(1, 1),
                                                padding="SAME",
                                                activation=tf.nn.relu,
                                                kernel_regularizer=tf.keras.regularizers.L2(l2=self.beta),
                                                name="conv_5")
            
            # sixth convolutional layer                                                                             
            conv_6= tf.compat.v1.layers.conv2d( conv_5,
                                                filters=128,
                                                kernel_size=(3, 3),
                                                strides=(1, 1),
                                                padding="SAME",
                                                activation=tf.nn.relu,
                                                kernel_regularizer=tf.keras.regularizers.L2(l2=self.beta),
                                                name="conv_6")
            
            # second max pooling layer
            pool_7 = tf.compat.v1.layers.max_pooling2d( conv_6,
                                                        pool_size=(2, 2),
                                                        strides=1,
                                                        padding="SAME",
                                                        name="pool_7")
            
            # second dropout layer
            drop_8 = tf.compat.v1.layers.dropout(pool_7, training=self.is_training, name="drop_8") # set the training parameter to determine if the 
                                                                                                          # model is training or not
                                                                                                          
                                                                                                          
        with tf.name_scope("fully_connected_layers"):
            # flattens the dropout 8 layer so it can be passed to the fully connected layers
            flattened = tf.compat.v1.layers.flatten(drop_8, name="flatten") # serves as the bridge between convolutional and fully connected layers
            # first fully connected layer
            fc_9 = tf.compat.v1.layers.dense(flattened, 
                                             units=1024,
                                             activation=tf.nn.relu,
                                             kernel_regularizer=tf.keras.regularizers.L2(l2=self.beta),
                                             name="fc_9")
            
            # third dropout layer
            drop_10 = tf.compat.v1.layers.dropout(fc_9, training=self.is_training, name="drop_10")
            # second fully connected layer and final output layer
            logits = tf.compat.v1.layers.dense(drop_10,
                                               units=num_classes, 
                                               kernel_regularizer=tf.keras.regularizers.L2(l2=self.beta))# produces the final output logits
        
        # return the final prediction logits
        return logits
    
    
    def _create_tf_dataset(self, x, y):
        # creates tensorflow dataset from numpy array
        dataset = tf.compat.v1.data.Dataset.zip((tf.compat.v1.data.Dataset.from_tensor_slices(x),  # creates the dataset by zipping the x and y values to gether, shufflig them repeatedly and then splitting them into batches
                                       tf.compat.v1.data.Dataset.from_tensor_slices(y))).shuffle(50).repeat().batch(self.batch_size)
        # returns the created dataset
        return dataset
        
    
    def _log_loss_and_acc(self, epoch, loss, acc, suffix):
        # add loss and accurac to the summary using summary writer
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=f"loss_{suffix}", simple_value=float(loss)),
                                              tf.compat.v1.Summary.Value(tag=f"acc_{suffix}", simple_value=float(acc))])
        self.summary_writer.add_summary(summary, epoch)

            
    def fit(self, x_train, y_train, x_test, y_test):
        # triggers the training of our neural network
        graph = tf.compat.v1.Graph() # computation graph for performing fit operation
        with graph.as_default():
            sess = tf.compat.v1.Session()
            # create our training and testing dataset
            train_dataset = self._create_tf_dataset(x_train, y_train) # batched training and test tensorflow dataset
            valid_dataset = self._create_tf_dataset(x_test, y_test)
        
            # create generic iterator
            # we create a data iterator based on the type and shape of t he training dataset
            iterator = tf.compat.v1.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
            # this returns the next item, but nothing has been loaed inot the iterator, so what is it retreiving
            next_tensor_batch = iterator.get_next()
            
            # create operations for initializing training and validation set
            train_init_ops = iterator.make_initializer(train_dataset)
            valid_init_ops = iterator.make_initializer(valid_dataset)
        
            input_tensor, labels =  next_tensor_batch
            
            num_classes = y_train.shape[1]
        
            # build network
            logits = self.build(input_tensor=input_tensor, num_classes=num_classes)
            logger.info("Built Network")
        
            predition = tf.nn.softmax(logits, name="predictions")
            loss_ops = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits), name="loss")
        
            # define optimizer
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
            train_ops = optimizer.minimize(loss_ops)
        
            correct = tf.equal(tf.argmax(predition,1), tf.argmax(labels, 1), name="correct")
            accuracy_ops = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
            
            # create global variables initializer
            initializer = tf.compat.v1.global_variables_initializer()
            logger.info("Initializing all variables")
            # run global variables initializer
            sess.run(initializer)
            logger.info("Initialized all variables")

            # run training data initializr
            sess.run(train_init_ops)
            logger.info("Initialized data Iterator")
            # create weights saver and summary writter
            self.saver = tf.compat.v1.train.Saver()
            self.summary_writer = tf.compat.v1.summary.FileWriter(logdir=self.logs_path)
        
            # implement actual training loop
            logger.info(f"Train CNN for {self.num_epochs} epochs")
            for epoch_idx in range(1, self.num_epochs+1):
                loss, _,  acc = sess.run([loss_ops, train_ops, accuracy_ops])
                # log loss and accuracy
                self._log_loss_and_acc(epoch_idx, loss, acc, "train")
                
                
                if epoch_idx % 10 == 0:
                    sess.run(valid_init_ops) # how does the loss and accuracy know to use the training data?
                    valid_loss, valid_acc = sess.run([loss_ops, accuracy_ops], feed_dict={self.is_training: False})
                    self._log_loss_and_acc(epoch_idx, valid_loss, valid_acc, "valid")
                    logger.info(f"==============>Epoch:{epoch_idx}")    
                    logger.info(f"\tTraining loss: {loss}")    
                    logger.info(f"\tTraining accuracy: {acc}")
                    logger.info(f"\tValidation loss: {valid_loss}")
                    logger.info(f"\tValidation accuracy: {valid_acc}")
                
                # create checkpoints for every epoch
                self.saver.save(sess, self.save_path)
                
                
                

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout,
                        level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("Getting the FashionMNST dataset")
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    
    
    logger.info("Shape of training data: ")
    logger.info(f"Train: {x_train.shape}")
    logger.info(f"Test: {x_test.shape}")
    
    logger.info("Adding channel axis to the matrix")
    x_train = x_train[:, :, :, np.newaxis]
    x_test = x_test[:, :, :, np.newaxis]
    
    logger.info("Simple transformation by dividing pixels by 255")
    x_train = x_train/255
    x_test = x_test/255
    
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_train =  y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)
    
    num_classes = len(np.unique(y_train)) # is the y_train one hot encoded?
    
    logger.info("Turn output into one hot encoding")
    # performs one hot encoding on y_train and y_test
    y_train = np_utils.to_categorical(y_train, num_classes=num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes=num_classes)
    
    cnn_params = {
        "learning_rate": 3e-4,
        "num_epochs": 100,
        "beta": 1e-3,
        "batch_size": 32
    }
    
    logger.info("Initializing CNN")
    simple_cnn = SimpleCNN(**cnn_params)
    logger.info("Training CNN")
    simple_cnn.fit(x_train, y_train, x_test, y_test)
    print("Done!")


    