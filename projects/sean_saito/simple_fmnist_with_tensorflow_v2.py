# this is the implementation of the fashion mnist classifier with tensorflow version 2
import tensorflow as tf
import numpy as np
from keras.datasets import fashion_mnist
import keras
import os
import sys
import logging
from keras import utils as np_utils

# logger setup
log_format = '%(asctime)s | %(levelname)s: %(message)s'
logging.basicConfig(format=log_format, level=logging.NOTSET)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(console_handler)


class FashionMnistClassifier:
    def __init__(self, learning_rate, num_epochs, dropout, batch_size, regularization_factor, name="fashion_mnist_cnn"):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.dropout = dropout
        self.batch_size = batch_size
        self.regularization_factor = regularization_factor
        self.logger = logger
        self.saves_dir = "saves"
        self.logs_dir = "logs"
        self.name = name
        
        # set model and summary writer initially to None
        self.model = None
        self.writer = None
        
        # create the saves adn logs directory if they dont exist
        os.makedirs(self.saves_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # create paths for saving weights and logs
        self.saves_path = os.path.join(self.saves_dir, self.name)
        self.logs_path = os.path.join(self.logs_dir, name)
        
    def build(self,input_shape, num_outputs):
        # define model and build layers
        # inputs = tf.keras.Input(shape=input_tensor.shape, batch_size=self.batch_size)
        self.model = tf.keras.Sequential()
        with tf.name_scope("convolutional_layers"):
            # sequentially pass data through the layers
            conv_1 = tf.keras.layers.Conv2D(
                                            filters=16,
                                            kernel_size=(5, 5),
                                            strides=(1, 1),
                                            padding="SAME",
                                            activation=tf.nn.relu,
                                            kernel_regularizer=tf.keras.regularizers.L2(l2=self.regularization_factor),
                                            name="conv1", input_shape=input_shape)
            self.model.add(conv_1)
            conv_2 = tf.keras.layers.Conv2D(
                                            filters=32,
                                            kernel_size=(3, 3),
                                            strides=(1, 1),
                                            padding="SAME",
                                            activation=tf.nn.relu,
                                            kernel_regularizer=tf.keras.regularizers.L2(l2=self.regularization_factor),
                                            name="conv2"
                                            )
            self.model.add(conv_2)
            
            
            pool_3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                            strides=1,
                                            padding="SAME",
                                            name="pool3")
            self.model.add(pool_3)

            drop_4 = tf.keras.layers.Dropout(rate=self.dropout, name="drop4")
            self.model.add(drop_4)
            
            # second convolution block
            conv_5 = tf.keras.layers.Conv2D(
                                            filters=64,
                                            kernel_size=(3, 3),
                                            strides=(1, 1),
                                            padding="SAME",
                                            activation=tf.nn.relu,
                                            kernel_regularizer=tf.keras.regularizers.L2(l2=self.regularization_factor),
                                            name="conv5"
                                            )
            self.model.add(conv_5)
            
            conv_6 = tf.keras.layers.Conv2D(
                                            filters=128,
                                            kernel_size=(3, 3),
                                            strides=(1, 1),
                                            padding="SAME",
                                            activation=tf.nn.relu,
                                            kernel_regularizer=tf.keras.regularizers.L2(l2=self.regularization_factor),
                                            name="conv6"
                                            )
            self.model.add(conv_6)
            
            pool_7 = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                            strides=1,
                                            padding="SAME",
                                            name="pool7")
            self.model.add(pool_7)

            drop_8 = tf.keras.layers.Dropout(rate=self.dropout, name="drop8")
            self.model.add(drop_8)
        
        with tf.name_scope("fully_connected_layers"):
            # flattenining layer
            flattened = tf.keras.layers.Flatten(name="flattend")
            self.model.add(flattened)
            fc_9 = tf.keras.layers.Dense(units=1024,
                                         activation=tf.keras.activations.relu, 
                                         kernel_regularizer=tf.keras.regularizers.L2(l2=self.regularization_factor),
                                         name="fc9")
            self.model.add(fc_9)
            drop_10 = tf.keras.layers.Dropout(rate=self.dropout, name="drop10")
            self.model.add(drop_10)
            logits = tf.keras.layers.Dense(units=num_outputs, 
                                           kernel_regularizer=tf.keras.regularizers.L2(l2=self.regularization_factor), 
                                           name="logits")
            self.model.add(logits)
        
        # build the model and show the model summary
        logger.info("Building Model")
        self.model.build()
        self.model.summary()
        logger.info("Done building Model")
        
    def _create_tf_dataset(self, x, y):
        # turn numpy array into tensorlow dataset
        dataset = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(x), 
                                       tf.data.Dataset.from_tensor_slices(y))).shuffle(50).repeat().batch(self.batch_size)
        # return created dataset
        return dataset
    
    def _log_loss_and_acc(self, epoch, loss, acc, suffix):
        # write loss and accuracy into summary writer for tensor board
        with self.writer.as_default():
            tf.summary.scalar(f'loss_{suffix}', loss, epoch)
            tf.summary.scalar(f'acc_{suffix}', loss, epoch)
            
    @tf.function
    def _train_step(self, optimizer, loss_fn, x_train, y_train, train_loss_fn, train_acc_fn):
        with tf.GradientTape() as tape:
            predictions = self.model(x_train, training=True)
            loss = loss_fn(predictions, y_train)
        # get the gradients from the just concluded inference
        grads = tape.gradient(loss, self.model.trainable_variables)
        # use those gradients to optimize trainable parameters using the optimizer
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        # calculate the training loss and training accuracy
        train_loss_fn(loss)
        train_acc_fn(y_train, predictions)
    
    @tf.function
    def _validation_step(self, x_test, y_test, loss_fn, valid_loss_fn, valid_acc_fn):
        predictions = self.model(x_test, training=False)
        loss = loss_fn(y_test, predictions)
        # calculate validation loss and validation accuracy
        valid_loss_fn(loss)
        valid_acc_fn(y_test, predictions)
            
    def fit(self, x_train, y_train, x_test, y_test):
        # if the model has not yet been built, build the model
        if self.model == None:
            self.build()
        
        # create training and validation dataset
        train_dataset = self._create_tf_dataset(x_train, y_train)
        valid_dataset = self._create_tf_dataset(x_test, y_test)
        train_iterator = iter(train_dataset)
        valid_iterator = iter(valid_dataset)
        
        # create loss object and optimizer
        loss_obj = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam()
        
        # we define our metrics
        train_loss = tf.keras.metrics.Mean(name="train_loss", dtype=tf.float32)
        train_acc = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")
        valid_loss = tf.keras.metrics.Mean(name="validation_loss", dtype=tf.float32)
        valid_acc = tf.keras.metrics.CategoricalAccuracy(name="validation_accuracy")
        
        self.writer = tf.summary.create_file_writer(self.logs_path)
        
        for epoch in range(1, self.num_epochs+1):
            # new epoch starting
            # start training loop
            x_train, y_train = next(train_iterator)
            self._train_step(optimizer, loss_obj, x_train, y_train, train_loss, train_acc)
            self._log_loss_and_acc(epoch, train_loss.result(), train_acc.result(), suffix="train")
            logger.info(f"==============>Epoch:{epoch} training ending")
            
            # if epocoh is a multiple of 10 start validation loop
            if epoch % 10 == 0:
                logger.info(f"==============>Epoch:{epoch} validation starting")
                x_test, y_test = next(valid_iterator)
                self._validation_step(x_test, y_test, loss_obj, valid_loss, valid_acc)
                self._log_loss_and_acc(epoch, valid_loss.result(), valid_acc.result(), suffix="validation")
                logger.info(f"==============>Epoch:{epoch}")    
                logger.info(f"\tTraining loss: {train_loss.result()}")    
                logger.info(f"\tTraining accuracy: {train_acc.result()}")
                logger.info(f"\tValidation loss: {valid_loss.result()}")
                logger.info(f"\tValidation accuracy: {valid_acc.result()}")
                logger.info(f"==============>Epoch:{epoch} validation ending")

            
            # at the end of every epoch reset the loss and accuracy values for training and validation
            # Reset metrics every epoch
            train_loss.reset_states()
            valid_loss.reset_states()
            train_acc.reset_states()
            valid_acc.reset_states()
            
            # save model weights at the end of every epoch
            self.model.save_weights(self.saves_path)
            logger.info(f"==============>Epoch:{epoch} completed successfully")
                
    def predict(self, features):
        logits = self.model(features)
        return logits
    
    def predict_category(self, features):
        logits = self.predict(features)
        return np.argmax(logits)
    
    
    
class ModelBuilder:
    def __init__(self, dataset_fetch_function, model):
        # setup dataset
        logger.info("Getting the FashionMNST dataset")
        (x_train, y_train), (x_test, y_test) = dataset_fetch_function()
        
        logger.info("Shape of training data: ")
        logger.info(f"Train: {x_train.shape}")
        logger.info(f"Test: {x_test.shape}")
        
        logger.info("Adding channel axis to the matrix")
        x_train = x_train[:, :, :, np.newaxis]
        x_test = x_test[:, :, :, np.newaxis]
        
        logger.info("Simple transformation by dividing pixels by 255")
        x_train = x_train/255
        x_test = x_test/255
        
        self.x_train = x_train.astype(np.float32)
        self.x_test = x_test.astype(np.float32)
        self.y_train =  y_train.astype(np.float32)
        self.y_test = y_test.astype(np.float32)
        
        self.testing_data = (self.x_test[:5], self.y_test[:5])
        
        num_classes = len(np.unique(y_train)) # is the y_train one hot encoded?
        
        logger.info("Turn output into one hot encoding")
        # performs one hot encoding on y_train and y_test
        self.y_train = np_utils.to_categorical(self.y_train, num_classes=num_classes)
        self.y_test = np_utils.to_categorical(self.y_test, num_classes=num_classes)
        
        self.model = model
        self.model.build((28, 28, 1), num_classes)
    
    def build(self):
        logger.info("Training CNN")
        self.model.fit(self.x_train, self.y_train, self.x_test, self.y_test)
    
    def test():
        pass
    
    
    
if __name__ == '__main__':
    test_params = {"learning_rate": 3e-4,
                   "num_epochs": 100,
                   "dropout": 1e-5,
                   "batch_size": 600,
                   "regularization_factor": 1e-3}
    logger.info("Initializing fashion mnist classifer model...")
    classifier = FashionMnistClassifier(**test_params)
    logger.info("Initializing model buildier with fashion mnist and fashion mnist classifier")
    builder = ModelBuilder(fashion_mnist.load_data, classifier)
    logger.info("Building model")
    builder.build()
    logger.info("Done!")

