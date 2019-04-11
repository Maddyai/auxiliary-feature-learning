from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os.path


class Base(object):
    def __init__(self):
        self.layer_size = (0.9, 0.75, 0.6)


class AutoEncoder(Base):
    def __init__(self, input_shape, training_size):
        super().__init__()

        # -----------------------------Stopping Criteria
        self.model_name = 'models/best_autoencoder_model_'
        self.model_name += 'train_size_' + str(training_size) + '_'
        for suffix in self.layer_size:
            self.model_name += str(suffix)
        self.model_name += '.h5'
        self.callbacks = [
            EarlyStopping(monitor='val_loss', patience=2),
            ModelCheckpoint(filepath=self.model_name,
                            monitor='val_loss', save_best_only=True)]

        ''' ------------------------------Stacked auto encoder that
        compresses the data to lower dimention '''

        input_lyr = Input(shape=(input_shape,))
        encoded = Dense(int(input_shape*self.layer_size[0]), activation='relu')(input_lyr)
        encoded = Dense(int(input_shape*self.layer_size[1]), activation='relu')(encoded)
        encoded = Dense(int(input_shape*self.layer_size[2]), activation='relu')(encoded)

        decoded = Dense(int(input_shape*self.layer_size[1]), activation='relu')(encoded)
        decoded = Dense(int(input_shape*self.layer_size[0]), activation='relu')(decoded)
        decoded = Dense(input_shape, activation='relu')(decoded)

        self.encoder = Model(input_lyr, encoded)
        self.autoencoder = Model(input_lyr, decoded)

        self.encoder.compile(optimizer='adam', loss='mean_squared_error')
        self.autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    def train(self, x_train, x_val):
        if not os.path.isfile(self.model_name):
            self.autoencoder.fit(x_train, x_train,
                                 epochs=10000,
                                 batch_size=256,
                                 shuffle=True,
                                 callbacks=self.callbacks,
                                 validation_data=(x_val, x_val))

    def encoded_data(self, x_train, x_val, x_test):
        self.autoencoder.load_weights(self.model_name)
        x_train = self.encoder.predict(x_train)
        x_val = self.encoder.predict(x_val)
        x_test = self.encoder.predict(x_test)
        return x_train, x_val, x_test


class NeuralNetwork(Base):
    def __init__(self, input_shape, output_shape, training_size):
        super().__init__()
        input_shape = int(input_shape*self.layer_size[-1])

        # -----------------------------Stopping Criteria
        self.model_name = 'models/best_neuralnetwork_model'
        self.model_name += 'train_size_' + str(training_size) + '_'
        self.model_name += '.h5'

        self.callbacks = [
            EarlyStopping(monitor='val_acc', patience=12),
            ModelCheckpoint(filepath=self.model_name,
                            monitor='val_acc', save_best_only=True)]

        ''' ------------------------------Stacked auto encoder that
        compresses the data to lower dimention '''

        # input_lyr = Input(shape=(input_shape,))
        # hidden_lyr = Dense(int(input_shape), activation='sigmoid')(input_lyr)
        # hidden_lyr = Dense(int(input_shape), activation='sigmoid')(hidden_lyr)
        # output_lyr = Dense(output_shape, activation='softmax')(hidden_lyr)

        # self.nn_model = Model(input_lyr, output_lyr)
        # # self.nn_model.compile(optimizer='adam', loss='mean_squared_error')
        # self.nn_model.compile(loss='binary_crossentropy', optimizer='adam',
        #                       metrics=['accuracy'])
        from keras.models import Sequential
        from keras.layers import Dense, Dropout
        self.nn_model = Sequential()
        self.nn_model.add(Dense(input_shape, input_dim=input_shape, activation='relu'))
        # self.nn_model.add(Dropout(0.5))
        self.nn_model.add(Dense(input_shape, activation='relu'))
        # self.nn_model.add(Dropout(0.5))
        self.nn_model.add(Dense(output_shape, activation='sigmoid'))

        self.nn_model.compile(loss='binary_crossentropy',
                              optimizer='adam', metrics=['accuracy'])

    def train(self, x_train, y_train, x_val, y_val):
        if not os.path.isfile(self.model_name):
            self.nn_model.fit(x_train, y_train,
                              epochs=10000,
                              batch_size=256,
                              shuffle=True,
                              callbacks=self.callbacks,
                              validation_data=(x_val, y_val))

        self.nn_model.load_weights(self.model_name)
        self.val_acc = self.nn_model.evaluate(x_val, y_val, verbose=0)
        print('[INFO] Accuracy on Validation dataset: {0:.2f}%'.format(self.val_acc[1]*100))

    def evaluate(self, x_test, y_test):
        self._load_best_model()
        self.test_acc = self.nn_model.evaluate(x_test, y_test)
        print('[INFO] Final Accuracy on Test dataset : {0:.2f}%'.format(self.test_acc[1]*100))

    def _load_best_model(self):
        self.nn_model.load_weights(self.model_name)

    def result(self):
        return (self.val_acc[1], self.test_acc[1])
