from keras.layers import Input, Dense
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

class AutoEncoder():
    def __init__(self):

        # -----------------------------Stopping Criteria

        self.callbacks = [
            EarlyStopping(monitor='val_loss', patience=2),
            ModelCheckpoint(filepath='best_autoencoder_model.h5', monitor='val_loss', save_best_only=True)]

        encoded_size = 3
        input_shape = 9

        ''' ------------------------------Stacked auto encoder that compresses the data to lower dimention '''

        input_val = Input(shape=(input_shape,))
        encoded = Dense(encoded_size*3, activation='relu')(input_val)
        encoded = Dense(encoded_size*2, activation='relu')(encoded)
        encoded = Dense(encoded_size, activation='relu')(encoded)

        decoded = Dense(encoded_size*2, activation='relu')(encoded)
        decoded = Dense(encoded_size*3, activation='relu')(decoded)
        decoded = Dense(input_shape, activation='sigmoid')(decoded)

        self.autoencoder = Model(input_val, decoded)
        self.encoder = Model(input_val, encoded)

        self.encoder.compile(optimizer='adam', loss='mean_squared_error')
        self.autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    def train(self, df):
        x_train, x_test = train_test_split(df, test_size=0.25)

        self.autoencoder.fit(x_train, x_train,
                             epochs=10000,
                             batch_size=256,
                             shuffle=True,
                             callbacks=self.callbacks,
                             validation_data=(x_test, x_test))

        self.encodeed_x_train = self.encoder.predict(x_train)
        self.encodeed_x_test = self.encoder.predict(x_test)
