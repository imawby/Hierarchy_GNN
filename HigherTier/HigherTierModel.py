from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense

def HigherTierModel(nVariables):
    
    networkInputs = Input(shape=(nVariables, ))
    
    # Convolutions first?
    #x_c = signals
    #x_c = Convolution1D(64, 10, kernel_initializer='lecun_uniform',  activation='relu')(x_c)
    #x_c = Dropout(dropoutRate)(x_c)
    #x_c = Convolution1D(64, 10, kernel_initializer='lecun_uniform',  activation='relu')(x_c)
    #x_c = Dropout(dropoutRate)(x_c)
    #x_c = Convolution1D(64, 10, kernel_initializer='lecun_uniform',  activation='relu')(x_c)
    #x_c = Dropout(dropoutRate)(x_c)
    #x_c = Convolution1D(64, 3, kernel_initializer='lecun_uniform',  activation='relu')(x_c)
    #x_c = Flatten()(x_c)

    # Lets put the convolutions through a couple of dense layers first
    x = Dense(128, activation="relu", kernel_initializer='lecun_uniform')(networkInputs)
    #x = Dropout(dropoutRate)(x)
    x = Dense(128, activation="relu", kernel_initializer='lecun_uniform')(x)
    #x = Dropout(dropoutRate)(x)
    x = Dense(64, activation="relu", kernel_initializer='lecun_uniform')(x)
    
    prediction = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=networkInputs, outputs=prediction)
    
    return model