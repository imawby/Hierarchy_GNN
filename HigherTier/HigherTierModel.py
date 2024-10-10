from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout, Concatenate

def HigherTierModel(nVariables, dropoutRate=0.5):
    
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
    x = Dropout(dropoutRate)(x)
    x = Dense(128, activation="relu", kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(64, activation="relu", kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(32, activation="relu", kernel_initializer='lecun_uniform')(x)
    
    prediction = Dense(3, activation='softmax')(x)
    
    model = Model(inputs=networkInputs, outputs=prediction)
    
    return model


##########################################################################################################################
##########################################################################################################################

def HigherTierFinalOutputModel(nVariables, dropoutRate=0.5):
    
    networkInputs = Input(shape=(nVariables, ))
    
    # Lets put the convolutions through a couple of dense layers first
#     x = Dense(128, activation="relu", kernel_initializer='lecun_uniform')(networkInputs)
#     x = Dropout(dropoutRate)(x)
#     x = Dense(128, activation="relu", kernel_initializer='lecun_uniform')(x)
#     x = Dropout(dropoutRate)(x)
#     x = Dense(64, activation="relu", kernel_initializer='lecun_uniform')(x)
#     x = Dropout(dropoutRate)(x)
#     x = Dense(32, activation="relu", kernel_initializer='lecun_uniform')(x)
    
    prediction = Dense(1, activation='sigmoid')(networkInputs)
    
    model = Model(inputs=networkInputs, outputs=prediction)
    
    return model


##########################################################################################################################
##########################################################################################################################

def PrimaryShowerFinalOutputModel(nVariables, dropoutRate=0.5):
    
    networkInputs = Input(shape=(nVariables, ))
    
    # Lets put the convolutions through a couple of dense layers first
    x = Dense(128, activation="relu", kernel_initializer='lecun_uniform')(networkInputs)
    x = Dropout(dropoutRate)(x)
    x = Dense(128, activation="relu", kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(64, activation="relu", kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(32, activation="relu", kernel_initializer='lecun_uniform')(x)
    
    prediction = Dense(1, activation='sigmoid')(networkInputs)
    
    model = Model(inputs=networkInputs, outputs=prediction)
    
    return model


##########################################################################################################################
##########################################################################################################################



def HigherTierGroupModel(nVariables, dropoutRate=0.5):
    
    networkInputs = Input(shape=(nVariables, ))
    
    orientationBranch_0 = OrientationBranch(networkInputs, dropoutRate)
    orientationBranch_1 = OrientationBranch(networkInputs, dropoutRate)
    orientationBranch_2 = OrientationBranch(networkInputs, dropoutRate)
    orientationBranch_3 = OrientationBranch(networkInputs, dropoutRate)
    
    prediction_0 = Dense(3, activation='softmax', name='orientation_0')(orientationBranch_0)
    prediction_1 = Dense(3, activation='softmax', name='orientation_1')(orientationBranch_1)
    prediction_2 = Dense(3, activation='softmax', name='orientation_2')(orientationBranch_2)
    prediction_3 = Dense(3, activation='softmax', name='orientation_3')(orientationBranch_3)
    
    combinedBranches = AllBranches(prediction_0, prediction_1, prediction_2, prediction_3, dropoutRate)
    prediction_4 = Dense(2, activation='softmax', name='final')(combinedBranches)
    
    model = Model(inputs=networkInputs, outputs=[prediction_0, prediction_1, prediction_2, prediction_3, prediction_4])
    
    return model


##########################################################################################################################
##########################################################################################################################
# need to have other bits too...
def OrientationBranch(branchInputs, dropoutRate):
    ################################
    # Start branch
    ################################
    x = Dense(128, activation="relu", kernel_initializer='lecun_uniform')(branchInputs)
    x = Dropout(dropoutRate)(x)
    x = Dense(128, activation="relu", kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(64, activation="relu", kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(32, activation="relu", kernel_initializer='lecun_uniform')(x)
    
    return x


def AllBranches(orientationBranch_0, orientationBranch_1, orientationBranch_2, orientationBranch_3, dropoutRate):

    x = Concatenate()([orientationBranch_0, orientationBranch_1, orientationBranch_2, orientationBranch_3])
    x = Dense(64, activation="relu", kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(32, activation="relu", kernel_initializer='lecun_uniform')(x)
    
    return x
