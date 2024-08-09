import tensorflow as tf
from tensorflow.keras import layers, Model

def identity_block(X, filters, kernel_size):
    F1, F2, F3 = filters
    
    X_shortcut = X
    
    X = layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid')(X)
    X = layers.BatchNormalization(axis=3)(X)
    X = layers.Activation('relu')(X)
    
    X = layers.Conv2D(filters=F2, kernel_size=kernel_size, strides=(1, 1), padding='same')(X)
    X = layers.BatchNormalization(axis=3)(X)
    X = layers.Activation('relu')(X)
    
    X = layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid')(X)
    X = layers.BatchNormalization(axis=3)(X)
    
    X = layers.add([X, X_shortcut])
    X = layers.Activation('relu')(X)
    
    return X

def ResNet(input_shape=(64, 64, 3), classes=6):
    X_input = layers.Input(input_shape)
    
    X = layers.ZeroPadding2D((3, 3))(X_input)
    
    X = layers.Conv2D(256, (7, 7), strides=(2, 2))(X)
    X = layers.BatchNormalization(axis=3)(X)
    X = layers.Activation('relu')(X)
    X = layers.MaxPooling2D((3, 3), strides=(2, 2))(X)
    
    X = identity_block(X, [64, 64, 256], 3)
    X = identity_block(X, [64, 64, 256], 3)
    
    X = layers.GlobalAveragePooling2D()(X)
    
    X = layers.Dense(classes, activation='softmax')(X)
    
    model = Model(inputs=X_input, outputs=X, name='ResNet')
    
    return model

model = ResNet(input_shape=(64, 64, 3), classes=6)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()