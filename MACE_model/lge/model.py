from keras.applications import DenseNet121
from keras import layers, models
from keras import optimizers


# model
def DenseNetmodel():
    densenet_base = DenseNet121(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    model = models.Sequential()
    densenet_base.trainable = False
    densenet_base.summary()
    model.add(densenet_base)

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.summary()
    # train

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['accuracy'])

    return model


def makemodel(basemodel):
    model = models.Sequential()
    basemodel.trainable = True
    # vgg_base.summary()
    model.add(basemodel)

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(1, activation='sigmoid'))

    # model.summary()
    # train
    return model
