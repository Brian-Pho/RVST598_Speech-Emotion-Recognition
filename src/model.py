from keras import Input, layers, models


def build_model():
    """
    Build a multi-input neural network using the Keras functional API.

    :return:
    """
    signal_shape = ()

    amplitude = Input(shape=signal_shape)
    x_a = layers.Conv2D(32, (3, 3), activation='relu')(amplitude)
    x_a = layers.MaxPooling2D((2, 2))(x_a)
    x_a = layers.Conv2D(32, (3, 3), activation='relu')(x_a)
    x_a = layers.MaxPooling2D((2, 2))(x_a)
    x_a = layers.Conv2D(32, (3, 3), activation='relu')(x_a)
    x_a = layers.MaxPooling2D((2, 2))(x_a)

    phase = Input(shape=signal_shape)
    x_p = layers.Conv2D(32, (3, 3), activation='relu')(amplitude)
    x_p = layers.MaxPooling2D((2, 2))(x_p)
    x_p = layers.Conv2D(32, (3, 3), activation='relu')(x_p)
    x_p = layers.MaxPooling2D((2, 2))(x_p)
    x_p = layers.Conv2D(32, (3, 3), activation='relu')(x_p)
    x_p = layers.MaxPooling2D((2, 2))(x_p)

    concatenated = layers.concatenate([x_a, x_p], axis=-1)

    answer = layers.Dense(answer_vocabulary_size,
                          activation='sigmoid')(concatenated)
    model = models.Model([amplitude, phase], answer)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    return model
