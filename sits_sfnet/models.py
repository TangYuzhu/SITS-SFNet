"""Architectures retained from the research source; encoder dropout stays in inference mode as in predict_step."""

import tensorflow as tf

NUM_BANDS = 16


def build_unet(input_size, output_channels):
    inputs = tf.keras.layers.Input(input_size)
    conv1 = tf.keras.layers.Conv2D(
        64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(inputs)
    conv1 = tf.keras.layers.Conv2D(
        64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = tf.keras.layers.Conv2D(
        128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(pool1)
    conv2 = tf.keras.layers.Conv2D(
        128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = tf.keras.layers.Conv2D(
        256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(pool2)
    conv3 = tf.keras.layers.Conv2D(
        256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = tf.keras.layers.Conv2D(
        512, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(pool3)
    conv4 = tf.keras.layers.Conv2D(
        512, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv4)
    drop4 = tf.keras.layers.Dropout(0.5)(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)
    conv5 = tf.keras.layers.Conv2D(
        1024, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(pool4)
    conv5 = tf.keras.layers.Conv2D(
        1024, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv5)
    drop5 = tf.keras.layers.Dropout(0.5)(conv5)
    up6 = tf.keras.layers.Conv2D(
        512, 2, activation="relu", padding="same", kernel_initializer="he_normal"
    )(tf.keras.layers.UpSampling2D(size=(2, 2))(drop5))
    merge6 = tf.keras.layers.concatenate([drop4, up6], axis=3)
    conv6 = tf.keras.layers.Conv2D(
        512, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(merge6)
    conv6 = tf.keras.layers.Conv2D(
        512, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv6)
    up7 = tf.keras.layers.Conv2D(
        256, 2, activation="relu", padding="same", kernel_initializer="he_normal"
    )(tf.keras.layers.UpSampling2D(size=(2, 2))(conv6))
    merge7 = tf.keras.layers.concatenate([conv3, up7], axis=3)
    conv7 = tf.keras.layers.Conv2D(
        256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(merge7)
    conv7 = tf.keras.layers.Conv2D(
        256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv7)
    up8 = tf.keras.layers.Conv2D(
        128, 2, activation="relu", padding="same", kernel_initializer="he_normal"
    )(tf.keras.layers.UpSampling2D(size=(2, 2))(conv7))
    merge8 = tf.keras.layers.concatenate([conv2, up8], axis=3)
    conv8 = tf.keras.layers.Conv2D(
        128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(merge8)
    conv8 = tf.keras.layers.Conv2D(
        128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv8)
    up9 = tf.keras.layers.Conv2D(
        64, 2, activation="relu", padding="same", kernel_initializer="he_normal"
    )(tf.keras.layers.UpSampling2D(size=(2, 2))(conv8))
    merge9 = tf.keras.layers.concatenate([conv1, up9], axis=3)
    conv9 = tf.keras.layers.Conv2D(
        64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(merge9)
    conv9 = tf.keras.layers.Conv2D(
        64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv9)
    conv10 = tf.keras.layers.Conv2D(
        output_channels, 3, activation="softmax", padding="same", kernel_initializer="he_normal"
    )(conv9)
    model = tf.keras.Model(inputs=inputs, outputs=conv10)
    return model


def build_encoder(inputs):
    avg_pool = tf.keras.layers.GlobalAvgPool2D()(inputs)
    avg_pool = tf.keras.layers.Dense(
        NUM_BANDS // 4, kernel_initializer="he_normal", activation="relu"
    )(avg_pool)
    avg_pool = tf.keras.layers.Dense(NUM_BANDS, kernel_initializer="he_normal", activation="relu")(
        avg_pool
    )
    max_pool = tf.keras.layers.GlobalMaxPooling2D()(inputs)
    max_pool = tf.keras.layers.Dense(
        NUM_BANDS // 4, kernel_initializer="he_normal", activation="relu"
    )(max_pool)
    max_pool = tf.keras.layers.Dense(NUM_BANDS, kernel_initializer="he_normal", activation="relu")(
        max_pool
    )
    cbam_feature = tf.keras.layers.Add()([avg_pool, max_pool])
    cbam_feature = tf.keras.layers.Activation("hard_sigmoid")(cbam_feature)
    x = tf.keras.layers.Multiply()([inputs, cbam_feature])
    conv1 = tf.keras.layers.Conv2D(
        32, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(x)
    conv1 = tf.keras.layers.Conv2D(
        32, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = tf.keras.layers.Conv2D(
        64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(pool1)
    conv2 = tf.keras.layers.Conv2D(
        64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = tf.keras.layers.Conv2D(
        128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(pool2)
    conv3 = tf.keras.layers.Conv2D(
        128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = tf.keras.layers.Conv2D(
        256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(pool3)
    conv4 = tf.keras.layers.Conv2D(
        256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv4)
    drop4 = tf.keras.layers.Dropout(0.5)(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)
    conv5 = tf.keras.layers.Conv2D(
        512, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(pool4)
    conv5 = tf.keras.layers.Conv2D(
        512, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv5)
    drop5 = tf.keras.layers.Dropout(0.5)(conv5)
    model = tf.keras.Model(inputs=inputs, outputs=(conv1, conv2, conv3, conv4, drop5))
    return model


def build_auxiliary_decoder(inputs, mask):
    c_1, c_2, c_3, c_4, c_5 = inputs
    mask1 = tf.keras.layers.AveragePooling2D((2, 2))(mask)
    mask2 = tf.keras.layers.AveragePooling2D((4, 4))(mask)
    mask3 = tf.keras.layers.AveragePooling2D((8, 8))(mask)
    o_5 = tf.keras.layers.UpSampling2D((2, 2))(c_5)
    x = tf.keras.layers.concatenate([c_4, o_5, mask3], axis=3)
    x = tf.keras.layers.Conv2D(
        256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(x)
    x = tf.keras.layers.Conv2D(
        256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.concatenate([c_3, x, mask2], axis=3)
    x = tf.keras.layers.Conv2D(
        128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(x)
    x = tf.keras.layers.Conv2D(
        128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.concatenate([c_2, x, mask1], axis=3)
    x = tf.keras.layers.Conv2D(
        64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(x)
    x = tf.keras.layers.Conv2D(
        64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.concatenate([c_1, x, mask], axis=3)
    x = tf.keras.layers.Conv2D(
        32, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(x)
    x = tf.keras.layers.Conv2D(
        32, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(x)
    return x


def build_plain_decoder(inputs):
    c_1, c_2, c_3, c_4, c_5 = inputs
    o_5 = tf.keras.layers.UpSampling2D((2, 2))(c_5)
    x = tf.keras.layers.concatenate([c_4, o_5], axis=3)
    x = tf.keras.layers.Conv2D(
        256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(x)
    x = tf.keras.layers.Conv2D(
        256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.concatenate([c_3, x], axis=3)
    x = tf.keras.layers.Conv2D(
        128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(x)
    x = tf.keras.layers.Conv2D(
        128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.concatenate([c_2, x], axis=3)
    x = tf.keras.layers.Conv2D(
        64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(x)
    x = tf.keras.layers.Conv2D(
        64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.concatenate([c_1, x], axis=3)
    x = tf.keras.layers.Conv2D(
        32, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(x)
    x = tf.keras.layers.Conv2D(
        32, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(x)
    return x


def build_sits_sfnet(input_size, output_channels):
    inputs = tf.keras.layers.Input(input_size)
    x1 = inputs[..., 0:NUM_BANDS]
    x2 = inputs[..., NUM_BANDS : NUM_BANDS * 2]
    x3 = inputs[..., NUM_BANDS * 2 : NUM_BANDS * 3]
    mask = inputs[..., NUM_BANDS * 3 :]
    encoder = build_encoder(tf.keras.layers.Input((256, 256, NUM_BANDS)))
    o1_1, o1_2, o1_3, o1_4, o1_5 = encoder(x1, training=False)
    o2_1, o2_2, o2_3, o2_4, o2_5 = encoder(x2, training=False)
    o3_1, o3_2, o3_3, o3_4, o3_5 = encoder(x3, training=False)
    o_1 = tf.keras.layers.concatenate([o1_1, o2_1, o3_1], axis=3)
    o_2 = tf.keras.layers.concatenate([o1_2, o2_2, o3_2], axis=3)
    o_3 = tf.keras.layers.concatenate([o1_3, o2_3, o3_3], axis=3)
    o_4 = tf.keras.layers.concatenate([o1_4, o2_4, o3_4], axis=3)
    o_5 = tf.keras.layers.concatenate([o1_5, o2_5, o3_5], axis=3)
    left_output = (o_1, o_2, o_3, o_4, o_5)
    right_output = build_auxiliary_decoder(left_output, mask)
    outputs = tf.keras.layers.Conv2D(
        output_channels, 3, activation="softmax", padding="same", kernel_initializer="he_normal"
    )(right_output)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
