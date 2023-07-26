from typing import Callable

import tensorflow as tf

from .defines import FogParams

def encoder(
    num_heads: int = 2,
    model_dim: int = 2,
    dropout: float = 0
) -> Callable[[tf.Tensor], tf.Tensor]:
    def layer(x: tf.Tensor):
        y = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=model_dim,
            dropout=dropout
        )(query=x, key=x, value=x)
        y = tf.keras.layers.Add()([x, y])
        y = tf.keras.layers.LayerNormalization()(y)
        ys = y
        y = tf.keras.layers.Dense(model_dim)(y)
        y = tf.keras.layers.Dropout(dropout)(y)
        y = tf.keras.layers.Dense(model_dim)(y)
        y = tf.keras.layers.Dropout(dropout)(y)
        y = tf.keras.layers.Add()([ys, y])
        y = tf.keras.layers.LayerNormalization()(y)
        return y
    return layer

def fog_encoder(
    model_dim: int,
    block_size: int,
    patch_size: int,
    num_encoders: int,
    num_lstms: int,
    num_heads: int,
    batch_size: int,
    training: bool = True,
    dropout: float = 0,
):
    def layer(x: tf.Tensor) -> tf.Tensor:
        sequence_len = block_size / patch_size
        y = tf.keras.layers.Dense(model_dim)(x)
        pos_encoder = tf.tile(
            tf.Variable(
                initial_value=tf.random.normal(shape=(1, sequence_len, model_dim), stddev=0.02),
                trainable=True
            ),
            multiples=[batch_size, 1, 1]
        )
        if training:
            pos_encoder = tf.roll(
                pos_encoder,
                shift=tf.random.uniform(
                    shape=(batch_size,),
                    minval=-sequence_len,
                    maxval=0,
                    dtype=tf.int32
                ),
                axis=batch_size * [1],
            )

        y = tf.keras.layers.Add()[y, pos_encoder]
        y = tf.keras.layers.Dropout(dropout)(y)
        for _ in range(num_encoders):
            y = encoder(num_heads=num_heads, model_dim=model_dim, dropout=dropout)(y)
        for _ in range(num_lstms):
            y = tf.keras.layers.LSTM(model_dim, return_sequences=True)(y)
            y = tf.keras.layers.Bidirectional()(y)
    return layer

def fog_model(
    inputs: tf.Tensor,
    params: FogParams,
    num_classes: int
) -> tf.keras.Model:
    y = fog_encoder(
        model_dim=params.model_dim,
        block_size=params.block_size,
        patch_size=params.patch_size,
        num_encoders=params.num_encoders,
        num_lstms=params.num_lstms,
        num_heads=params.num_heads,
        batch_size=params.batch_size,
        training=params.training,
        dropout=params.dropout
    )(inputs)
    y = tf.keras.layers.Dense(num_classes)(y)
    y = tf.keras.layers.Activation(tf.keras.activations.hard_sigmoid)(y)
    model = tf.keras.Model(inputs=inputs, outputs=y)
    return model
