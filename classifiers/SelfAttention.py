import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow import keras

class PositionalEncoding(layers.Layer):
    def __init__(self, num_hiddens=1, dropout=0.5, max_len=256, **kwargs):
        super(PositionalEncoding, self).__init__()
        
        # Create a long enough `P`
        self.P = np.zeros((1, max_len, num_hiddens))
        X = np.arange(max_len).reshape(-1, 1) / np.power(max_len, np.arange(0, num_hiddens, 2) / num_hiddens)
        self.P[:, :, 0::2] = np.sin(X)
        self.P[:, :, 1::2] = np.cos(X)

        self.P = tf.cast(tf.convert_to_tensor(self.P), tf.float32)
    def get_config(self):
        return super().get_config()

    def call(self, inputs, **kwargs):
        inputs = inputs + self.P[:, :inputs.shape[1], :]
        return inputs
  


def self_attention(maxlen, d_model, num_heads):
    input_shape = (maxlen, num_heads)
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.GlobalAveragePooling1D()(inputs)
    
    # Determine the output shape after global average pooling
    output_shape = x.shape[1]
    
    # Reshape to match the desired output shape
    x = layers.Reshape((1, output_shape))(x)
    
    # Calculate padding size
    padding_size = input_shape[1] // 2 - output_shape // 2
    
    # Apply padding
    x = layers.ZeroPadding1D(padding=(padding_size, padding_size))(x)
    
    # Reshape to match the desired output shape
    # x = layers.Reshape((input_shape[0], input_shape[1], 1))(x)
    # attention part
    pos_encoding_layer = PositionalEncoding(max_len = maxlen)
    pos_encoded_inputs = pos_encoding_layer(x)
    normalization = layers.LayerNormalization(epsilon=1e-6)(pos_encoded_inputs)
    self_attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(normalization, normalization)
    x = layers.Dropout(0.5)(self_attention_output)
    res = x + inputs

    # Feed Forward part
    # x = layers.LayerNormalization(epsilon=1e-6)(res)
    # x = layers.Conv1D(filters=4, kernel_size=1, activation='relu')(x)
    # x = layers.Dropout(0.25)(x)
    # x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1, activation='relu')(x)

    x = layers.GlobalAveragePooling1D()(res)
    outputs = layers.Dense(d_model, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model