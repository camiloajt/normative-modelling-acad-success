"""Functions to create the networks that compose the adversarial autoencoder."""
from tensorflow import keras

def make_encoder_model_vae(n_features, h_dim, z_dim):
    """Creates the encoder."""
    inputs = keras.Input(shape=(n_features,))
    x = inputs
    for n_neurons_layer in h_dim:
        x = keras.layers.Dense(n_neurons_layer)(x)
        x = keras.layers.LeakyReLU()(x)

    encoded = keras.layers.Dense(z_dim)(x)
    model = keras.Model(inputs=inputs, outputs=encoded)
    return model

def make_decoder_model_vae(encoded_dim, n_features, h_dim):
    """Creates the decoder."""
    encoded = keras.Input(shape=(encoded_dim,))
    x = encoded
    for n_neurons_layer in h_dim:
        x = keras.layers.Dense(n_neurons_layer)(x)
        x = keras.layers.LeakyReLU()(x)

    reconstruction = keras.layers.Dense(n_features, activation='linear')(x)
    model = keras.Model(inputs=encoded, outputs=reconstruction)
    return model