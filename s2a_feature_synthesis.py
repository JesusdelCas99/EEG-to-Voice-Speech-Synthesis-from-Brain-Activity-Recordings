import sys
import pandas as pd
from glob import iglob
import configparser
import numpy as np
import os
import transience.utils as utils
import transience.preprocess as preprocess
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from nnmnkwii.preprocessing import delta_features
from nnmnkwii import paramgen
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Lambda, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal


# Read configuration file
CONFIG_FILE = r'./setup.cfg'
config = configparser.ConfigParser()
config.read(CONFIG_FILE)

# Load configuration settings
_PATH_BIDS = config['SYNTHESIS']['PATH_BIDS']
_NUM_MFCC = int(config['SYNTHESIS']['NUM_MFCC'])
_SENSOR_WIN = int(config['NETWORK']['SENSOR_WIN'])
_SENSOR_PCA = float(config['NETWORK']['SENSOR_PCA'])
_LR = float(config['NETWORK']['LR'])
_DROPOUT = float(config['NETWORK']['DROPOUT'])
_DELTA_WIN = eval(config['NETWORK']['DELTA_WIN'])
_NFOLDS = int(config['NETWORK']['NFOLDS'])
_SIGMOID_ARCH = eval(config['NETWORK']['SIGMOID_ARCH'])
_VAE_ARCH = eval(config['NETWORK']['VAE_ARCH'])
_LATENT_DIM = int(config['NETWORK']['LATENT_DIM'])
_EPOCHS = int(config['NETWORK']['EPOCHS'])
_INIT_WEIGHT_MEAN = float(config['NETWORK']['INIT_WEIGHT_MEAN'])
_INIT_WEIGHT_STD = float(config['NETWORK']['INIT_WEIGHT_STD'])


class DeltaFeatures(TransformerMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X, y=None):
        pass

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X):
        return delta_features(X, _DELTA_WIN)


def create_preprocessing_pipeline_mfcc_synthesis():
    """
    Creates a preprocessing pipeline for Mel-Frequency Cepstral Coefficients (MFCC) synthesis.

    This function constructs a data processing pipeline using a list of transformation steps. The steps include:
    - Delta features extraction to capture the dynamic aspects of the audio features.
    - Standard scaling to normalize the feature values.
    - A placeholder step with a custom regressor to complete the pipeline structure. 

    Returns
    --------------------------------------
    - A scikit-learn `Pipeline` object with the specified preprocessing steps.
    """
    pipeline = []
    pipeline.append(('delta_features', DeltaFeatures()))
    pipeline.append(('standardizer', StandardScaler()))
    pipeline.append(('dummy', preprocess.DummyCustomRegressor()))
    p = Pipeline(pipeline)
    p.is_fitted_ = False
    return p



def build_sigmoid_model(input_dim, output_dim):
    """
    Builds and compiles a Keras model for classifying audio features as voiced or unvoiced.

    This model is designed for tasks involving the conversion of EEG signals into voice/unvoiced features. 
    It includes dense layers with ReLU activation and dropout for regularization, and a final output layer 
    with a sigmoid activation function to predict the probability of voiced or unvoiced features.

    Parameters:
    --------------------------------------
    - input_dim (`int`): Number of input features.
    - output_dim (`int`): Number of output units.

    Returns:
    --------------------------------------
    - `tf.keras.Model`: Compiled Keras model with defined architecture.
    """

    # Create input layer
    input_layer = Input(shape=(input_dim,))
    x = input_layer

    # Add dense layers with ReLU activation and dropout
    for num_units in _SIGMOID_ARCH:
        x = Dense(num_units, activation="relu")(x)
        x = Dropout(_DROPOUT)(x)
    
    # Add output layer with sigmoid activation
    output_layer = Dense(output_dim, activation='sigmoid')(x)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    # Configure optimizer and compile model
    optimizer = tf.keras.optimizers.SGD(learning_rate=_LR, clipnorm=1.0, momentum=0.9)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=[tf.keras.metrics.Accuracy()])
    
    return model



def gaussian_nll(y_true, y_pred):
    """
    Keras implmementation of multivariate Gaussian negative loglikelihood loss function.
    This implementation implies diagonal covariance matrix.

    Parameters
    ----------
    - ytrue (`tf.Tensor`): Ground truth values with shape `[n_samples, n_dims]`.
    - ypreds (`tf.Tensor`): Predicted mu and log_var values with shape `[n_samples, n_dims*2]`.

    Returns
    ------------
    - `float`: Negative loglikelihood averaged over samples (per batch).
    """
    reg_val = tf.constant(1e-3, dtype=tf.float32)
    ihalf = tf.constant(0.5, dtype=tf.float32)
    l2pi = tf.constant(np.log(2.0 * np.pi), dtype=tf.float32)

    n_dims = tf.shape(y_pred)[1] // 2
    mu = y_pred[:, 0:n_dims]
    var = tf.math.exp(y_pred[:, n_dims:]) + reg_val
    log_std = tf.math.log(tf.math.sqrt(var))

    log2pi = tf.cast(n_dims, tf.float32) * ihalf * l2pi
    log_sigma_trace = tf.math.reduce_sum(log_std, axis=1)
    mse = ihalf * tf.math.reduce_sum(tf.math.square((y_true - mu) / tf.math.exp(log_std)), axis = 1)
    
    log_likelihood = log2pi + log_sigma_trace + mse

    return log_likelihood



def freeze_decoder(model):
    """
    Freeze all layers in a model to prevent weight updates during training.

    Parameters:
    --------------------------------------
    - model (`tf.keras.Model`): Model with layers to freeze.

    Returns:
    --------------------------------------
    - `tf.keras.Model`: Input model with all layers set as non-trainable.
    """
    
    # Iterate through layers in the model
    for layer in model.layers:
        # Set 'trainable' attribute to False
        layer.trainable = False
    
    # Return model with frozen layers
    return model


@tf.function
def sampling(args):
    """
    Reparameterization trick for sampling in a Variational Autoencoder (VAE).

    This function generates a sample from a multivariate Gaussian distribution 
    using the provided mean (`z_mean`) and log variance (`z_log_var`). It implements 
    the reparameterization trick, which allows gradients to flow through the 
    stochastic sampling process during backpropagation in a VAE.

    Parameters:
    --------------------------------------
    - args: A tuple containing:
        - `z_mean` (`tf.Tensor`): The mean of the latent Gaussian distribution.
        - `z_log_var` (`tf.Tensor`): The log variance of the latent Gaussian distribution.

    Returns:
    --------------------------------------
    - `tf.Tensor`: Sampled latent vector.
    """
    # Extract mean and log variance from the input arguments
    z_mean, z_log_var = args

    # Check for NaN or Inf in z_mean and z_log_var
    z_mean = tf.debugging.check_numerics(z_mean, "NaN or Inf found in z_mean")
    z_log_var = tf.debugging.check_numerics(z_log_var, "NaN or Inf found in z_log_var")

    # Get the batch size from the shape of the mean tensor
    batch = tf.shape(z_mean)[0]
    # Get the dimensionality of the latent space
    dim = tf.shape(z_mean)[1]
    # Generate random normal noise with the same shape as the mean tensor
    epsilon = tf.random.normal(shape=(batch, dim))
    # Sample from the Gaussian distribution using the reparameterization trick
    z = z_mean + tf.exp(0.5 * z_log_var) * epsilon

    # Check for NaN or Inf in the sampled latent vector z
    z = tf.debugging.check_numerics(z, "NaN or Inf found in z")

    return z



def build_audio_to_audio_features_vae(input_dim, output_dim):
    """
    Constructs a Variational Autoencoder (VAE) model for audio feature extraction. 
    This function creates both the encoder and decoder parts of the VAE as multi-layer 
    perceptrons (MLPs) with ReLU activation functions. The VAE model is designed to 
    process audio data and encode it into a latent space representation, which is then 
    decoded back to the original audio feature space. The model includes dropout for 
    regularization and uses a Gaussian Negative Log Likelihood (NLL) loss function 
    for reconstruction and a KL divergence loss function for regularization.
    
    Parameters:
    --------------------------------------
    - input_dim (`int`): Dimensionality of the input data.
    - output_dim (`int`): Dimensionality of the output data.

    Returns:
    --------------------------------------
    - `tf.keras.Model`: The compiled VAE model.
    - `tf.keras.Model`: The encoder part of the VAE.
    - `tf.keras.Model`: The decoder part of the VAE.
    """
    # Encoder
    input_layer = Input(shape=(input_dim,), name = 'a2a_enc_input')
    x_enc = input_layer
    for k, h_dim in enumerate(_VAE_ARCH):
        x_enc = Dense(h_dim, activation='relu', name = f'a2a_enc_dense_{k}', kernel_initializer = RandomNormal(mean=_INIT_WEIGHT_MEAN, stddev=_INIT_WEIGHT_STD))(x_enc)
        x_enc = Dropout(_DROPOUT, name = f'a2a_enc_dropout_{k}')(x_enc)  # Added dropout rate for regularization

    # Latent space
    z_mean = Dense(_LATENT_DIM, name='a2a_enc_z_mean', kernel_initializer = RandomNormal(mean=_INIT_WEIGHT_MEAN, stddev=_INIT_WEIGHT_STD))(x_enc)
    z_log_var = Dense(_LATENT_DIM, name='a2a_enc_z_log_var', kernel_initializer = RandomNormal(mean=_INIT_WEIGHT_MEAN, stddev=_INIT_WEIGHT_STD))(x_enc)
    z = Lambda(sampling, output_shape=(_LATENT_DIM,), name='a2a_enc_z')([z_mean, z_log_var])

    # Encoder model
    encoder = Model(inputs=input_layer, outputs=[z_mean, z_log_var, z], name = 'a2a_enc_model')

    # Decoder
    latent_inputs = Input(shape=(_LATENT_DIM,), name = 'a2a_dec_input')
    x_dec = latent_inputs
    for k, h_dim in enumerate(reversed(_VAE_ARCH)):
        x_dec = Dense(h_dim, activation='relu', name = f'a2a_dec_dense_{k}', kernel_initializer = RandomNormal(mean=_INIT_WEIGHT_MEAN, stddev=_INIT_WEIGHT_STD))(x_dec)
        x_dec = Dropout(_DROPOUT, name = f'a2a_dec_dropout_{k}')(x_dec)  # Added dropout rate for regularization

    # Output layer
    output_layer = Dense(output_dim, activation='linear', name = 'a2a_dec_output', kernel_initializer = RandomNormal(mean=_INIT_WEIGHT_MEAN, stddev=_INIT_WEIGHT_STD))(x_dec)
    # Decoder model
    decoder = Model(inputs=latent_inputs, outputs=output_layer, name = 'a2a_dec_model')

    # VAE model
    vae_output = decoder(encoder(input_layer)[2])  # Use only the sampled z as input to the decoder
    full_output = tf.keras.layers.Concatenate()([vae_output, z_mean, z_log_var])
    vae = Model(inputs=input_layer, outputs=full_output, name='a2a_vae_model')

    # Total VAE loss
    @tf.function
    def vae_loss(y_true, y_pred):
        # Extract components from y_pred
        vae_output = y_pred[:, :-2 * _LATENT_DIM]  # VAE output [batch_size x 100]
        z_mean = y_pred[:, -2 * _LATENT_DIM: -_LATENT_DIM]  # Mean [batch_size x 16]
        z_log_var = y_pred[:, -_LATENT_DIM:]  # Log variance [batch_size x 16]

        # Reconstruction loss (batch_size,)
        reconstruction_loss = gaussian_nll(y_true, vae_output)
        # KL divergence loss (batch_size,)
        kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        return tf.reduce_mean(reconstruction_loss + kl_loss)

    # Add custom loss to the model and compile it
    optimizer = tf.keras.optimizers.SGD(learning_rate=_LR, clipnorm=1.0, momentum=0.9)
    vae.compile(optimizer=optimizer, loss=vae_loss)

    return vae, encoder, decoder


def build_sensor_to_audio_features_vae(input_dim, output_dim, pretrained_decoder):
    """
    Constructs a Variational Autoencoder (VAE) model for sensor data to audio feature extraction,
    utilizing a pre-trained decoder model. This function focuses on training the encoder part
    of the VAE while using the given pre-trained decoder to perform the reconstruction of 
    audio features from the latent space.

    Parameters:
    --------------------------------------
    - input_dim (`int`): Dimensionality of the sensor input data.
    - output_dim (`int`): Dimensionality of the audio feature output data.
    - pretrained_decoder (`tf.keras.Model`): Pre-trained decoder model used for the reconstruction of audio features.

    Returns:
    --------------------------------------
    - `tf.keras.Model`: The compiled VAE model with the pre-trained decoder and a newly built encoder.
    """
    # Encoder
    input_layer = Input(shape=(input_dim,), name = 's2a_enc_input')
    x = input_layer
    for k, h_dim in enumerate(_VAE_ARCH):
        x = Dense(h_dim, activation='relu', name = f's2a_enc_dense_{k}', kernel_initializer = RandomNormal(mean=_INIT_WEIGHT_MEAN, stddev=_INIT_WEIGHT_STD))(x)
        x = Dropout(_DROPOUT, name = f's2a_enc_dropout_{k}')(x)  # Added dropout rate for regularization

    # Latent space
    z_mean = Dense(_LATENT_DIM, name='s2a_enc_z_mean', kernel_initializer = RandomNormal(mean=_INIT_WEIGHT_MEAN, stddev=_INIT_WEIGHT_STD))(x)
    z_log_var = Dense(_LATENT_DIM, name='s2a_enc_z_log_var', kernel_initializer = RandomNormal(mean=_INIT_WEIGHT_MEAN, stddev=_INIT_WEIGHT_STD))(x)
    z = Lambda(sampling, output_shape=(_LATENT_DIM,), name='s2a_enc_z')([z_mean, z_log_var])

    # Encoder model
    encoder = Model(inputs = input_layer, outputs = [z_mean, z_log_var, z], name = 's2a_enc_model')

    # Define the VAE model using the pretrained decoder
    vae_output = pretrained_decoder(encoder(input_layer)[2])  # Use only the sampled z as input to the decoder
    full_output = tf.keras.layers.Concatenate()([vae_output, z_mean, z_log_var])
    vae = Model(inputs=input_layer, outputs = full_output, name = 's2a_vae_model')

    # Total VAE loss
    @tf.function
    def vae_loss(y_true, y_pred):
        # Extract components from y_pred
        vae_output = y_pred[:, :-2 * _LATENT_DIM]  # VAE output [batch_size x 100]
        z_mean = y_pred[:, -2 * _LATENT_DIM: -_LATENT_DIM]  # Mean [batch_size x 16]
        z_log_var = y_pred[:, -_LATENT_DIM:]  # Log variance [batch_size x 16]

        # Reconstruction loss (batch_size,)
        reconstruction_loss = gaussian_nll(y_true, vae_output)
        # KL divergence loss (batch_size,)
        kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        return tf.reduce_mean(reconstruction_loss + kl_loss)

    # Add custom loss to the model and compile it
    optimizer = tf.keras.optimizers.SGD(learning_rate=_LR, clipnorm=1.0, momentum=0.9)
    vae.compile(optimizer=optimizer, loss=vae_loss)

    return vae



def train_models(sensor_feat_train, sensor_feat_val, audio_feat_train, audio_feat_val):
    """
    Define and train models for mapping features between audio and sensor data.

    - For MFCC, BAP, and LF0 features, the function builds and trains two Variational Autoencoders (VAEs):
      1. An audio-to-audio VAE that maps audio features to audio features.
      2. A sensor-to-audio VAE that uses the decoder from the first VAE to map sensor features to audio features.

    - For VUV (Voiced/Unvoiced) features, the function builds and trains a simple sigmoid classification model.

    Parameters:
    --------------------------------------
    - sensor_feat_train (numpy.ndarray): Training data for sensor features.
    - sensor_feat_val (numpy.ndarray): Validation data for sensor features.
    - audio_feat_train (numpy.ndarray): Training data for audio features.
    - audio_feat_val (numpy.ndarray): Validation data for audio features.

    Returns:
    --------------------------------------
    - models (dict): Dictionary of trained models for each feature type ('mfcc', 'bap', 'lf0', 'vuv').
    """
    # Set seeds for reproducibility to ensure consistent results across different runs
    np.random.seed(1234)
    tf.random.set_seed(1234)

    # Dictionary to hold the trained models
    models = {}

    # Callback to stop training early if validation loss does not improve for 15 epochs
    early_stop_callback = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )

    for feat_type in ['mfcc', 'bap', 'lf0', 'vuv']:
        # Log the current feature type being processed
        print(f'*****Training models for {feat_type} features...')
        print('------------------------------------------------')
        if feat_type != 'vuv':
            print(f'*****Training VAE model with {feat_type} features for audio-to-audio transformation...')
        # Prepare target datasets and define model parameters based on the feature type
        if feat_type == 'mfcc':
            targets_train = np.hstack([audio_feat_train[:, :_NUM_MFCC], audio_feat_train[:, (_NUM_MFCC + 2):(_NUM_MFCC + 2) + _NUM_MFCC]])
            targets_val = np.hstack([audio_feat_val[:, :_NUM_MFCC], audio_feat_val[:, (_NUM_MFCC + 2):(_NUM_MFCC + 2) + _NUM_MFCC]])
            input_dim = targets_train.shape[1]
            output_dim = 2 * len(_DELTA_WIN) * _NUM_MFCC
        elif feat_type == 'bap':
            targets_train = np.hstack([audio_feat_train[:, _NUM_MFCC][:, np.newaxis], audio_feat_train[:, (_NUM_MFCC + 1) * 2][:, np.newaxis]])
            targets_val = np.hstack([audio_feat_val[:, _NUM_MFCC][:, np.newaxis], audio_feat_val[:, (_NUM_MFCC + 1) * 2][:, np.newaxis]])
            input_dim = targets_train.shape[1]
            output_dim = 2 * len(_DELTA_WIN)
        elif feat_type == 'lf0':
            targets_train = np.hstack([audio_feat_train[:, _NUM_MFCC + 1][:, np.newaxis], audio_feat_train[:, (_NUM_MFCC + 2) * 2 - 1][:, np.newaxis]])
            targets_val = np.hstack([audio_feat_val[:, _NUM_MFCC + 1][:, np.newaxis], audio_feat_val[:, (_NUM_MFCC + 2) * 2 - 1][:, np.newaxis]])
            input_dim = targets_train.shape[1]
            output_dim = 2 * len(_DELTA_WIN)
        
        # Build and compile the VAE model for audio-to-audio feature mapping
        if feat_type in ['mfcc', 'bap', 'lf0']:
            dnn, _, decoder = build_audio_to_audio_features_vae(input_dim, output_dim)
            dnn.fit(targets_train, targets_train,
                    batch_size=128,
                    epochs=_EPOCHS,
                    shuffle=False,
                    verbose=2,
                    validation_data=(targets_val, targets_val),
                    callbacks=[early_stop_callback])
            # Freeze the decoder part of the model to prevent it from being updated during sensor-to-audio mapping
            frozen_decoder = freeze_decoder(decoder)

        # Define the input dimension for the sensor-to-audio feature mapping VAE 
        input_dim = sensor_feat_train.shape[1]
        
        # Choose model-building function based on feature type for sensor-to-audio VAE
        if feat_type == 'mfcc' or feat_type == 'bap' or feat_type == 'lf0':
            build_fn = build_sensor_to_audio_features_vae
        elif feat_type == 'vuv':
            # For VUV (Voiced/Unvoiced), set targets and use a sigmoid model for binary classification
            targets_train = audio_feat_train[:, -1]
            targets_val = audio_feat_val[:, -1]
            output_dim = 1
            build_fn = build_sigmoid_model
        

        # Build and compile the VAE model for sensor-to-audio feature mapping
        if feat_type == 'vuv':
            # Use a simple sigmoid model for VUV features
            dnn = build_fn(input_dim, output_dim)
        else:
            # Use VAE with a frozen decoder for MFCC, BAP, and LF0 features
            dnn = build_fn(input_dim, output_dim, frozen_decoder)

        # Train the model
        print(f'*****Training VAE model with {feat_type} features for sensor-to-audio transformation...')
        dnn.fit(sensor_feat_train, targets_train,
                batch_size=128,
                epochs=_EPOCHS,
                shuffle=False,
                verbose=2,
                validation_data=(sensor_feat_val, targets_val),
                callbacks=[early_stop_callback])
        

        # Save the trained VAE model in the 'models' dictionary, indexed by feature type
        models[feat_type] = dnn

    return models



def predict_speech_features(models, x, mu, sigma):
    """
    Predicts speech features (MFCC, BAP, LF0, VUV) for the given input sample using trained models and applies
    statistical denormalization and maximum likelihood parameter generation (MLPG) for smoother outputs.

    Parameters:
    --------------------------------------
    - models (`dict`): Dictionary of trained models for each feature type. 
                       Keys should include 'mfcc', 'bap', 'lf0', and 'vuv'.
    - x (`numpy.ndarray`): Input sensor features for prediction.
    - mu (`numpy.ndarray`): Mean values for denormalization of predicted features.
    - sigma (`numpy.ndarray`): Standard deviation values for denormalization of predicted features.

    Returns:
    --------------------------------------
    - `list[numpy.ndarray]`: A list of predicted features for each type ('mfcc', 'bap', 'lf0', 'vuv').
    """
    features = []

    # Iterate over each feature type to make predictions and process them
    for feat_type in ['mfcc', 'bap', 'lf0', 'vuv']:
        # Generate predictions using the corresponding model
        if feat_type == 'vuv':
            pred = models[feat_type].predict(x)
        else: 
            pred = models[feat_type].predict(x)[:, :-2 * _LATENT_DIM]
        
        # Initialize mean and standard deviation based on the feature type
        if feat_type == 'mfcc':
            # # Extract mean and standard deviation for MFCC features
            m = np.concatenate((mu[0:_NUM_MFCC].flatten(), mu[_NUM_MFCC+2:_NUM_MFCC*2+2].flatten()))
            s = np.concatenate((sigma[0:_NUM_MFCC].flatten(), sigma[_NUM_MFCC + 2:_NUM_MFCC*2+2].flatten()))
        elif feat_type == 'bap':
            # Extract mean and standard deviation for BAP features
            m = np.array([mu[_NUM_MFCC], mu[_NUM_MFCC*2+2]])
            s = np.array([sigma[_NUM_MFCC], sigma[_NUM_MFCC*2+2]])
        elif feat_type == 'lf0':
            # Extract mean and standard deviation for LF0 features
            m = np.array([mu[_NUM_MFCC+1], mu[_NUM_MFCC*2+3]])
            s = np.array([sigma[_NUM_MFCC+1], sigma[_NUM_MFCC*2+3]])
        elif feat_type == 'vuv':
            # For VUV (Voiced/Unvoiced flag), round the predictions to binary values (0 or 1)
            features.append(np.round(pred))
            continue
        
        # Denormalize predictions
        y = m[np.newaxis,:] + s[np.newaxis,:] * pred[:,0:(pred.shape[1]//2)]

        # Compute variance for MLPG application
        y_var = s[np.newaxis,:]**2 * np.exp(pred[:,(pred.shape[1]//2):])
        
        # Apply MLPG to denormalize and smooth predictions, then append to features list
        y = paramgen.mlpg(y, y_var, _DELTA_WIN)
        features.append(y)

    return features



if __name__ == '__main__':
    
    # Number of folds for cross-validation
    nfolds = _NFOLDS

    # Initialize K-Fold cross-validation with the specified number of folds
    kf = KFold(n_splits=nfolds, shuffle=False)

    # Load metadata of participants from the BIDS dataset
    participants = pd.read_csv(os.path.join(_PATH_BIDS, 'participants.tsv'), delimiter='\t')
    # participants = pd.DataFrame({'participant_id': ['M06']})

    # Iterate over each participant in the dataset
    for p_id, participant in enumerate(participants['participant_id']):
        
        # Construct the path to the current participant's directory
        participant_dir = os.path.join(_PATH_BIDS, f'{participant}')

        # Load raw sensor and acoustic features for the participant
        basename = [os.path.basename(f) for f in iglob(os.path.join(participant_dir, 'hg_feat', '*.npy'))]
        sensor_raw_data = utils.load_numpy_dataset(os.path.join(participant_dir, 'hg_feat'), basename)
        acoustic_raw_data = utils.load_numpy_dataset(os.path.join(participant_dir, 'audio_feat'), basename)

        # Split the data into training and testing sets (30% for testing)
        dataset_indices = np.arange(len(acoustic_raw_data))
        train_indices, test_indices = train_test_split(dataset_indices, test_size=0.3, random_state=42)
        train_acoustic = [acoustic_raw_data[i] for i in train_indices] 
        train_sensor = [sensor_raw_data[i] for i in train_indices]
        test_sensor = [sensor_raw_data[i] for i in test_indices]

        # Perform K-Fold cross-validation on the training sets
        for k, (train, validation) in enumerate(kf.split(train_acoustic)):
            
            # Log participant ID and current fold number
            print(f'Participant {participant}, Fold #{k}')
            print('*****Loading the data...')

            # Initialize preprocessing pipelines for sensor and acoustic data
            sensor_pipeline = preprocess.create_preprocessing_pipeline_sensor(_SENSOR_WIN, _SENSOR_PCA)
            acoustic_pipeline = create_preprocessing_pipeline_mfcc_synthesis()
           
            # Split training data into training and validation subsets for the current fold
            train_sensor_data = [train_sensor[i] for i in train]
            validation_sensor_data = [train_sensor[i] for i in validation]
            train_acoustic_data = [train_acoustic[i] for i in train]
            validation_acoustic_data = [train_acoustic[i] for i in validation]

            # Apply preprocessing pipelines to the sensor data
            train_sensor_data = preprocess.preprocess_data(sensor_pipeline, train_sensor_data)
            validation_sensor_data = preprocess.preprocess_data(sensor_pipeline, validation_sensor_data)
            
            # Extract VUV (Voice/Unvoiced) features from the acoustic data
            train_vuv = np.concatenate([y[:, -1] for y in train_acoustic_data])[:, np.newaxis]
            validation_vuv = np.concatenate([y[:, -1] for y in validation_acoustic_data])[:, np.newaxis]

            # Apply preprocessing pipelines to the acoustic data (excluding VUV features)
            train_acoustic_data = preprocess.preprocess_data(acoustic_pipeline, [y[:, :-1] for y in train_acoustic_data])
            validation_acoustic_data = preprocess.preprocess_data(acoustic_pipeline, [y[:, :-1] for y in validation_acoustic_data])
            
            # Stack the processed training and validation data
            X_train = np.vstack(train_sensor_data)
            X_val = np.vstack(validation_sensor_data)
            Y_train = np.hstack([np.vstack(train_acoustic_data), train_vuv])
            Y_val = np.hstack([np.vstack(validation_acoustic_data), validation_vuv])

            """
            print(f"Maximo X_train: {np.max(X_train)}")
            print(f"Maximo X_train: {np.max(X_val)}")
            print(f"Maximo X_train: {np.max(Y_train)}")
            print(f"Maximo X_train: {np.max(Y_val)}")
            
            # Verify if there are any NaN values in the arrays
            if np.isnan(X_train).any():
                print("NaN values found in X_train")
            else:
                print("No NaN values in X_train")

            if np.isnan(X_val).any():
                print("NaN values found in X_val")
            else:
                print("No NaN values in X_val")

            if np.isnan(Y_train).any():
                print("NaN values found in Y_train")
            else:
                print("No NaN values in Y_train")

            if np.isnan(Y_val).any():
                print("NaN values found in Y_val")
            else:
                print("No NaN values in Y_val")
            """
            print(Y_train.shape)
            print(Y_val.shape)
            print(X_train.shape)
            print(X_val.shape)
            
            # Train the models using the preprocessed training data
            models = train_models(X_train, X_val, Y_train, Y_val)

            # Log the start of test inference
            print('*****Test inference...')

            # Create directory to store predictions from the current fold
            predictions_dir = os.path.join(participant_dir, 'validation', f'fold_{k}')
            os.makedirs(predictions_dir, exist_ok=True)

            # Retrieve mean and scale values from the standardizer in the acoustic pipeline
            mu = acoustic_pipeline['standardizer'].mean_
            std = acoustic_pipeline['standardizer'].scale_

            # Iterate over each test sensor data sample
            for i in range(len(test_sensor)):

                # Extract the base filename (without extension) for the current test sample
                fname = os.path.splitext(basename[test_indices[i]])[0]

                # Log the processing of the current test sample
                print(f'Processing file {fname}')

                # Preprocess the test sensor data for the current sample
                x = preprocess.preprocess_data(sensor_pipeline, test_sensor[i])

                # Predict speech features using the trained models and save the results
                y_predicted = predict_speech_features(models, x, mu, std)
                np.save(os.path.join(predictions_dir, f"{fname}.npy"), np.hstack(y_predicted))
