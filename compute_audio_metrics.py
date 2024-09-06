from pystoi import stoi
from pesq import pesq
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import mean_squared_error
import configparser
import pandas as pd
import os
from glob import iglob
import sys
import numpy as np
from transience import pyworld as pwm
import pyworld as pw
import json


# Read configuration file
CONFIG_FILE = r'./setup.cfg'
config = configparser.ConfigParser()
config.read(CONFIG_FILE)

# Load configuration settings
_PATH_BIDS = config['SYNTHESIS']['PATH_BIDS']
_NUM_MFCC = int(config['SYNTHESIS']['NUM_MFCC'])
_FS = int(config['SYNTHESIS']['TARGET_SR'])
_FFTLEN = int(config['NETWORK']['FFTLEN'])
_FRAME_PERIOD = float(config['SYNTHESIS']['FRAMESHIFT'])


def melcd(X, Y):
    """
    Computes the Mean Mel-Cepstrum Distortion (MCD) for time-aligned mel-cepstrum sequences.

    Parameters:
    --------------------------------------
    - X (`np.ndarray`): Array of target Mel-cepstrum features with shape (T, D), where:
        - T is the number of timeframes, indicating the temporal length of the sequence.
        - D is the number of Mel Frequency Cepstral Coefficients (MFCCs), representing the dimensionality of each timeframe.
    - Y (`np.ndarray`): Array of reference Mel-cepstrum features with shape (T, D), as in (T, D).

    Returns:
    --------------------------------------
    - `float`: Mean mel-cepstrum distortion in dB.

    Raises:
    --------------------------------------
    - `ValueError`: If the shapes of the input arrays do not match.
    """

    # Check if the shapes of the input arrays match
    if X.shape == Y.shape:
        # Scaling factor for distortion calculation
        scaling_factor = 10 * np.sqrt(2) / np.log(10)
        # Number of frames (or time steps)
        T = X.shape[0] 
        # Calculate MCD using the Euclidean distance between corresponding frames
        melcd = scaling_factor * (1/T) * np.sum(np.sqrt(np.sum(np.square(X - Y), axis=1)))
    else:
        raise ValueError(f"The shapes of the input arrays {X.shape} and {Y.shape} do not match.")

    return melcd



def bap_distortion(X, Y):
    """ 
    Computes the Band Aperiodicities Distortion (BAPD) for time-aligned band aperiodicity sequences.

    Parameters:
    --------------------------------------
    - X (`np.ndarray`): Array of target band aperiodicity features with shape (T, D), where:
        - T is the number of timeframes, indicating the temporal length of the sequence.
        - D is the number of frequency bands, representing the dimensionality of each timeframe.
    - Y (`np.ndarray`): Array of reference band aperiodicity features with shape (T, D), as in (T, D).

    Returns:
    --------------------------------------
    - `float`: Band aperiodicity distortion score in dB.

    Raises:
    --------------------------------------
    - `ValueError`: If the shapes of the input arrays do not match.
    """

    # Check if the shapes of the input arrays match
    if X.shape == Y.shape:
        # Scaling factor for distortion calculation
        scaling_factor = 10 * np.sqrt(2) / np.log(10)
        # Number of frames (or time steps)
        num_frames = X.shape[0]  
        # Calculate Band Aperiodicity Distortion using the Euclidean distance between corresponding frames
        bap_distortion = scaling_factor * (1 / num_frames) * np.sum(np.sqrt(np.square(X - Y)))
    else:
        raise ValueError(f"The shapes of the input arrays {X.shape} and {Y.shape} do not match.")

    return bap_distortion



def calculate_melcd(clean_features, predicted_features):
    """
    Calculate the Mel-Cepstral Distortion (MCD) for each sample 
    between clean and predicted audio features.

    Parameters:
    --------------------------------------
    - clean_features (`list of np.ndarray`): List of arrays containing the clean audio features for each sample.
    - predicted_features (`list of np.ndarray`): List of arrays containing the predicted audio features for each sample.

    Returns:
    --------------------------------------
    - `list of float`: List of MCD values for each sample.
    """
    melcd_list = []

    # Iterate through each pair of clean and predicted audio feature arrays
    for clean_sample, predicted_sample in zip(clean_features, predicted_features):
        # Extract MFCC coefficients from both clean and predicted audio features
        clean_mfcc = clean_sample[:, :_NUM_MFCC]
        predicted_mfcc = predicted_sample[:, :_NUM_MFCC]

        # Compute the Mel-Cepstral Distortion (MCD) between clean and predicted MFCCs
        mcd_score = melcd(predicted_mfcc, clean_mfcc)
        melcd_list.append(mcd_score)

    return melcd_list



def calculate_bap_distortion(clean_features, predicted_features):
    """
    Calculate the Band Aperiodicities Distortion (BAPD) for each sample 
    between clean and predicted audio features.

    Parameters:
    --------------------------------------
    - clean_features (`list of np.ndarray`): List of arrays containing the clean audio features for each sample.
    - predicted_features (`list of np.ndarray`): List of arrays containing the predicted audio features for each sample.

    Returns:
    --------------------------------------
    - `list of float`: List of BAPD values for each sample.
    """
    bap_distortion_list = []

    # Iterate through each pair of clean and predicted audio feature arrays
    for clean_sample, predicted_sample in zip(clean_features, predicted_features):
        # Extract band aperiodicity values from both clean and predicted audio features
        clean_bap = clean_sample[:, -3]
        predicted_bap = predicted_sample[:, -3]

        # Compute the Band Aperiodicities Distortion (BAPD) between clean and predicted values
        bap_distortion_score = bap_distortion(clean_bap, predicted_bap)
        bap_distortion_list.append(bap_distortion_score)

    return bap_distortion_list



def calculate_lf0_rmse(clean_features, predicted_features):
    """
    Calculate the Root Mean Square Error (RMSE) of the log fundamental frequency (log F0)
    for each sample between clean and predicted audio features.

    Parameters:
    --------------------------------------
    - clean_features (`list of np.ndarray`): List of arrays containing the clean audio features for each sample.
    - predicted_features (`list of np.ndarray`): List of arrays containing the predicted audio features for each sample.

    Returns:
    --------------------------------------
    - `list of float`: List of RMSE values for log F0 for each sample.
    """
    lf0_rmse_list = []

    # Iterate through each pair of clean and predicted audio feature arrays
    for clean_sample, predicted_sample in zip(clean_features, predicted_features):
        # Extract log F0 values from both clean and predicted audio features
        clean_lf0 = clean_sample[:, -2]
        predicted_lf0 = predicted_sample[:, -2]

        # Extract voice/unvoiced decision coefficients from both clean and predicted audio features
        clean_vuv = clean_sample[:, -1]
        predicted_vuv = predicted_sample[:, -1]

        # Apply voicing decisions to the log F0 values
        clean_lf0_vuv = clean_lf0 * clean_vuv
        predicted_lf0_vuv = predicted_lf0 * predicted_vuv

        # Compute RMSE between clean and predicted log F0 values and append to the list
        rmse = np.sqrt(mean_squared_error(clean_lf0_vuv, predicted_lf0_vuv))
        lf0_rmse_list.append(rmse)

    return lf0_rmse_list




def calculate_vuv_error_rate(clean_features, predicted_features):
    """
    Calculate the Voice-Unvoiced (VUV) error rate for each sample 
    between clean and predicted audio features.

    Parameters:
    --------------------------------------
    - clean_features (`list of np.ndarray`): List of arrays containing the clean audio features for each sample.
    - predicted_features (`list of np.ndarray`): List of arrays containing the predicted audio features for each sample.

    Returns:
    --------------------------------------
    - `list of float`: List of VUV error rates as percentage values for each sample.
    """
    vuv_error_rate_list = []

    # Iterate through each pair of clean and predicted audio feature arrays
    for clean_sample, predicted_sample in zip(clean_features, predicted_features):
        # Extract voice/unvoiced decisions from both clean and predicted audio feature arrays
        clean_vuv = clean_sample[:, -1]
        predicted_vuv = predicted_sample[:, -1]

        # Calculate the total number of frames
        num_frames = clean_vuv.size

        # Compute the number of mismatches between clean and predicted voicing decisions
        mismatches = np.sum(clean_vuv != predicted_vuv)

        # Calculate the VUV error rate as a percentage and append to the list
        vuv_error_rate = (mismatches / num_frames) * 100 
        vuv_error_rate_list.append(vuv_error_rate)

    return vuv_error_rate_list



def calculate_stoi(clean_features, predicted_features):
    """
    Calculate the Short-Time Objective Intelligibility (STOI) score for each sample 
    between clean and predicted audio features.

    Parameters:
    --------------------------------------
    - clean_features (`list of np.ndarray`): List of arrays containing the clean audio features for each sample.
    - predicted_features (`list of np.ndarray`): List of arrays containing the predicted audio features for each sample.

    Returns:
    --------------------------------------
    - `list of float`: List of STOI scores for each sample.
    """
    stoi_list = []
    # Iterate through each pair of clean and predicted audio feature arrays
    for clean_sample, predicted_sample in zip(clean_features, predicted_features):
        # Convert feature arrays to time-domain signals using vocoder 
        clean_signal = minmax_scale(vocoder(clean_sample), feature_range=(-1, 1))
        predicted_signal = minmax_scale(vocoder(predicted_sample), feature_range=(-1, 1))

        # Compute the Short-Time Objective Intelligibility (STOI) score for the current pair of signals and append to the list
        stoi_score = stoi(clean_signal, predicted_signal, _FS, extended=False)
        stoi_list.append(stoi_score)

    return stoi_list



def calculate_pesq(clean_features, predicted_features):
    """
    Calculate the Perceptual Evaluation of Speech Quality (PESQ) score for each sample 
    between clean and predicted audio features.

    Parameters:
    --------------------------------------
    - clean_features (`list of np.ndarray`): List of arrays containing the clean audio features for each sample.
    - predicted_features (`list of np.ndarray`): List of arrays containing the predicted audio features for each sample.
    - fs (`int`): Sampling frequency of the signals.
    - fftlen (`int`): Length of the FFT window.
    - frame_period (`float`): Frame period in seconds.

    Returns:
    --------------------------------------
    - `list of float`: List of PESQ scores for each sample.
    """
    pesq_list = []

    # Iterate through each pair of clean and predicted audio feature arrays
    for clean_sample, predicted_sample in zip(clean_features, predicted_features):
        # Convert feature arrays to time-domain signals using vocoder
        clean_signal = minmax_scale(vocoder(clean_sample), feature_range=(-1, 1))
        predicted_signal = minmax_scale(vocoder(predicted_sample), feature_range=(-1, 1))   
        
        # Compute the PESQ score for the current pair of signals and append to the list
        pesq_score = pesq(_FS, clean_signal, predicted_signal, 'nb')
        pesq_list.append(pesq_score)

    return pesq_list


    
def vocoder(params):
    """
    Synthesize an audio waveform from given audio features using WORLD vocoder.

    Parameters:
    --------------------------------------
    - params (`np.ndarray`): Matrix of audio features.

    Returns:
    --------------------------------------
    - `np.ndarray`: Synthesized audio waveform.
    """

    mgc = params[:, :-3]                # Mel-Generalized Cepstrum
    bap = params[:, -3][:, np.newaxis]  # Band Aperiodicity
    lf0 = params[:, -2][:, np.newaxis]  # Log F0
    vuv = params[:, -1][:, np.newaxis]  # Voice/Unvoiced decision

    # Decode spectral envelope from MGC
    sp = pw.decode_spectral_envelope(mgc.copy(order='C'), fs=_FS, fft_size=_FFTLEN)
    # Decode aperiodicity from BAP
    ap = pw.decode_aperiodicity(bap.copy(order='C'), fs=_FS, fft_size=_FFTLEN)
    # Calculate F0 values from LF0 and VUV
    f0 = np.exp(lf0) * vuv

    # Decode WORLD vocoder parameters
    f0, sp, ap = pwm.decode_world_params(lf0, vuv, mgc, bap, _FS)
    # Synthesize waveform using WORLD vocoder
    waveform = pw.synthesize(f0.flatten(), sp, ap, fs=_FS, frame_period=1000 * _FRAME_PERIOD)

    return waveform



if __name__ == '__main__':
    
    # Initialize a dictionary to store evaluation metrics for each participant and fold
    evaluation_metrics = {}

    # Load participant metadata from the BIDS dataset
    participants = pd.read_csv(os.path.join(_PATH_BIDS, 'participants.tsv'), delimiter='\t')
    # participants = pd.DataFrame({'participant_id': ['sub-11']})

    # Iterate over each participant in the dataset
    for participant in participants['participant_id']:
        # Initialize a dictionary entry for the current participant
        evaluation_metrics[f'{participant}'] = {}

        # Initialize dictionaries to store clean and predicted sample outputs
        clean_sample_outputs = {}
        predicted_sample_outputs = {}

        # Iterate over each validation fold directory for the current participant
        for k, fold in enumerate(iglob(os.path.join(_PATH_BIDS, participant, 'validation', '*'))):
            # Log the participant ID and the current fold number for tracking progress
            print(f'Participant {participant}, Fold #{k}')
            print('*****Loading the data...')

            # List of paths to the clean audio feature files for the current fold
            clean_paths = [os.path.join(_PATH_BIDS, participant, 'audio_feat', os.path.basename(f)) for f in iglob(os.path.join(fold, '*.npy'))]

            # Load clean audio features 
            clean_feat = [np.load(file_path) for file_path in clean_paths]
            # Load predicted audio features
            predicted_feat = [np.load(file_path) for file_path in iglob(os.path.join(fold, '*.npy'))]

            
            # Iterate through each pair of clean and predicted audio feature arrays
            for s, (clean_sample, predicted_sample) in enumerate(zip(clean_feat, predicted_feat)):
                # Create independent copies of the arrays to avoid modifying the original data
                clean_sample = np.copy(clean_sample)
                predicted_sample = np.copy(predicted_sample)

                # Convert log F0 to linear scale
                clean_sample[:, -2] = np.exp(clean_sample[:, -2])
                predicted_sample[:, -2] = np.exp(predicted_sample[:, -2])

                # Append clean and predicted samples to their respective dictionaries
                clean_sample_outputs.setdefault(f's{s}', []).append(clean_sample)
                predicted_sample_outputs.setdefault(f's{s}', []).append(predicted_sample)
            

            # Calculate Mel-Cepstral Distortion (MCD) scores
            melcd_scores = calculate_melcd(clean_feat, predicted_feat)
            # Calculate Band Aperiodicities Distortion (BAPD) scores
            bapd_scores = calculate_bap_distortion(clean_feat, predicted_feat)
            # Calculate Root Mean Square Error (RMSE) of log fundamental frequency (log F0)
            lf0_rmse_scores = calculate_lf0_rmse(clean_feat, predicted_feat)
            # Calculate Voice-Unvoiced (VUV) error rates
            vuv_error_rates = calculate_vuv_error_rate(clean_feat, predicted_feat)
            # Calculate Short-Time Objective Intelligibility (STOI) scores
            stoi_scores = calculate_stoi(clean_feat, predicted_feat) 
            # Calculate PESQ (Perceptual Evaluation of Speech Quality) scores
            pesq_scores = calculate_pesq(clean_feat, predicted_feat)

            # Store evaluation metrics for the current fold
            evaluation_metrics[f'{participant}'][f'f{k}'] = {
                'melcd': melcd_scores,              # Mel-Cepstral Distortion scores
                'bapd': bapd_scores,                # Band Aperiodicities Distortion scores
                'lf0_rmse': lf0_rmse_scores,        # Root Mean Square Error of log fundamental frequency (log F0)
                'vuv_error_rate': vuv_error_rates,  # Voice-Unvoiced error rates
                'stoi': stoi_scores,                # Short-Time Objective Intelligibility scores
                'pesq': pesq_scores                 # PESQ (Perceptual Evaluation of Speech Quality) scores
            }
            
               
        # Compute the average clean and predicted features across all folds for each test sample
        mean_clean_feat = []
        mean_predicted_feat = []
        for k in clean_sample_outputs.keys():
            # Stack feature matrices across all folds for the current sample and compute the mean
            mean_clean_sample = np.mean(np.stack(clean_sample_outputs[k], axis=2), axis=2)
            mean_predicted_sample = np.mean(np.stack(predicted_sample_outputs[k], axis=2), axis=2)

            # Convert the linear F0 back to log scale
            mean_clean_sample[:, -2] = np.log(mean_clean_sample[:, -2])
            mean_predicted_sample[:, -2] = np.log(mean_predicted_sample[:, -2])
            
            # Append the average clean and predicted audio feature matrices to their respective lists
            mean_clean_feat.append(mean_clean_sample)
            mean_predicted_feat.append(mean_predicted_sample)


        # Calculate average STOI (Short-Time Objective Intelligibility) scores using mean predicted features
        mean_stoi_scores = calculate_stoi(mean_clean_feat, mean_predicted_feat) 
        # Calculate average PESQ (Perceptual Evaluation of Speech Quality) scores using mean predicted features
        mean_pesq_scores = calculate_pesq(mean_clean_feat, mean_predicted_feat)

        # Store the average STOI and PESQ scores for the current participant
        evaluation_metrics[f'{participant}']['mean_stoi'] = mean_stoi_scores
        evaluation_metrics[f'{participant}']['mean_pesq'] = mean_pesq_scores

    
    # Save evaluation metrics to a JSON file
    with open('./metrics/data.json', 'w') as file:
        json.dump(evaluation_metrics, file, indent=4)