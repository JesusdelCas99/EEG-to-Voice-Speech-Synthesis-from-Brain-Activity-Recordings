import os
import numpy as np
import pandas as pd
import scipy.signal as signal
import soundfile as sf
import configparser
from glob import iglob
from transience import preprocess
from transience import pyworld as pw
import sys

def load_configuration(config_file):
    """Load configuration settings from a file."""
    config = configparser.ConfigParser()
    config.read(config_file)
    return {
        'NUM_MFCC': int(config['SYNTHESIS']['NUM_MFCC']),
        'FRAMESHIFT': float(config['SYNTHESIS']['FRAMESHIFT']),
        'EEG_SR': int(config['SYNTHESIS']['EEG_SR']),
        'AUDIO_SR': int(config['SYNTHESIS']['AUDIO_SR']),
        'TARGET_SR': int(config['SYNTHESIS']['TARGET_SR']),
        'PATH_BIDS': config['SYNTHESIS']['PATH_BIDS']
    }

def process_participant(participant, config):
    """Process data for a single participant."""
    print(f'Processing participant: {participant}')
    
    # Paths for EEG and audio data
    eeg_path_pattern = os.path.join(config['PATH_BIDS'], participant, 'eeg', '*.npy')
    audio_path_pattern = os.path.join(config['PATH_BIDS'], participant, 'audio', '*.wav')


    # Load and process EEG data
    eeg_labels = [os.path.splitext(os.path.basename(f))[0] for f in iglob(eeg_path_pattern)]
    eeg_data = [np.transpose(np.load(f)) for f in iglob(eeg_path_pattern)]

    # Extract High Gamma features from EEG data
    hg_features = [preprocess.extractHG(data, config['EEG_SR'], config['FRAMESHIFT']) for data in eeg_data]

    # Save High Gamma features
    hg_feat_path = os.path.join(config['PATH_BIDS'], participant, 'hg_feat')
    os.makedirs(hg_feat_path, exist_ok=True)
    for label, features in zip(eeg_labels, hg_features):
        np.save(os.path.join(hg_feat_path, label), features)

    # Load and process audio data
    audio_files = iglob(audio_path_pattern)
    audio_data = [sf.read(f)[0] for f in audio_files]

    audio_resampled = [signal.resample_poly(data, config['TARGET_SR'], config['AUDIO_SR']) for data in audio_data]

    # Extract audio features and save them
    audio_feat_path = os.path.join(config['PATH_BIDS'], participant, 'audio_feat')
    os.makedirs(audio_feat_path, exist_ok=True)
    
    for i, data in enumerate(audio_resampled):
        # Extract features using WORLD vocoder
        f0, sp, ap = pw.world(data, config['TARGET_SR'], 0.015 * 1000)

        lf0, vuv, mfcc, bap = pw.code_world_params(f0, sp, ap, config['TARGET_SR'], config['NUM_MFCC'])
        
        # Combine features into a single matrix
        audio_features = np.hstack((mfcc, bap, lf0[:, np.newaxis], vuv[:, np.newaxis]))
        
        # Save audio features
        np.save(os.path.join(audio_feat_path, eeg_labels[i]), audio_features)

if __name__ == "__main__":
    # Load configuration
    config_file = './setup.cfg'
    config = load_configuration(config_file)
    
    # Load participant list
    participants_file = os.path.join(config['PATH_BIDS'], 'participants.tsv')
    participants_df = pd.read_csv(participants_file, delimiter='\t')

    # Process each participant
    for participant_id in participants_df['participant_id']:
        process_participant(participant_id, config)
