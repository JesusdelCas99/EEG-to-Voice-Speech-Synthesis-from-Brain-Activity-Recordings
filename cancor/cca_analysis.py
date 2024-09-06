import os
import configparser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from glob import iglob
from pathlib import Path
import matplotlib.ticker as ticker  
import sys

def load_configuration(config_file):
    """Load configuration settings from a file."""
    config = configparser.ConfigParser()
    config.read(config_file)
    return {
        'NUM_MFCC': int(config['SYNTHESIS']['NUM_MFCC']),
        'PATH_BIDS': config['SYNTHESIS']['PATH_BIDS']
    }

def load_participants(path_bids):
    """Load participants from the specified path."""
    participants_file = os.path.join(path_bids, 'participants.tsv')
    return pd.read_csv(participants_file, delimiter='\t')

def load_features(path_bids, participant, num_mfcc):
    """Load EEG and audio features for a given participant."""
    eeg_files = list(iglob(os.path.join(path_bids, participant, 'eeg', '*.npy')))
    hg_files = list(iglob(os.path.join(path_bids, participant, 'hg_feat', '*.npy')))
    audio_files = list(iglob(os.path.join(path_bids, participant, 'audio_feat', '*.npy')))
    
    eeg_raw_data_labels = [os.path.basename(f) for f in eeg_files]
    eeg_hg = [np.load(f) for f in hg_files]
    audio_mfcc = [np.load(f)[:, :num_mfcc] for f in audio_files]
    
    return eeg_raw_data_labels, eeg_hg, audio_mfcc

def perform_cca(eeg_hg_data, audio_mfcc_data):
    """Perform CCA and return correlation coefficient."""
    if not eeg_hg_data or not audio_mfcc_data:
        return []
    
    eeg_hg_data = np.vstack(eeg_hg_data)
    audio_mfcc_data = np.vstack(audio_mfcc_data)
    
    cca = CCA()
    cca.fit(eeg_hg_data, audio_mfcc_data)
    A_, B_ = cca.transform(eeg_hg_data, audio_mfcc_data)
    
    return np.corrcoef(A_[:, 0], B_[:, 0])[0, 1]

def analyze_participant(participant, path_bids, num_mfcc):
    """Analyze data for a single participant and save plots."""
    eeg_raw_data_labels, eeg_hg, audio_mfcc = load_features(path_bids, participant, num_mfcc)
    words = set(label.split('_')[1].replace('.npy', '') for label in eeg_raw_data_labels)
    
    print(f'Participant {participant} - Number of different spoken words: {len(words)}')
    
    # Perform overall CCA
    eeg_hg_data = [eeg_hg[idx] for idx in range(len(eeg_raw_data_labels))]
    audio_mfcc_data = [audio_mfcc[idx] for idx in range(len(eeg_raw_data_labels))]
    rho_overall = perform_cca(eeg_hg_data, audio_mfcc_data)

    # Perform CCA for each vowel
    vowels = ['A', 'E', 'I', 'O', 'U']
    rho_vowels = {vowel: [] for vowel in vowels}
    
    for vowel in vowels:
        print(f'Analyzing vowel {vowel}')
        eeg_hg_data_vowel = [eeg_hg[idx] for idx, label in enumerate(eeg_raw_data_labels) if vowel in label]
        audio_mfcc_data_vowel = [audio_mfcc[idx] for idx, label in enumerate(eeg_raw_data_labels) if vowel in label]
        rho_vowels[vowel].append(perform_cca(eeg_hg_data_vowel, audio_mfcc_data_vowel))
    
    # Perform CCA for each word
    rho_words = {word: [] for word in words}
    
    for word in words:
        eeg_hg_data_word = [eeg_hg[idx] for idx, label in enumerate(eeg_raw_data_labels) if word in label]
        audio_mfcc_data_word = [audio_mfcc[idx] for idx, label in enumerate(eeg_raw_data_labels) if word in label]
        rho_words[word].append(perform_cca(eeg_hg_data_word, audio_mfcc_data_word))
    
    plot_results(rho_overall, rho_vowels, rho_words, participant)


def plot_results(rho_overall, rho_vowels, rho_words, participant):
    """Generate and save plots for the CCA results."""

    # Apply the 'classic' style
    plt.style.use('classic')
    
    _, axes = plt.subplots(3, 2, figsize=(19.5, 14.9))
    vowels = ['A', 'E', 'I', 'O', 'U']
    words = list(rho_words.keys())
    
    # Flatten axes array for easier indexing
    axes_flat = axes.flatten()

    # Plot vowel analysis as a bar chart
    vowel_rhos = [np.mean(rho_vowels[vowel]) for vowel in vowels]  # Mean of the correlations for each vowel
    axes_flat[0].bar(range(1, len(vowels) + 1), vowel_rhos, color='#3454a4', edgecolor='black', width=0.5)
    axes_flat[0].set_ylim(0, 1)
    axes_flat[0].set_xlim(0.25, len(vowels) + 0.75)
    axes_flat[0].set_ylabel('Coef. de Correlación de Pearson')
    axes_flat[0].set_xticks(range(1, len(vowels) + 1))
    axes_flat[0].set_xticklabels(vowels)
    axes_flat[0].legend(['Análisis de Vocales'])
    axes_flat[0].grid(False)

    # Add horizontal line for overall CCA
    axes_flat[0].axhline(y=rho_overall, color='r', linestyle='--', label=f'ACC global: {rho_overall:.2f}')
    axes_flat[0].legend(loc='lower right', bbox_to_anchor=(1, 0.1))
    axes_flat[0].yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    # Plot individual bar plots for each vowel
    for i, vowel in enumerate(vowels):
        ax = axes_flat[i + 1]
        
        # Collect data for each word containing the vowel
        words_with_vowel = sorted([word for word in words if vowel in word])

        print(words_with_vowel)

        word_scores = [rho_words[word] for word in words_with_vowel]
        
        # Flatten the list of lists and create the bar plot
        flattened_scores = [score for sublist in word_scores for score in sublist]
        labels = [f'{word}' for word in words_with_vowel]  # Labels for individual words
        
        # Compute the mean value of all scores for plotting the mean line
        mean_value = np.mean(flattened_scores)
        
        # Create a bar plot
        bar_width = 0.5
        x = np.arange(len(words_with_vowel))
        ax.bar(x, [np.mean(rho_words[word]) for word in words_with_vowel], width=bar_width, color='#3454a4', edgecolor='black', tick_label=labels)
        
        # Plot a horizontal line for the mean value
        ax.axhline(mean_value, color='r', linestyle='--', label=f'Media: {mean_value:.2f}')
        
        ax.set_xlim(-1, len(words_with_vowel))
        ax.set_ylim(0, 1)
        ax.set_ylabel('Coef. de Correlación de Pearson')
        ax.grid(False)
        ax.legend(loc='lower right', bbox_to_anchor=(1, 0.1))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')


    plt.subplots_adjust(
        top=0.94,      # Top margin
        bottom=0.15,   # Bottom margin
        left=0.125,    # Left margin
        right=0.9,     # Right margin
        hspace=0.295,  # Height space between plots
        wspace=0.17    # Width space between plots
    )

    # Save the figure to an EPS file named after the participant
    filename = f'./cancor/acc_{participant}.eps'
    plt.savefig(filename, format='eps')
    plt.show()



def main():
    """Main function to load configuration and process participants."""
    # Set the working directory to the parent directory of the script's directory
    script_dir = Path(__file__).resolve().parent
    parent_dir = script_dir.parent
    os.chdir(parent_dir)
    
    config_file = './setup.cfg'
    config = load_configuration(config_file)
    
    path_bids = config['PATH_BIDS']
    num_mfcc = config['NUM_MFCC']
    
    participants = load_participants(path_bids)
    
    for participant in participants['participant_id']:
        analyze_participant(participant, path_bids, num_mfcc)

if __name__ == '__main__':
    main()
