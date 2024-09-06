import json
import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib.ticker as ticker  

# Set the plot style to 'classic'
plt.style.use('classic')

def get_fixed_ticks(limits, num_ticks):
    """Generate fixed ticks based on the y-axis limits and desired number of ticks."""
    min_limit, max_limit = limits
    return np.linspace(min_limit, max_limit, num_ticks)

if __name__ == "__main__":

    # Load evaluation metrics from JSON file
    with open('./metrics/data.json', 'r') as file:
        evaluation_metrics = json.load(file)

    # Define the list of metrics to be plotted
    metrics = ['melcd', 'bapd', 'lf0_rmse', 'vuv_error_rate', 'stoi', 'pesq']

    # Extract participant names (keys) from the evaluation metrics
    participants = list(evaluation_metrics.keys())

    # Initialize a dictionary to collect y-axis limits for each subplot index
    subplot_limits = {i: [] for i in range(len(metrics))}

    # First pass: Plot all figures and collect y-axis limits
    for participant in participants:
        fig, axes = plt.subplots(3, 2, figsize=(19.5, 14.9))
        axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing

        for i, metric in enumerate(metrics):
            ax = axes[i]

            # Extract metric data for each fold for the current participant
            fold_data = [evaluation_metrics[participant][f'f{fold}'][metric] for fold in range(5)]
            
            # Compute the mean value for plotting
            if metric not in ['stoi', 'pesq']:
                mean_value = np.mean(fold_data, axis=0)
            elif metric == 'lf0':
                mean_value = np.log(np.mean(np.exp(fold_data), axis=0))
            else:
                mean_value = evaluation_metrics[participant][f'mean_{metric}']
            
            fold_data.append(mean_value)
            ax.boxplot(fold_data)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            if metric == 'stoi':
                subplot_limits[i].append((0, 1))
            else:
                subplot_limits[i].append(ax.get_ylim())

        plt.close(fig)

    # Determine fixed y-axis limits for each subplot index
    fixed_limits = {}
    for i, limits in subplot_limits.items():
        min_limit = min(l[0] for l in limits)
        max_limit = max(l[1] for l in limits)
        fixed_limits[i] = (min_limit, max_limit)

    # Second pass: Create figures again with fixed y-axis limits
    for participant in participants:
        fig, axes = plt.subplots(3, 2, figsize=(19.5, 14.9))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            ax = axes[i]

            # Set y-axis label based on the current metric
            y_labels = {
                'melcd': 'Distorsión Mel-Cepstral (dB)',
                'bapd': 'Distorsión de Aperiodicidades (dB)',
                'lf0_rmse': 'RECM log F0',
                'vuv_error_rate': 'Tasa de Error SNS (%)',
                'stoi': 'STOI',
                'pesq': 'PESQ'
            }
            ax.set_ylabel(y_labels.get(metric, ''))

            # Extract metric data for each fold for the current participant
            fold_data = [evaluation_metrics[participant][f'f{fold}'][metric] for fold in range(5)]
            
            # Compute the mean value for plotting
            if metric not in ['stoi', 'pesq']:
                mean_value = np.mean(fold_data, axis=0)
            elif metric == 'lf0':
                mean_value = np.log(np.mean(np.exp(fold_data), axis=0))
            else:
                mean_value = evaluation_metrics[participant][f'mean_{metric}']
            
            fold_data.append(mean_value)
            
            bplot = ax.boxplot(fold_data, labels=[f'Fold {i+1}' for i in range(5)] + ['Media'], patch_artist=True)
            # fill with colors
            for patch in bplot['boxes']:
                patch.set_color('black')
                patch.set_facecolor('#3454a4')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_ylim(fixed_limits[i])
            ticks = get_fixed_ticks(fixed_limits[i], 6)
            ax.yaxis.set_major_locator(ticker.FixedLocator(ticks))
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
            ax.grid(False)

        plt.subplots_adjust(
            top=0.94,      # Top margin
            bottom=0.15,   # Bottom margin
            left=0.125,    # Left margin
            right=0.9,     # Right margin
            hspace=0.295,  # Height space between plots
            wspace=0.17    # Width space between plots
        )

        # Save the figure to an EPS file named after the participant
        filename = f'./metrics/{participant}.eps'
        plt.savefig(filename, format='eps')

    # Display the plots on the screen
    plt.show()
