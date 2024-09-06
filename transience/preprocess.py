import math
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.pipeline import Pipeline
from python_speech_features import delta
import scipy
from nnmnkwii.preprocessing import delta_features
from scipy.fft import rfft, rfftfreq
import sys
import pywt
import matplotlib.pyplot as plt

#Small helper function to speed up the hilbert transform
hilbert3 = lambda x: scipy.signal.hilbert(x,axis=0)

def extractHG(data, sr, frame_period):
    """
    Window data and extract frequency-band envelope using the hilbert transform
    
    Parameters
    ----------
    data: array (samples, channels)
        EEG time series
    sr: int
        Sampling rate of the data
    windowLength: float
        Length of window (in seconds) in which spectrogram will be calculated
    frameshift: float
        Shift (in seconds) after which next window will be extracted
    Returns
    ----------
    feat: array (windows, channels)
        Frequency-band feature matrix
    """
    

    #Linear detrend
    data = scipy.signal.detrend(data,axis=0)
    #Number of windows
    numWindows = int(data.shape[0] / sr / frame_period + 1)
    #Filter High-Gamma Band
    sos = scipy.signal.iirfilter(4, [70/(sr/2),170/(sr/2)],btype='bandpass',output='sos')
    data = scipy.signal.sosfiltfilt(sos,data,axis=0)
    #Attenuate first harmonic of line noise
    sos = scipy.signal.iirfilter(4, [98/(sr/2),102/(sr/2)],btype='bandstop',output='sos')
    data = scipy.signal.sosfiltfilt(sos,data,axis=0)
    #Attenuate second harmonic of line noise
    sos = scipy.signal.iirfilter(4, [148/(sr/2),152/(sr/2)],btype='bandstop',output='sos')
    data = scipy.signal.sosfiltfilt(sos,data,axis=0)
    #Create feature space
    data = np.abs(hilbert3(data)) 
    feat = np.zeros((numWindows,data.shape[1]))
    for win in range(numWindows):
        start= int(np.floor((win*frame_period)*sr))
        stop = int(np.floor(start+frame_period*sr))
        feat[win,:] = np.mean(data[start:stop,:],axis=0)
    
    return feat

def normalize_data(dataset, mean=None, std=None):
    if mean is None and std is None:
        d = np.vstack(dataset)
        mean = np.mean(d, axis=0)
        std = np.std(d, axis=0)

    for x in dataset:
        x -= mean
        x /= std

    return mean, std


def window_stack(x, stepsize=1, width=3):
    w2 = math.floor(width / 2.0)
    x2 = np.vstack([np.tile(x[0], (w2, 1)), x, np.tile(x[-1], (w2, 1))])  # Edges are padded with the first/last element of the sequence
    var = np.hstack([x2[i:1 + i - width or None:stepsize] for i in range(0, width)])
    return var


def add_noise(dataset, noise_std):
    if isinstance(dataset, list):
        return [x + noise_std * np.random.standard_normal(size=x.shape) for x in dataset]
    else:
        return dataset + noise_std * np.random.standard_normal(size=dataset.shape)


# EEG pipeline (adds contextual frame information to raw data)
def create_preprocessing_pipeline_sensor(win_len, pca_components):
    pipeline = []
    if win_len > 1:
        pipeline.append(('stacker', FeatureStacker(win_len)))
    if pca_components:
        pipeline.append(('mean_removal', StandardScaler(with_std=False)))
        pipeline.append(('pca', PCA(n_components=pca_components, svd_solver='full')))
    pipeline.append(('standardizer', StandardScaler()))
    pipeline.append(('dummy', DummyCustomRegressor()))
    p = Pipeline(pipeline)
    p.is_fitted_ = False
    return p

# MFCC pipeline

def create_preprocessing_pipeline_mfcc(mfcc_order, delta_win):
    pipeline = []
    pipeline.append(('slicer', FeatureSlicing(np.arange(mfcc_order))))
    pipeline.append(('dynamic_params', MFCCDeltaAcc(delta_win)))
    pipeline.append(('standardizer', StandardScaler()))
    pipeline.append(('dummy', DummyCustomRegressor()))
    p = Pipeline(pipeline)
    p.is_fitted_ = False
    return p


def preprocess_data(pipeline, X):
    dataset = X if isinstance(X, list) else [X]
    if not pipeline.is_fitted_:
        D = np.vstack(dataset)
        pipeline.fit(D, D)
        pipeline.is_fitted_ = True
    transformed_dataset = [pipeline.predict(x) for x in dataset]
    return transformed_dataset if isinstance(X, list) else transformed_dataset[0]


class FeatureStacker(TransformerMixin):
    def __init__(self, win_len=1, **kwargs):
        super().__init__(**kwargs)
        self.win_ = win_len

    def fit(self, X, y=None):
        pass

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X):
        return X if self.win_ <= 1 else window_stack(X, stepsize=1, width=self.win_)


class FeatureSlicing(TransformerMixin):
    def __init__(self, slicing_index, **kwargs):
        super().__init__(**kwargs)
        self.idx_ = slicing_index

    def fit(self, X, y=None):
        pass

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X):
        return X[:, self.idx_]


class MFCCDeltaAcc(TransformerMixin):
    def __init__(self, delta_win, **kwargs):
        super().__init__(**kwargs)
        self.delta_win_ = delta_win
        # self.acc_win_ = acc_win

    def fit(self, X, y=None):
        pass

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X):
        delta_feats = delta_features(X, _DELTA_WIN)
        # acc_feats = delta(delta_feats, self.acc_win_)
        # return np.hstack([X, delta_feats, acc_feats])
        return np.hstack([X, delta_feats])


class DummyCustomRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X, y=None):
        pass

    def score(self, X, y=None):
        return 0.0

    def predict(self, X):
        return X

def create_preprocessing_pipeline_mfcc_synthesis():
    """
    Creates preprocessing pipeline for mfcc synthesis
    """
    pipeline = []
    pipeline.append(('delta_features', DeltaFeatures()))
    pipeline.append(('standardizer', StandardScaler()))
    pipeline.append(('dummy', DummyCustomRegressor()))
    p = Pipeline(pipeline)
    p.is_fitted_ = False
    return p

_DELTA_WIN = [(0, 0, np.array([1.0])),             # static
              (1, 1, np.array([-0.5, 0.0, 0.5]))]  # delta

class DeltaFeatures(TransformerMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X, y=None):
        pass

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X):
        return delta_features(X, _DELTA_WIN)