import numpy as np
from scipy.signal import welch
from tqdm import tqdm
from functools import wraps
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import describe

# 周波数帯域の定義
FREQUENCY_BANDS = {
    "theta": [4, 8],
    "alpha": [8, 12],
    "low_beta": [12, 16],
    "high_beta": [16, 25],
    "gamma": [25, 45]
}

def extract_band_powers_for_1D_segments(data, fs=128, window_sec=10, overlap_sec=5, relative=False):
    """
    データをオーバーラップありでセグメント化し、各セグメントの特徴量を抽出
    
    Parameters:
    -----------
    data : array-like
        入力データ (trials, channels, time_points)
    fs : int
        サンプリング周波数
    window_sec : int
        セグメントの長さ（秒）
    overlap_sec : int
        オーバーラップの長さ（秒）
    relative : bool
        相対バンドパワーを計算するかどうか
    
    Returns:
    --------
    features : array-like
        セグメント化された特徴量 (n_segments_total, bands*channels, 1)
    """
    window_points = window_sec * fs
    overlap_points = overlap_sec * fs
    stride = window_points - overlap_points
    
    n_trials, n_channels, _ = data.shape
    all_segments = []
    
    for trial in tqdm(range(n_trials), desc="トライアルの処理"):
        # 1トライアルごとにセグメント化
        for start in range(0, data.shape[2] - window_points + 1, stride):
            segment = data[trial, :, start:start + window_points]
            segment_features = []
            
            for band in FREQUENCY_BANDS.values():
                for channel in range(n_channels):
                    power = bandpower(segment[channel], fs, band,
                                   window_sec=window_sec, relative=relative)
                    segment_features.append(power)
            
            all_segments.append(segment_features)
    
    features = np.array(all_segments)[..., np.newaxis]
    return features

def extract_band_powers_for_2D_14ch_segments(data, fs=128, window_sec=10, overlap_sec=5, relative=False):
    """
    14チャンネル用の2D特徴量抽出（セグメント化対応）
    
    Returns:
    --------
    features : array-like
        セグメント化された特徴量 (n_segments_total, bands, channels, views)
    """
    window_points = window_sec * fs
    overlap_points = overlap_sec * fs
    stride = window_points - overlap_points
    
    n_trials, n_channels, _ = data.shape
    n_bands = len(FREQUENCY_BANDS)
    all_segments = []
    
    for trial in tqdm(range(n_trials), desc="トライアルの処理"):
        for start in range(0, data.shape[2] - window_points + 1, stride):
            segment = data[trial, :, start:start + window_points]
            segment_features = np.zeros((n_bands, n_channels, 2))  # 2 views
            
            for b, band in enumerate(FREQUENCY_BANDS.values()):
                for ch in range(n_channels):
                    power = bandpower(segment[ch], fs, band,
                                   window_sec=window_sec, relative=relative)
                    # 左右対称のview
                    segment_features[b, ch, 0] = power  # 前から後ろ
                    segment_features[b, ch, 1] = power  # 左右対称
            
            all_segments.append(segment_features)
    
    return np.array(all_segments)

def extract_band_powers_for_2D_32ch_segments(data, fs=128, window_sec=10, overlap_sec=5, relative=False):
    """
    32チャンネル用の2D特徴量抽出（セグメント化対応）
    
    Parameters:
    -----------
    data : array-like
        入力データ (trials, channels, time_points)
    fs : int
        サンプリング周波数
    window_sec : int
        セグメントの長さ（秒）
    overlap_sec : int
        オーバーラップの長さ（秒）
    relative : bool
        相対バンドパワーを計算するかどうか
    
    Returns:
    --------
    features : array-like
        セグメント化された特徴量 (n_segments_total, bands, channels, views)
    """
    window_points = window_sec * fs
    overlap_points = overlap_sec * fs
    stride = window_points - overlap_points
    
    n_trials, n_channels, _ = data.shape
    n_bands = len(FREQUENCY_BANDS)
    all_segments = []
    
    for trial in tqdm(range(n_trials), desc="トライアルの処理"):
        for start in range(0, data.shape[2] - window_points + 1, stride):
            segment = data[trial, :, start:start + window_points]
            segment_features = np.zeros((n_bands, n_channels, 2))  # 2 views
            
            for b, band in enumerate(FREQUENCY_BANDS.values()):
                for ch in range(n_channels):
                    power = bandpower(segment[ch], fs, band,
                                   window_sec=window_sec, relative=relative)
                    # 左右対称のview
                    segment_features[b, ch, 0] = power  # 前から後ろ
                    segment_features[b, ch, 1] = power  # 左右対称
            
            all_segments.append(segment_features)
    
    return np.array(all_segments)

def bandpower(data, fs, band, window_sec=None, relative=False):
    """
    特定の周波数帯域のパワーを計算
    """
    if window_sec is None:
        nperseg = min(fs * 4, len(data))
    else:
        nperseg = min(fs * window_sec, len(data))
    
    freqs, psd = welch(data, fs, nperseg=nperseg)
    
    # 指定された帯域のインデックスを取得
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    
    # バンドパワーを計算
    band_power = np.trapz(psd[idx_band], freqs[idx_band])
    
    if relative:
        total_power = np.trapz(psd, freqs)
        band_power /= total_power
    
    return band_power

def plot_feature_distributions(features, model_type, save_dir):
    """特徴量の分布を可視化"""
    os.makedirs(save_dir, exist_ok=True)
    
    if model_type == '1DCNN':
        features_flat = features.squeeze()
    else:  # 2DCNN
        features_flat = features.reshape(features.shape[0], -1)
    
    # 基本統計量の計算
    stats = describe(features_flat)
    
    # 統計情報の保存
    stats_dict = {
        'mean': float(np.mean(stats.mean)),  # 修正
        'variance': float(np.mean(stats.variance)),  # 修正
        'skewness': float(np.mean(stats.skewness)),  # 修正
        'kurtosis': float(np.mean(stats.kurtosis))   # 修正
    }
    
    with open(os.path.join(save_dir, 'feature_stats.json'), 'w') as f:
        json.dump(stats_dict, f, indent=4)
    
    # ヒストグラムの作成
    plt.figure(figsize=(10, 6))
    sns.histplot(data=features_flat.flatten(), bins=50)
    plt.title('Feature Distribution')
    plt.xlabel('Feature Value')
    plt.ylabel('Count')
    plt.savefig(os.path.join(save_dir, 'feature_distribution.png'))
    plt.close()
    
    return stats_dict

def convert_log_features(features):
    return np.log(features)

