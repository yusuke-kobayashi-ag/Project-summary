import numpy as np
import scipy.signal as signal
from sklearn.preprocessing import StandardScaler,RobustScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load
import os
import pandas as pd
from tqdm import tqdm

# 共通の周波数帯域定義を追加
FREQUENCY_BANDS = {
    "theta": [4, 8],
    "alpha": [8, 12],
    "low_beta": [12, 16],
    "high_beta": [16, 25],
    "gamma": [25, 45]
}


def cache_features(func):
    """
    特徴量抽出関数のキャッシュデコレータ（進捗表示付き）
    """
    def wrapper(*args, **kwargs):
        # キャッシュディレクトリの作成
        cache_dir = './cache/features'
        os.makedirs(cache_dir, exist_ok=True)
        
        # キャッシュファイル名の生成
        params = f"{func.__name__}_"
        params += "_".join([f"{k}={v}" for k, v in kwargs.items()])
        cache_file = os.path.join(cache_dir, f"{params}.npz")
        
        # キャッシュが存在する場合はロード
        if os.path.exists(cache_file):
            print(f"Loading cached features from {cache_file}")
            with np.load(cache_file) as data:
                features = data['features']
                if 'stats' in data:
                    stats = data['stats'].item()
                    return features, stats
                return features
        
        print(f"Computing features using {func.__name__}...")
        # キャッシュが存在しない場合は計算して保存
        result = func(*args, **kwargs)
        
        if isinstance(result, tuple):
            features, stats = result
            np.savez(cache_file, features=features, stats=stats)
        else:
            features = result
            np.savez(cache_file, features=features)
        
        print(f"Saved features to cache: {cache_file}")
        return result
    
    return wrapper

def plot_feature_distributions(features, model_type='1DCNN', save_dir='./results/feature_distributions'):
    """
    特徴量の分布を可視化する関数
    
    Parameters:
    -----------
    features : numpy.ndarray
        特徴量データ
    model_type : str
        モデルタイプ ('1DCNN' or '2DCNN')
    save_dir : str
        保存先ディレクトリ
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # オリジナルデータの分布
    plt.figure(figsize=(10, 6))
    plt.hist(features.flatten(), bins=50, density=True)
    plt.title('Original Feature Distribution')
    plt.xlabel('Feature Value')
    plt.ylabel('Density')
    plt.savefig(f'{save_dir}/original_distribution.png')
    plt.close()
    
    # log変換後の分布
    min_val = np.min(features)
    if min_val <= 0:
        offset = abs(min_val) + 1e-10
        features_log = np.log(features + offset)
    else:
        features_log = np.log(features)
    
    plt.figure(figsize=(10, 6))
    plt.hist(features_log.flatten(), bins=50, density=True)
    plt.title('Log-transformed Feature Distribution')
    plt.xlabel('Feature Value')
    plt.ylabel('Density')
    plt.savefig(f'{save_dir}/log_transformed_distribution.png')
    plt.close()
    
    # スケーリング後の分布
    if model_type == '1DCNN':
        features_reshaped = features_log.reshape(features_log.shape[0], -1)
    else:  # 2DCNN
        features_reshaped = features_log.reshape(features_log.shape[0], -1)
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_reshaped)
    
    plt.figure(figsize=(10, 6))
    plt.hist(features_scaled.flatten(), bins=50, density=True)
    plt.title('Scaled Feature Distribution')
    plt.xlabel('Feature Value')
    plt.ylabel('Density')
    plt.savefig(f'{save_dir}/scaled_distribution.png')
    plt.close()
    
    # 統計情報の出力
    stats = {
        'original': {
            'mean': np.mean(features),
            'std': np.std(features),
            'min': np.min(features),
            'max': np.max(features)
        },
        'log_transformed': {
            'mean': np.mean(features_log),
            'std': np.std(features_log),
            'min': np.min(features_log),
            'max': np.max(features_log)
        },
        'scaled': {
            'mean': np.mean(features_scaled),
            'std': np.std(features_scaled),
            'min': np.min(features_scaled),
            'max': np.max(features_scaled)
        }
    }
    
    # 統計情報をファイルに保存
    with open(f'{save_dir}/distribution_stats.txt', 'w') as f:
        for stage, values in stats.items():
            f.write(f'\n{stage.upper()} Statistics:\n')
            for stat, value in values.items():
                f.write(f'{stat}: {value:.4f}\n')

    return stats

def bandpower(data, sf, band, window_sec=10, relative=False, window='hann',noverlap=None):
    band = np.asarray(band)
    low, high = band
    nperseg = min(window_sec * sf, len(data))

    if nperseg < 2:
        return 0
    
    if noverlap is None:
        noverlap = nperseg // 2  # 50%オーバーラップをデフォルトに

    freqs, psd = signal.welch(data, sf, nperseg=nperseg, window=window,noverlap=noverlap,detrend=False)
    
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    bp = np.trapz(psd[idx_band], freqs[idx_band])

    if relative:
        bp /= np.trapz(psd, freqs)

    return bp


#@cache_features
def extract_band_powers_for_1D(eeg_data, fs=128, window_sec=10, relative=False):
    """
    1DCNN用の特徴量抽出
    Returns: (n_trials, features=bands*channels, 1)
    """
    bands = FREQUENCY_BANDS
    
    all_trial_features = []
    
    for trial in eeg_data:
        trial_features = []
        # バンドを外側のループに
        for band in bands.values():
            # チャンネルを内側のループに
            for channel in range(trial.shape[0]):
                trial_features.append(
                    bandpower(trial[channel], fs, band,
                            window_sec=window_sec, relative=relative)
                )
        
        all_trial_features.append(trial_features)
    
    features = np.array(all_trial_features)  # (n_trials, bands*channels)
    features = features[..., np.newaxis]  # (n_trials, bands*channels, 1)
    
    return features

@cache_features
def extract_band_powers_for_1D_segments(eeg_data, fs=128, window_sec=10, relative=False, overlap_sec=5):
    """
    セグメント分割を行う1DCNN用の特徴量抽出
    Returns: (n_trials, n_segments, bands*channels, 1)
    """
    bands = FREQUENCY_BANDS
    
    segment_samples = window_sec * fs
    overlap_samples = overlap_sec * fs
    step_samples = segment_samples - overlap_samples
    
    all_trial_features = []
    
    for trial in tqdm(eeg_data, desc="トライアルの処理"):
        trial_length = trial.shape[1]
        n_segments = (trial_length - overlap_samples) // step_samples
        
        trial_segments = []
        for i in range(n_segments):
            start_idx = i * step_samples
            end_idx = start_idx + segment_samples
            
            if end_idx > trial_length:
                break
                
            segment = trial[:, start_idx:end_idx]
            segment_features = []
            
            for band in bands.values():
                for channel in range(trial.shape[0]):
                    segment_features.append(
                        bandpower(segment[channel], fs, band,
                                window_sec=window_sec, relative=relative)
                    )
            
            trial_segments.append(segment_features)
        
        all_trial_features.append(trial_segments)
    
    features = np.array(all_trial_features)
    features = features[..., np.newaxis]
    
    return features

@cache_features
def extract_band_powers_for_2D_14ch(eeg_data, fs=128, window_sec=10, relative=False, noverlap=None, clipping=False):

    bands = FREQUENCY_BANDS
    
    # 入力データの順序（これは eeg_data の順序と一致している必要があります）
    input_order = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    
    # 3つの異なるチャンネル順序
    channel_orders = [
        ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'],  # 前頭部から後頭部
        ['AF3', 'AF4', 'F8', 'F7', 'F3', 'F4', 'FC6', 'FC5', 'T7', 'T8', 'P8', 'P7', 'O1', 'O2'],   # 左右対称                       
                         
    ]
    
    all_features = []
    
    for order in channel_orders:
        features = []
        for trial in eeg_data:
        
            trial_features = []
            for band in bands.values():
                band_features = []
                for channel in order:
                    channel_idx = input_order.index(channel)  # 入力データの順序に基づいてインデックスを取得
                    band_features.append(bandpower(trial[channel_idx], fs, band, window_sec=window_sec, relative=relative, noverlap=noverlap))
                trial_features.append(band_features)
            features.append(trial_features)
        
        features = np.array(features)
        features = features.reshape(features.shape[0], 5, 14, 1)
        all_features.append(features)
    
    # 3つの特徴量を結合して (none, 5, 14, 3) の形状にする
    combined_features = np.concatenate(all_features, axis=3)

    # 外れ値の除去
    if clipping:
        # 99パーセンタイルでクリッピング
        clip_value = np.percentile(combined_features, 99)
        combined_features = np.clip(combined_features, None, clip_value)
        
        # 1パーセンタイルでクリッピング
        lower_clip = np.percentile(combined_features, 1)
        combined_features = np.clip(combined_features, lower_clip, None)
    
    return combined_features

@cache_features
def extract_band_powers_for_2D_32ch(eeg_data, fs=128, window_sec=10, relative=False, noverlap=None, clipping=False):
    bands = FREQUENCY_BANDS
    
    # 入力データの順序（これは eeg_data の順序と一致している必要があります）
    input_order = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']
    
    # 32チャンネルの異なる順序
    channel_orders = [
        # 前頭部から後頭部への配置
        ['Fp1', 'AF3', 'F7', 'F3', 'FC5', 'FC1', 'T7', 'C3', 'CP5', 'CP1',
         'P7', 'P3', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP2',
         'CP6', 'C4', 'T8', 'FC2', 'FC6', 'F4', 'F8', 'AF4', 'Fp2',
         'Fz', 'Cz', 'Pz'],  

        # 左右対称な配置
        ['Fp1', 'Fp2', 'AF3', 'AF4', 'F7', 'F8', 'F3', 'F4', 'FC5', 'FC6',
         'FC1', 'FC2', 'T7', 'T8', 'C3', 'C4', 'CP5', 'CP6', 'CP1', 'CP2',
         'P7', 'P8', 'P3', 'P4', 'PO3', 'PO4', 'O1', 'O2', 'Fz', 'Cz', 
         'Pz', 'Oz']  # 左右対称
    ]
    
    all_features = []
    
    for order in channel_orders:
        features = []
        for trial in eeg_data:
            trial_features = []
            for band in bands.values():
                band_features = []
                for channel in order:
                    channel_idx = input_order.index(channel)  # 入力データの順序に基づいてインデックスを取得
                    band_features.append(bandpower(trial[channel_idx], fs, band, window_sec=window_sec, relative=relative, noverlap=noverlap))
                trial_features.append(band_features)
            features.append(trial_features)
        
        features = np.array(features)
        features = features.reshape(features.shape[0], 5, 32, 1)
        all_features.append(features)
    
    # 3つの特徴量を結合して (none, 5, 14, 3) の形状にする
    combined_features = np.concatenate(all_features, axis=3)
    
    if clipping:
        # 外れ値の除去
        # 99パーセンタイルでクリッピング
        clip_value = np.percentile(combined_features, 99)
        combined_features = np.clip(combined_features, None, clip_value)
        
        # 1パーセンタイルでクリッピング
        lower_clip = np.percentile(combined_features, 1)
        combined_features = np.clip(combined_features, lower_clip, None)
    
    return combined_features


@cache_features
def extract_band_powers_for_2D_32ch_(eeg_data, fs=128, window_sec=10, relative=False, noverlap=None, clipping=False):
    bands = FREQUENCY_BANDS  # 5つの周波数帯域
    
    # 入力データの順序
    input_order = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']
    
    # 電極位置の定義（8x9グリッド）
    electrode_positions = {
        'AF3': (0, 2), 'AF4': (0, 6), 'Fp1': (0, 3), 'Fp2': (0, 5),
        'F7': (1, 0), 'F3': (1, 2), 'Fz': (1, 4), 'F4': (1, 6), 'F8': (1, 8),
        'FC5': (2, 1), 'FC1': (2, 3), 'FC2': (2, 5), 'FC6': (2, 7),
        'T7': (3, 0), 'C3': (3, 2), 'Cz': (3, 4), 'C4': (3, 6), 'T8': (3, 8),
        'CP5': (4, 1), 'CP1': (4, 3), 'CP2': (4, 5), 'CP6': (4, 7),
        'P7': (5, 0), 'P3': (5, 2), 'Pz': (5, 4), 'P4': (5, 6), 'P8': (5, 8),
        'PO3': (6, 3), 'PO4': (6, 5),
        'O1': (7, 3), 'Oz': (7, 4), 'O2': (7, 5)
    }
    
    # 出力配列の初期化 (trials, 8, 9, bands)
    n_trials = len(eeg_data)
    features = np.zeros((n_trials, 8, 9, len(bands)))
    
    # プログレスバーの設定
    pbar = tqdm(total=n_trials * len(bands) * len(electrode_positions), 
                desc="特徴量抽出中")
    
    # 各トライアルについて処理
    for trial_idx, trial in enumerate(eeg_data):
        # 各周波数帯域について処理
        for band_idx, band in enumerate(bands.values()):
            # 各電極について処理
            for channel in input_order:
                if channel in electrode_positions:
                    channel_idx = input_order.index(channel)
                    power = bandpower(trial[channel_idx], fs, band, 
                                    window_sec=window_sec, 
                                    relative=relative, 
                                    noverlap=noverlap)
                    i, j = electrode_positions[channel]
                    features[trial_idx, i, j, band_idx] = power
                    pbar.update(1)
    
    pbar.close()
    
    if clipping:
        print("外れ値の除去を実行中...")
        # 外れ値の除去
        clip_value = np.percentile(features, 99)
        features = np.clip(features, None, clip_value)
        
        lower_clip = np.percentile(features, 1)
        features = np.clip(features, lower_clip, None)
    print(f"特徴量の形状: {features.shape}")
    return features

@cache_features
def extract_band_powers_for_2D_2ch(eeg_data, fs=128, window_sec=10, relative=False, noverlap=None, clipping=False):
    bands = FREQUENCY_BANDS
    
    # 入力データの順序（これは eeg_data の順序と一致している必要があります）
    input_order = ['Fp1', 'Fp2']
    
    # 32チャンネルの異なる順序
    channel_orders = [
        # 前頭部から後頭部への配置
        ['Fp1', 'Fp2'],  
    ]
    
    all_features = []
    
    for order in channel_orders:
        features = []
        for trial in eeg_data:
            trial_features = []
            for band in bands.values():
                band_features = []
                for channel in order:
                    channel_idx = input_order.index(channel)  # 入力データの順序に基づいてインデックスを取得
                    band_features.append(bandpower(trial[channel_idx], fs, band, window_sec=window_sec, relative=relative, noverlap=noverlap))
                trial_features.append(band_features)
            features.append(trial_features)
        
        features = np.array(features)
        features = features.reshape(features.shape[0], 5, 2, 1)
        all_features.append(features)
    
    # 3つの特徴量を結合して (none, 5, 14, 3) の形状にする
    combined_features = np.concatenate(all_features, axis=3)
    
    if clipping:
        # 外れ値の除去
        # 99パーセンタイルでクリッピング
        clip_value = np.percentile(combined_features, 99)
        combined_features = np.clip(combined_features, None, clip_value)
        
        # 1パーセンタイルでクリッピング
        lower_clip = np.percentile(combined_features, 1)
        combined_features = np.clip(combined_features, lower_clip, None)
    
    return combined_features

@cache_features
def extract_temporal_band_powers_2d(eeg_data, fs=128, window_sec=2, step_sec=1, relative=False, noverlap=None):
    """
    時系列を考慮した特徴量抽出
    Returns: (n_trials, n_timesteps, n_bands, n_channels, n_views)
    
    Parameters:
    -----------
    eeg_data : numpy.ndarray
        入力EEGデータ (n_trials, n_channels, n_samples)
    fs : int
        サンプリング周波数
    window_sec : float
        各時間窓の長さ（秒）
    step_sec : float
        時間窓のスライド幅（秒）
    relative : bool
        相対パワーを計算するかどうか
    noverlap : int
        バンドパワー計算時のオーバーラップ
    """
    bands = FREQUENCY_BANDS
    
    # 時間窓のサンプル数を計算
    window_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)
    
    all_trial_features = []
    
    for trial in tqdm(eeg_data, desc="Processing trials"):
        trial_length = trial.shape[1]
        n_steps = (trial_length - window_samples) // step_samples + 1
        
        timestep_features = []
        for step in range(n_steps):
            start_idx = step * step_samples
            end_idx = start_idx + window_samples
            
            if end_idx > trial_length:
                break
                
            window = trial[:, start_idx:end_idx]
            
            # 各時間窓でバンドパワーを計算
            band_features = []
            for band_name, band_range in bands.items():
                channel_features = []
                for ch in range(trial.shape[0]):
                    power = bandpower(window[ch], fs, band_range,
                                   window_sec=window_sec,
                                   relative=relative,
                                   noverlap=noverlap)
                    channel_features.append(power)
                band_features.append(channel_features)
            
            timestep_features.append(band_features)
        
        all_trial_features.append(timestep_features)
    
    # (n_trials, n_timesteps, n_bands, n_channels, 1)の形状に変換
    features = np.array(all_trial_features)[..., np.newaxis]
    
    return features

@cache_features
def extract_temporal_band_powers_1d(eeg_data, fs=128, window_sec=2, step_sec=1, relative=False, noverlap=None):
    """
    時系列を考慮した特徴量抽出（1D形式）
    Returns: (n_trials, n_timesteps, features)
    
    Parameters:
    -----------
    eeg_data : numpy.ndarray
        入力EEGデータ (n_trials, n_channels, n_samples)
    fs : int
        サンプリング周波数
    window_sec : float
        各時間窓の長さ（秒）
    step_sec : float
        時間窓のスライド幅（秒）
    relative : bool
        相対パワーを計算するかどうか
    noverlap : int
        バンドパワー計算時のオーバーラップ
    """
    bands = FREQUENCY_BANDS
    
    # 時間窓のサンプル数を計算
    window_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)
    
    all_trial_features = []
    
    for trial in tqdm(eeg_data, desc="Processing trials"):
        trial_length = trial.shape[1]
        n_steps = (trial_length - window_samples) // step_samples + 1
        
        timestep_features = []
        for step in range(n_steps):
            start_idx = step * step_samples
            end_idx = start_idx + window_samples
            
            if end_idx > trial_length:
                break
                
            window = trial[:, start_idx:end_idx]
            
            # 各時間窓での特徴量を1次元に
            window_features = []
            for band_name, band_range in bands.items():
                for ch in range(trial.shape[0]):
                    power = bandpower(window[ch], fs, band_range,
                                   window_sec=window_sec,
                                   relative=relative,
                                   noverlap=noverlap)
                    window_features.append(power)
            
            timestep_features.append(window_features)
        
        all_trial_features.append(timestep_features)
    
    # (n_trials, n_timesteps, n_features)の形状に変換
    features = np.array(all_trial_features)
    features = np.array(all_trial_features)[..., np.newaxis]
    
    print(f"Feature shape: {features.shape}")  # 形状の確認
    # features.shape は (n_trials, n_timesteps, n_bands * n_channels) となる
    
    return features


def convert_log_features(features):
    """
    0値はそのままで、それ以外の値をlog変換する
    
    Parameters:
    -----------
    features : numpy.ndarray
        入力特徴量
    
    Returns:
    --------
    numpy.ndarray
        変換後の特徴量
    """
    log_features = np.log(features)
    return log_features