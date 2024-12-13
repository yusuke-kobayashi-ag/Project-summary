import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import json
import os
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from load_data import DEAP_EEG_CHANNELS
import mne
from mne.channels import make_standard_montage


def plot_importance_topomap(importance_scores, title, save_path):
    """チャンネルの重要度をtopomapで表示"""
    # チャンネル情報の作成
    ch_names = list(importance_scores.keys())
    ch_types = ['eeg'] * len(ch_names)
    
    # モンタージュの作成（10-20システム）
    info = mne.create_info(ch_names=ch_names, sfreq=128, ch_types=ch_types)
    montage = mne.channels.make_standard_montage('biosemi32')
    info.set_montage(montage, match_case=False)
    
    # データの準備
    data = np.array(list(importance_scores.values()))
    
    # 値の範囲を設定
    vlim = np.max(abs(data))
    
    # topomapの作成
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    
    im, _ = mne.viz.plot_topomap(data, info, 
                             axes=ax,
                             show=False,
                             size=2,
                             extrapolate='box',  # センサー付近のみ補間
                             sphere=(0.0, 0.0, 0.0, 0.09),
                             names=ch_names,
                          
                             sensors=True,)
    
    for text in ax.texts:
        text.set_fontsize(12)
    

    ax.set_title(f'Channel Importance Topomap for {title}')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig(save_path)
    plt.close(fig)

# 結果保存用のディレクトリ作成
os.makedirs('./results/32ch_2DCNN_gauss/analysis', exist_ok=True)

def mae(y_true, y_pred):
    """Mean Absolute Error"""
    return mean_absolute_error(y_true, y_pred)

# decode_prediction関数を追加
def decode_prediction(pred_dist):
    centers = np.linspace(-1, 1, 10)  # n_neurons=10
    if len(pred_dist.shape) == 1:
        pred_dist = pred_dist[np.newaxis, :]
    power = 2
    emphasized_dist = np.power(pred_dist, power)
    normalized_dist = emphasized_dist / np.sum(emphasized_dist, axis=1, keepdims=True)
    predictions = np.sum(normalized_dist * centers, axis=1)
    return predictions

def analyze_channel_importance(model, X_val, y_val, channels):
    """チャンネルの重要度を分析"""
    predictions = model.predict(X_val)
    predictions = decode_prediction(predictions)
    base_score = mae(y_val, predictions)
    importance_scores = {}
    
    for i, channel in enumerate(channels):
        X_modified = X_val.copy()
        X_modified[:, :, i, :] = 0  # チャンネルを0に
        
        modified_pred = model.predict(X_modified)
        modified_pred = decode_prediction(modified_pred)
        modified_score = mae(y_val, modified_pred)
        importance_scores[channel] = modified_score - base_score
    
    return importance_scores

def analyze_frequency_importance(model, X_val, y_val, freq_bands):
    """周波数帯域の重要度を分析"""
    predictions = model.predict(X_val)
    predictions = decode_prediction(predictions)
    base_score = mae(y_val, predictions)
    importance_scores = {}
    
    for freq_idx, band_name in enumerate(freq_bands):
        X_modified = X_val.copy()
        X_modified[:, freq_idx, :, :] = 0  # 周波数帯域を0に
        
        modified_pred = model.predict(X_modified)
        modified_pred = decode_prediction(modified_pred)
        modified_score = mae(y_val, modified_pred)
        importance_scores[band_name] = modified_score - base_score
    
    return importance_scores

def plot_channel_importance(importance_scores, title, save_path):
    """チャンネルの重要度をプロット"""
    plt.figure(figsize=(15, 5))
    channels = list(importance_scores.keys())
    scores = list(importance_scores.values())
    
    # 重要度でソート
    sorted_indices = np.argsort(scores)
    channels = [channels[i] for i in sorted_indices]
    scores = [scores[i] for i in sorted_indices]
    # 修正：符号の反転を削除
    
    plt.bar(channels, scores)
    plt.xticks(rotation=45)
    plt.title(f'Channel Importance for {title}')
    plt.ylabel('Importance Score (mae increase)')  # 説明を修正
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_frequency_importance(importance_scores, title, save_path):
    """周波数帯域の重要度をプロット"""
    plt.figure(figsize=(10, 5))
    bands = list(importance_scores.keys())
    scores = list(importance_scores.values())
    # 修正：符号の反転を削除
    
    plt.bar(bands, scores)
    plt.title(f'Frequency Band Importance for {title}')
    plt.ylabel('Importance Score (mae increase)')  # 説明を修正
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    # モデルの読み込み
    n_neurons = 10  # train_model_32ch_2DCNN_gauss.pyと同じ値を使用
    sigma = 0.25
    
    valence_model = keras.models.load_model(
        f'./results/32ch_2DCNN_gauss/models/neuron{n_neurons}_sigma{sigma}_valence_model.keras',
        compile=False
    )
    
    arousal_model = keras.models.load_model(
        f'./results/32ch_2DCNN_gauss/models/neuron{n_neurons}_sigma{sigma}_arousal_model.keras',
        compile=False
    )
    
    # チャンネル名とバンド名の定義
    channels = DEAP_EEG_CHANNELS
    
    freq_bands = ['theta', 'alpha', 'low_beta', 'high_beta', 'gamma']
    
    # データの読み込み（検証データを使用）
    # Note: 実際のデータ読み込みコードに置き換えてください
    X_val = np.load('./data/X_val.npy')  # 保存済みの検証データ
    y_val = np.load('./data/y_val.npy')
    
    # 重要度分析の実行
    print("Analyzing feature importance...")
    
    # Valenceの分析
    valence_channel_importance = analyze_channel_importance(
        valence_model, X_val, y_val[:, 0], channels
    )
    valence_freq_importance = analyze_frequency_importance(
        valence_model, X_val, y_val[:, 0], freq_bands
    )
    
    # Arousalの分析
    arousal_channel_importance = analyze_channel_importance(
        arousal_model, X_val, y_val[:, 1], channels
    )
    arousal_freq_importance = analyze_frequency_importance(
        arousal_model, X_val, y_val[:, 1], freq_bands
    )
    
    # 結果の可視化
    plot_channel_importance(
        valence_channel_importance,
        'Valence',
        './results/32ch_2DCNN_gauss/analysis/valence_channel_importance.png'
    )
    
    plot_channel_importance(
        arousal_channel_importance,
        'Arousal',
        './results/32ch_2DCNN_gauss/analysis/arousal_channel_importance.png'
    )
    
    plot_frequency_importance(
        valence_freq_importance,
        'Valence',
        './results/32ch_2DCNN_gauss/analysis/valence_frequency_importance.png'
    )
    
    plot_frequency_importance(
        arousal_freq_importance,
        'Arousal',
        './results/32ch_2DCNN_gauss/analysis/arousal_frequency_importance.png'
    )
    

    # メイン処理部分に追加
    plot_importance_topomap(
        valence_channel_importance,
        'Valence',
        './results/32ch_2DCNN_gauss/analysis/valence_importance_topomap.png'
    )

    plot_importance_topomap(
        arousal_channel_importance,
        'Arousal',
        './results/32ch_2DCNN_gauss/analysis/arousal_importance_topomap.png'
    )
    
    # 結果の保存
    analysis_results = {
        'channel_importance': {
            'valence': valence_channel_importance,
            'arousal': arousal_channel_importance
        },
        'frequency_importance': {
            'valence': valence_freq_importance,
            'arousal': arousal_freq_importance
        }
    }
    
    with open('./results/32ch_2DCNN_gauss/analysis/feature_importance.json', 'w') as f:
        json.dump(analysis_results, f, indent=4)
    
    print("Analysis completed. Results saved in ./results/32ch_2DCNN_gauss/analysis/")