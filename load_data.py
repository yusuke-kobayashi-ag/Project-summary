import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

DEAP_EEG_CHANNELS = [
        'Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2'
    ]

def load_deap_data_npy(file_path, selected_channels=None, preprocessing='eeg'):
    """
    Parameters:
    -----------
    preprocessing : str
        'eeg': EEG標準的な前処理（試行ごとのベースライン補正とDCオフセット除去）
        'none': 前処理なし
    """
    # DEAPデータセットのチャンネル順序
    DEAP_EEG_CHANNELS = [
        'Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2'
    ]
    # データ読み込み
    data = np.load(file_path)
    data = data.copy()
    
    # チャンネル選択
    if selected_channels is not None:
        channel_indices = [DEAP_EEG_CHANNELS.index(ch) for ch in selected_channels]
        selected_data = data[:, channel_indices, :]
        used_channels = selected_channels
    else:
        selected_data = data.copy()  # 元データを変更しないようにコピー
        used_channels = DEAP_EEG_CHANNELS

    baseline_samples = int(5 * 128)  # 5秒 × 128 Hz
    """
    # EEG前処理
    if preprocessing == 'eeg':
        # 試行ごとに処理
        for trial in range(selected_data.shape[0]):
            for channel in range(selected_data.shape[1]):
                # 1. ベースライン期間の平均を計算
                baseline_mean = np.mean(selected_data[trial, channel, :baseline_samples])
                
                # 2. ベースライン補正
                selected_data[trial, channel, :] -= baseline_mean
                
                # 3. 全期間の平均（DCオフセット）を除去
                #signal_mean = np.mean(selected_data[trial, channel, baseline_samples:])
                #selected_data[trial, channel, baseline_samples:] -= signal_mean
                
                # 4. 0平均単位分散への標準化
                #signal = selected_data[trial, channel, :]
                #std = np.std(signal)
                #if std != 0:
                    #selected_data[trial, channel, :] = signal / std
    """
    # ベースライン期間の除去（必須処理）
    selected_data = selected_data[:, :, baseline_samples:]
        
    
    # ラベルの読み込み
    subject_id = int(os.path.basename(file_path)[1:3])
    ratings_path = os.path.join(os.path.dirname(os.path.dirname(file_path)),'metadata_csv', 'participant_ratings.csv')
    ratings = pd.read_csv(ratings_path)
    subject_ratings = ratings[ratings['Participant_id'] == subject_id]
    labels = subject_ratings[['Valence', 'Arousal']].values

    channel_info_dict = {
        'names': used_channels,
        'sampling_rate': 128,
        'total_channels': len(used_channels),
        'eeg_channels': DEAP_EEG_CHANNELS
    }
    
    return selected_data, labels, channel_info_dict



if __name__ == "__main__":
    # テスト用のファイルパス
    test_file_path = "data/data_raw_preprocessed/s01.npy"  # 適切なパスに変更してください
    
    # テスト用のチャンネル選択（例：前頭部のチャンネル）
    test_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'Fz']
    
    try:
        # データ読み込みテスト
        data, labels, channel_info = load_deap_data_npy(
            file_path=test_file_path,
            selected_channels=test_channels,
            preprocessing='eeg'
        )
        
        print(f"データ形状: {data.shape}")
        print(f"ラベル形状: {labels.shape}")
        print(f"使用チャンネル: {channel_info['names']}")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")