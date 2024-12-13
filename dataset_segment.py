import numpy as np
import os
from load_data import load_deap_data_npy
from feature_extraction_segment import (
    extract_band_powers_for_1D_segments,
    extract_band_powers_for_2D_14ch_segments,
    extract_band_powers_for_2D_32ch_segments,
    convert_log_features,
    plot_feature_distributions
)

def make_segment_dataset(train_person_ids, test_person_ids, model_type='1DCNN', fs=128, window_sec=10, 
                        overlap_sec=5, relative=False, selected_channels=None, log_transform=False, 
                        plot_distributions=True):
    """
    セグメント化されたトレーニングデータと通常のテストデータを作成
    
    Parameters:
    -----------
    train_person_ids : list
        トレーニングデータの被験者ID
    test_person_ids : list
        テストデータの被験者ID
    model_type : str
        '1DCNN' または '2DCNN'
    fs : int
        サンプリング周波数
    window_sec : int
        窓幅（秒）
    overlap_sec : int
        オーバーラップ（秒）
    relative : bool
        相対バンドパワーを使用するかどうか
    selected_channels : list
        使用するチャンネルのリスト
    log_transform : bool
        特徴量に対数変換を適用するかどうか
    plot_distributions : bool
        特徴量分布を可視化するかどうか
    """
    # トレーニングデータの読み込みと処理
    print("トレーニングデータの作成中...")
    train_files = [f'data/data_raw_preprocessed/s{str(i).zfill(2)}.npy' for i in train_person_ids]
    X_train_all = []
    y_train_all = []
    
    for file_path in train_files:
        print(f'{file_path}を処理中...')
        X, y, _ = load_deap_data_npy(file_path, selected_channels=selected_channels)
        X_train_all.append(X)
        y_train_all.append(y)
    
    X_train = np.concatenate(X_train_all, axis=0)
    y_train = np.concatenate(y_train_all, axis=0)
    y_train = 2 * ((y_train - 1)/8) - 1  # スケール変換 [1,9] -> [-1,1]
    
    # テストデータの読み込みと処理
    print("テストデータの作成中...")
    test_files = [f'data/data_raw_preprocessed/s{str(i).zfill(2)}.npy' for i in test_person_ids]
    X_test_all = []
    y_test_all = []
    
    for file_path in test_files:
        print(f'{file_path}を処理中...')
        X, y, _ = load_deap_data_npy(file_path, selected_channels=selected_channels)
        X_test_all.append(X)
        y_test_all.append(y)
    
    X_test = np.concatenate(X_test_all, axis=0)
    y_test = np.concatenate(y_test_all, axis=0)
    y_test = 2 * ((y_test - 1)/8) - 1  # スケール変換 [1,9] -> [-1,1]
    
    # 特徴量抽出とyデータのセグメント化
    print("特徴量の抽出中...")
    window_points = window_sec * fs
    overlap_points = overlap_sec * fs
    stride = window_points - overlap_points
    
    # セグメント数の計算（トレーニングデータのみ）
    train_time_points = X_train.shape[2]
    segments_per_trial_train = (train_time_points - window_points) // stride + 1
    
    # yデータのセグメント化（トレーニングデータのみ）
    y_train_segmented = np.repeat(y_train, segments_per_trial_train, axis=0)
    y_test_segmented = y_test  # テストデータはセグメント化しない
    
    if model_type == '1DCNN':
        # トレーニングデータの特徴量抽出
        X_train_features = extract_band_powers_for_1D_segments(
            X_train, fs=fs, window_sec=window_sec,
            overlap_sec=overlap_sec, relative=relative
        )
        
        # テストデータを10秒ごとに分割して平均
        n_segments = X_test.shape[2] // (window_sec * fs)
        X_test_segments = []
        for i in range(n_segments):
            start = i * window_sec * fs
            end = start + window_sec * fs
            segment_features = extract_band_powers_for_1D_segments(
                X_test[:, :, start:end], 
                fs=fs, 
                window_sec=window_sec,
                overlap_sec=0, 
                relative=relative
            )
            X_test_segments.append(segment_features)
        # 全セグメントの平均を取る
        X_test_features = np.mean(np.stack(X_test_segments, axis=0), axis=0)
        
    else:  # 2DCNN
        if len(selected_channels) == 14:
            # トレーニングデータの特徴量抽出
            X_train_features = extract_band_powers_for_2D_14ch_segments(
                X_train, fs=fs, window_sec=window_sec,
                overlap_sec=overlap_sec, relative=relative
            )
            
            # テストデータを10秒ごとに分割して平均
            n_segments = X_test.shape[2] // (window_sec * fs)
            X_test_segments = []
            for i in range(n_segments):
                start = i * window_sec * fs
                end = start + window_sec * fs
                segment_features = extract_band_powers_for_2D_14ch_segments(
                    X_test[:, :, start:end],
                    fs=fs,
                    window_sec=window_sec,
                    overlap_sec=0,
                    relative=relative
                )
                X_test_segments.append(segment_features)
            # 全セグメントの平均を取る
            X_test_features = np.mean(np.stack(X_test_segments, axis=0), axis=0)
            
        else:
            # トレーニングデータの特徴量抽出
            X_train_features = extract_band_powers_for_2D_32ch_segments(
                X_train, fs=fs, window_sec=window_sec,
                overlap_sec=overlap_sec, relative=relative
            )
            
            # テストデータを10秒ごとに分割して平均
            n_segments = X_test.shape[2] // (window_sec * fs)
            X_test_segments = []
            for i in range(n_segments):
                start = i * window_sec * fs
                end = start + window_sec * fs
                segment_features = extract_band_powers_for_2D_32ch_segments(
                    X_test[:, :, start:end],
                    fs=fs,
                    window_sec=window_sec,
                    overlap_sec=0,
                    relative=relative
                )
                X_test_segments.append(segment_features)
            # 全セグメントの平均を取る
            X_test_features = np.mean(np.stack(X_test_segments, axis=0), axis=0)
    
    if log_transform:
        print("特徴量のlog変換...")
        X_train_features = convert_log_features(X_train_features)
        X_test_features = convert_log_features(X_test_features)
    
    if plot_distributions:
        save_dir = f'./results/{"32ch" if len(selected_channels)==32 else "14ch"}_{model_type}_segment/feature_distributions'
        print(f"{save_dir}に統計情報を保存...")
        plot_feature_distributions(X_train_features, model_type=model_type, save_dir=save_dir)
    
    print(f"トレーニングデータ: {X_train_features.shape}")
    print(f"トレーニングラベル: {y_train_segmented.shape}")
    print(f"テストデータ: {X_test_features.shape}")
    print(f"テストラベル: {y_test_segmented.shape}")
    
    return X_train_features, y_train_segmented, X_test_features, y_test_segmented