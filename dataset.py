import numpy as np
import os
from load_data import load_deap_data_npy
from feature_extraction import (
    extract_band_powers_for_1D,
    extract_band_powers_for_2D_14ch,
    extract_band_powers_for_2D_32ch,
    extract_band_powers_for_1D_segments,
    convert_log_features,
    plot_feature_distributions
)

def make_data_set(person_ids, model_type='1DCNN',  log_transform=False, fs=128, window_sec=10, relative=False, noverlap=None, selected_channels=None,use_segments=False, overlap_sec=5, plot_distributions=True):
    """
    データセットを作成する関数
    
    Parameters:
    -----------
    person_ids : list
        処理する被験者のID
    model_type : str
        '1DCNN' または '2DCNN'
    standardize : bool
        特徴量を標準化するかどうか
    log_transform : bool
        特徴量に対数変換を適用するかどうか
    fs : int
        サンプリング周波数
    window_sec : int
        窓幅（秒）
    relative : bool
        相対バンドパワーを使用するかどうか
    noverlap : int or None
        オーバーラップサンプル数（従来の特徴量抽出用）
    selected_channels : list or None
        使用するチャンネルのリスト。Noneの場合は32チャンネル全て使用
    use_segments : bool
        セグメント化処理を使用するかどうか
    overlap_sec : int
        セグメント化時のオーバーラップ（秒）
    """
    # データの読み込み
    file_paths = [f'data/data_raw_preprocessed/s{str(i).zfill(2)}.npy' for i in person_ids]
    X_all = []
    y_all = []

    for file_path in file_paths:
        print(f'{file_path}を処理中...')
        X, y, channels = load_deap_data_npy(file_path, selected_channels=selected_channels)
        X_all.append(X)
        y_all.append(y)

    X_combined = np.concatenate(X_all, axis=0)
    y_combined = np.concatenate(y_all, axis=0)
    y_combined = 2 * ((y_combined - 1)/8) - 1  # スケール変換 [1,9] -> [-1,1]

    print("特徴量の計算...")
    if use_segments:
        if model_type == '1DCNN':
            band_power_features = extract_band_powers_for_1D_segments(
                X_combined,
                fs=fs,
                window_sec=window_sec,
                overlap_sec=overlap_sec,
                relative=relative
            )
            print("セグメント化1DCNN特徴量を計算")
    else:
        # 従来の特徴量抽出処理
        if model_type == '1DCNN':
            band_power_features = extract_band_powers_for_1D(
                X_combined, fs=fs, window_sec=window_sec, relative=relative
            )
        elif model_type == '2DCNN':
            if len(selected_channels) == 14:
                band_power_features = extract_band_powers_for_2D_14ch(
                    X_combined, fs=fs, window_sec=window_sec,
                    relative=relative, noverlap=noverlap
                )
            else:  # 32チャンネル
                band_power_features = extract_band_powers_for_2D_32ch(
                    X_combined, fs=fs, window_sec=window_sec,
                    relative=relative, noverlap=noverlap
                )
    if plot_distributions:
        save_dir = f'./results/{"32ch" if len(selected_channels)==32 else "14ch"}_{model_type}'
        if use_segments:
            save_dir += '_segment'
        save_dir += '/feature_distributions'
        print(f"{save_dir}に統計情報が保存されました。")
        
        print("特徴量分布の可視化...")
        stats = plot_feature_distributions(
            band_power_features,
            model_type=model_type,
            save_dir=save_dir
        )
        print("分布の統計情報:")
        print(stats)

    if log_transform:
        print("特徴量のlog変換...")
        band_power_features = convert_log_features(
            band_power_features
        )

    print(f"features_shape:{band_power_features.shape}")
    print(f"labels_shape:{y_combined.shape}")

    return band_power_features, y_combined