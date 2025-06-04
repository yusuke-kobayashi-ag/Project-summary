import numpy as np
import os
from load_data import load_deap_data_npy, load_deap_data_dat
from feature_extraction import (
    extract_band_powers_for_1D,
    extract_band_powers_for_2D_14ch,
    extract_band_powers_for_2D_32ch,
    extract_band_powers_for_1D_segments,
    convert_log_features,
    plot_feature_distributions,
    extract_temporal_band_powers_1d,
    extract_temporal_band_powers_2d,
    extract_band_powers_for_2D_2ch
)
from feature_extraction import FREQUENCY_BANDS

def make_data_set(person_ids, model_type='1DCNN', log_transform=False, fs=128, window_sec=10, relative=False, 
                  noverlap=None, selected_channels=None, use_segments=False, overlap_sec=5, 
                  plot_distributions=True, gauss=False, clipping=False, data_format='npy'):
    """
    データセットを作成する関数
    
    Parameters:
    -----------
    person_ids : list
        処理する被験者のID
    model_type : str
        '1DCNN' または '2DCNN'
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
    data_format : str
        'npy'または'dat'。データファイルの形式を指定
    """
    # データの読み込み
    if data_format == 'npy':
        file_paths = [f'data/data_raw_preprocessed/s{str(i).zfill(2)}.npy' for i in person_ids]
        load_func = load_deap_data_npy
    elif data_format == 'dat':
        file_paths = [f'data/data_preprocessed_python/s{str(i).zfill(2)}.dat' for i in person_ids]
        load_func = load_deap_data_dat
    else:
        raise ValueError("data_formatは'npy'または'dat'である必要があります")

    X_all = []
    y_all = []

    for file_path in file_paths:
        print(f'{file_path}を処理中...')
        X, y, channels = load_func(file_path, selected_channels=selected_channels)
        X_all.append(X)
        y_all.append(y)

    X_combined = np.concatenate(X_all, axis=0)
    y_combined = np.concatenate(y_all, axis=0)
    
    if data_format == 'npy':
        y_combined = 2 * ((y_combined - 1)/8) - 1  # スケール変換 [1,9] -> [-1,1]
    else:  # datの場合は既に[-1,1]のスケール
        y_combined = y_combined[:, :2]  # valenceとarousalのみを使用

    # 周波数帯域の分析を実行
    print("周波数帯域の分析を実行中...")
    save_dir = f'./results/{"32ch" if len(selected_channels)==32 else "14ch"}_{model_type}'
    if use_segments:
        save_dir += '_segment'
    if gauss:
        save_dir += '_gauss'
    save_dir += '/frequency_analysis'

    print("特徴量の計算...")
    if use_segments:
        if model_type == '1DCNN':
            band_power_features = extract_band_powers_for_1D_segments(
                X_combined,
                fs=fs,
                window_sec=window_sec,
                overlap_sec=overlap_sec,
                relative=relative,
                selected_channels = selected_channels
            )
            print("セグメント化1DCNN特徴量を計算")
    else:
        # 従来の特徴量抽出処理
        if model_type == '1DCNN':
            band_power_features = extract_band_powers_for_1D(
                X_combined, fs=fs, window_sec=window_sec, relative=relative,
            )
        elif model_type == '2DCNN':
            if len(selected_channels) == 14:
                band_power_features = extract_band_powers_for_2D_14ch(
                    X_combined, fs=fs, window_sec=window_sec,
                    relative=relative, noverlap=noverlap
                )
            elif len(selected_channels) == 32:  # 32チャンネル
                band_power_features = extract_band_powers_for_2D_32ch(
                    X_combined, fs=fs, window_sec=window_sec,
                    relative=relative, noverlap=noverlap,clipping=clipping
                )
            else:
                band_power_features = extract_band_powers_for_2D_2ch(
                    X_combined, fs=fs, window_sec=window_sec,
                    relative=relative, noverlap=noverlap,clipping=clipping
                )

    if plot_distributions:
        save_dir = f'./results/{"32ch" if len(selected_channels)==32 else "14ch"}_{model_type}'
        if use_segments:
            save_dir += '_segment'
        if gauss:
            save_dir += '_gauss'
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

def make_temporal_dataset_2d(person_ids, window_sec=2, step_sec=1, fs=128, relative=False, 
                         selected_channels=None, log_transform=True, plot_distributions=True,
                         data_format='npy'):
    """
    時系列モデル用のデータセット作成関数
    
    Parameters:
    -----------
    person_ids : list
        処理する被験者のID
    window_sec : float
        各時間窓の長さ（秒）
    step_sec : float
        時間窓のスライド幅（秒）
    fs : int
        サンプリング周波数
    relative : bool
        相対バンドパワーを使用するかどうか
    selected_channels : list or None
        使用するチャンネルのリスト
    log_transform : bool
        特徴量に対数変換を適用するかどうか
    plot_distributions : bool
        特徴量の分布を可視化するかどうか
    data_format : str
        'npy'または'dat'。データファイルの形式を指定
    """
    # データの読み込み
    if data_format == 'npy':
        file_paths = [f'data/data_raw_preprocessed/s{str(i).zfill(2)}.npy' for i in person_ids]
        load_func = load_deap_data_npy
    elif data_format == 'dat':
        file_paths = [f'data/data_original/s{str(i).zfill(2)}.dat' for i in person_ids]
        load_func = load_deap_data_dat
    else:
        raise ValueError("data_formatは'npy'または'dat'である必要があります")

    X_all = []
    y_all = []

    for file_path in file_paths:
        print(f'{file_path}を処理中...')
        X, y, channels = load_func(file_path, selected_channels=selected_channels)
        X_all.append(X)
        y_all.append(y)

    X_combined = np.concatenate(X_all, axis=0)
    y_combined = np.concatenate(y_all, axis=0)
    
    if data_format == 'npy':
        y_combined = 2 * ((y_combined - 1)/8) - 1  # スケール変換 [1,9] -> [-1,1]
    else:  # datの場合は既に[-1,1]のスケール
        y_combined = y_combined[:, :2]  # valenceとarousalのみを使用

    # 時系列特徴量の抽出
    print("時系列特徴量の抽出中...")
    features = extract_temporal_band_powers_2d(
        X_combined,
        fs=fs,
        window_sec=window_sec,
        step_sec=step_sec,
        relative=relative
    )

    if plot_distributions:
        save_dir = './results/temporal_model/feature_distributions'
        print(f"{save_dir}に統計情報が保存されます")
        stats = plot_feature_distributions(
            features,
            model_type='temporal',
            save_dir=save_dir
        )
        print("分布の統計情報:")
        print(stats)

    if log_transform:
        print("特徴量のlog変換...")
        features = convert_log_features(features)

    print(f"features_shape:{features.shape}")
    print(f"labels_shape:{y_combined.shape}")

    return features, y_combined

def make_temporal_dataset_1d(person_ids, window_sec=2, step_sec=1, fs=128, relative=False, 
                         selected_channels=None, log_transform=True, plot_distributions=True,
                         data_format='npy'):
    """
    時系列モデル用のデータセット作成関数
    
    Parameters:
    -----------
    person_ids : list
        処理する被験者のID
    window_sec : float
        各時間窓の長さ（秒）
    step_sec : float
        時間窓のスライド幅（秒）
    fs : int
        サンプリング周波数
    relative : bool
        相対バンドパワーを使用するかどうか
    selected_channels : list or None
        使用するチャンネルのリスト
    log_transform : bool
        特徴量に対数変換を適用するかどうか
    plot_distributions : bool
        特徴量の分布を可視化するかどうか
    data_format : str
        'npy'または'dat'。データファイルの形式を指定
    """
    # データの読み込み
    if data_format == 'npy':
        file_paths = [f'data/data_raw_preprocessed/s{str(i).zfill(2)}.npy' for i in person_ids]
        load_func = load_deap_data_npy
    elif data_format == 'dat':
        file_paths = [f'data/data_original/s{str(i).zfill(2)}.dat' for i in person_ids]
        load_func = load_deap_data_dat
    else:
        raise ValueError("data_formatは'npy'または'dat'である必要があります")

    X_all = []
    y_all = []

    for file_path in file_paths:
        print(f'{file_path}を処理中...')
        X, y, channels = load_func(file_path, selected_channels=selected_channels)
        X_all.append(X)
        y_all.append(y)

    X_combined = np.concatenate(X_all, axis=0)
    y_combined = np.concatenate(y_all, axis=0)
    
    if data_format == 'npy':
        y_combined = 2 * ((y_combined - 1)/8) - 1  # スケール変換 [1,9] -> [-1,1]
    else:  # datの場合は既に[-1,1]のスケール
        y_combined = y_combined[:, :2]  # valenceとarousalのみを使用

    # 時系列特徴量の抽出
    print("時系列特徴量の抽出中...")
    features = extract_temporal_band_powers_1d(
        X_combined,
        fs=fs,
        window_sec=window_sec,
        step_sec=step_sec,
        relative=relative
    )

    if plot_distributions:
        save_dir = './results/temporal_model/feature_distributions'
        print(f"{save_dir}に統計情報が保存されます")
        stats = plot_feature_distributions(
            features,
            model_type='temporal',
            save_dir=save_dir
        )
        print("分布の統計情報:")
        print(stats)

    if log_transform:
        print("特徴量のlog変換...")
        features = convert_log_features(features)

    print(f"features_shape:{features.shape}")
    print(f"labels_shape:{y_combined.shape}")

    return features, y_combined

