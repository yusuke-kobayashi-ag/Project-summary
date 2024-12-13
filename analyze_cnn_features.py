import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import mean_squared_error
from load_data import DEAP_EEG_CHANNELS
import mne
import json

def decode_prediction(pred_dist):
    """予測分布から値をデコード"""
    centers = np.linspace(-1, 1, 10)  # n_neurons=10
    if len(pred_dist.shape) == 1:
        pred_dist = pred_dist[np.newaxis, :]
    power = 2
    emphasized_dist = np.power(pred_dist, power)
    normalized_dist = emphasized_dist / np.sum(emphasized_dist, axis=1, keepdims=True)
    predictions = np.sum(normalized_dist * centers, axis=1)
    return predictions

from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

def analyze_with_gradcam(model, X_val, freq_bands, channel_orders, title, save_path):
    """Grad-CAMを使用した分析"""
    # 各サンプルに対してGrad-CAMを計算
    gradcam_maps = []
    for i in range(len(X_val)):
        cam = make_gradcam_heatmap(
            model,
            X_val[i:i+1]  # バッチサイズ1で処理
        )
        gradcam_maps.append(cam)
    
    # 全サンプルの平均を計算
    average_cam = np.mean(gradcam_maps, axis=0)
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # データの形状を確認
    print("Grad-CAM shape:", average_cam.shape)
    
    # 各viewのデータを適切な形状に変換
    view1_data = average_cam[:, :, 0] if len(average_cam.shape) == 3 else average_cam
    view2_data = average_cam[:, :, 1] if len(average_cam.shape) == 3 else average_cam
    
    for i, (view_map, view_name, channels) in enumerate(zip(
        [view1_data, view2_data],
        ['Anatomical Order', 'Symmetrical Order'],
        channel_orders
    )):
        sns.heatmap(
            view_map,
            cmap='RdBu_r',
            center=0,
            xticklabels=channels,
            yticklabels=freq_bands,
            cbar_kws={'label': 'Activation Score'},
            ax=axes[i]
        )
        
        axes[i].set_title(f'{title} - {view_name} (Grad-CAM)')
        axes[i].set_xlabel('Channels')
        axes[i].set_ylabel('Frequency Bands')
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def make_gradcam_heatmap(model, input_data, pred_index=None):
    """Grad-CAMヒートマップを生成（tf-keras-visを使用）"""
    gradcam = Gradcam(
        model,
        model_modifier=ReplaceToLinear(),
        clone=True
    )
    
    def score_function(output):
        if pred_index is None:
            return output
        return output[:, pred_index]
    
    cam = gradcam(
        score_function,
        input_data,
        penultimate_layer=-1
    )
    
    return cam[0]  # バッチの最初の要素のみ返す

def analyze_band_channel_importance(model, X_val, y_val):
    """バンドパワーとチャンネルの重要度分析"""
    print("Input shape:", X_val.shape)  # (samples, bands, channels, views)
    
    # feature_extraction.pyと同じチャンネル順序を使用
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
         'Pz', 'Oz']
    ]
    
    base_pred = model.predict(X_val, verbose=0)
    base_pred = decode_prediction(base_pred)
    base_score = np.sqrt(mean_squared_error(y_val, base_pred))
    
    # 各viewごとの重要度マップ
    importance_maps = []
    
    for view in range(X_val.shape[3]):  # 各viewについて分析
        importance_map = np.zeros((X_val.shape[1], X_val.shape[2]))  # (5, 32)
        
        for b in range(X_val.shape[1]):  # 周波数バンド
            for ch in range(X_val.shape[2]):  # チャンネル
                X_modified = X_val.copy()
                X_modified[:, b, ch, view] = 0  # 特定のviewのみゼロに
                
                modified_pred = model.predict(X_modified, verbose=0)
                modified_pred = decode_prediction(modified_pred)
                modified_score = np.sqrt(mean_squared_error(y_val, modified_pred))
                importance_map[b, ch] = modified_score - base_score
        
        importance_maps.append(importance_map)
    
    return importance_maps[0], importance_maps[1], channel_orders

def plot_importance_analysis(importance_map1, importance_map2, freq_bands, channel_orders, title, save_path):
    """重要度マップの可視化（各view用）"""
    view_names = ['Anatomical Order', 'Symmetrical Order']
    importance_maps = [importance_map1, importance_map2]
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    for i, (imp_map, view_name, channels) in enumerate(zip(importance_maps, view_names, channel_orders)):
        sns.heatmap(imp_map,
                   cmap='RdBu_r',
                   center=0,
                   xticklabels=channels,
                   yticklabels=freq_bands,
                   cbar_kws={'label': 'Importance Score'},
                   ax=axes[i])
        
        axes[i].set_title(f'{title} - {view_name}')
        axes[i].set_xlabel('Channels')
        axes[i].set_ylabel('Frequency Bands')
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compare_view_importance(importance_map1, importance_map2, freq_bands, channel_orders):
    """2つのviewの重要度を比較"""
    view1_mean = np.mean(np.abs(importance_map1))
    view2_mean = np.mean(np.abs(importance_map2))
    
    print("\nView Importance Comparison:")
    print(f"View 1 (Anatomical) mean importance: {view1_mean:.4f}")
    print(f"View 2 (Symmetrical) mean importance: {view2_mean:.4f}")
    print(f"More effective view: {'Anatomical' if view1_mean > view2_mean else 'Symmetrical'}")
    
    # 各周波数帯での比較
    print("\nFrequency Band Comparison:")
    for i, band in enumerate(freq_bands):
        band_view1 = np.mean(np.abs(importance_map1[i]))
        band_view2 = np.mean(np.abs(importance_map2[i]))
        print(f"{band:10}: View1={band_view1:.4f}, View2={band_view2:.4f}")

if __name__ == "__main__":
    # 結果保存用のディレクトリ作成
    os.makedirs('./results/32ch_2DCNN_gauss/cnn_analysis', exist_ok=True)
    
    # モデルの読み込み
    n_neurons = 10
    sigma = 0.25
    
    valence_model = keras.models.load_model(
        f'./results/32ch_2DCNN_gauss/models/neuron{n_neurons}_sigma{sigma}_valence_model.keras',
        compile=False
    )
    
    arousal_model = keras.models.load_model(
        f'./results/32ch_2DCNN_gauss/models/neuron{n_neurons}_sigma{sigma}_arousal_model.keras',
        compile=False
    )
    
    # データの読み込み
    X_val = np.load('./data/X_val.npy')
    y_val = np.load('./data/y_val.npy')
    
    # 周波数帯域の定義
    freq_bands = ['theta', 'alpha', 'low_beta', 'high_beta', 'gamma']
    
    # Valenceの分析
    print("Analyzing Valence model...")
    valence_imp1, valence_imp2, channel_orders = analyze_band_channel_importance(valence_model, X_val, y_val[:, 0])
    plot_importance_analysis(
        valence_imp1,
        valence_imp2,
        freq_bands,
        channel_orders,
        'Valence',
        './results/32ch_2DCNN_gauss/cnn_analysis/valence_importance.png'
    )
    
    # Arousalの分析
    print("Analyzing Arousal model...")
    arousal_imp1, arousal_imp2, channel_orders = analyze_band_channel_importance(arousal_model, X_val, y_val[:, 1])
    plot_importance_analysis(
        arousal_imp1,
        arousal_imp2,
        freq_bands,
        channel_orders,
        'Arousal',
        './results/32ch_2DCNN_gauss/cnn_analysis/arousal_importance.png'
    )

    # Valenceの分析
    print("\nAnalyzing Valence model...")
    valence_imp1, valence_imp2, channel_orders = analyze_band_channel_importance(valence_model, X_val, y_val[:, 0])
    compare_view_importance(valence_imp1, valence_imp2, freq_bands, channel_orders)
    
    # Arousalの分析
    print("\nAnalyzing Arousal model...")
    arousal_imp1, arousal_imp2, channel_orders = analyze_band_channel_importance(arousal_model, X_val, y_val[:, 1])
    compare_view_importance(arousal_imp1, arousal_imp2, freq_bands, channel_orders)
    
    # 結果の保存
    np.save('./results/32ch_2DCNN_gauss/cnn_analysis/valence_importance_view1.npy', valence_imp1)
    np.save('./results/32ch_2DCNN_gauss/cnn_analysis/valence_importance_view2.npy', valence_imp2)
    np.save('./results/32ch_2DCNN_gauss/cnn_analysis/arousal_importance_view1.npy', arousal_imp1)
    np.save('./results/32ch_2DCNN_gauss/cnn_analysis/arousal_importance_view2.npy', arousal_imp2)

     # Valenceの分析
    analyze_with_gradcam(
        valence_model,
        X_val,
        freq_bands,
        channel_orders,
        'Valence',
        './results/32ch_2DCNN_gauss/cnn_analysis/valence_gradcam.png'
    )
    
    # Arousalの分析
    analyze_with_gradcam(
        arousal_model,
        X_val,
        freq_bands,
        channel_orders,
        'Arousal',
        './results/32ch_2DCNN_gauss/cnn_analysis/arousal_gradcam.png'
    )
    
    # 結果の要約を保存
    results_summary = {
        'valence': {
            'view1': {
                'max_importance': float(np.max(valence_imp1)),
                'min_importance': float(np.min(valence_imp1)),
                'mean_importance': float(np.mean(valence_imp1)),
                'most_important_band': freq_bands[np.argmax(np.mean(valence_imp1, axis=1))],
                'most_important_channel': channel_orders[0][np.argmax(np.mean(valence_imp1, axis=0))]
            },
            'view2': {
                'max_importance': float(np.max(valence_imp2)),
                'min_importance': float(np.min(valence_imp2)),
                'mean_importance': float(np.mean(valence_imp2)),
                'most_important_band': freq_bands[np.argmax(np.mean(valence_imp2, axis=1))],
                'most_important_channel': channel_orders[1][np.argmax(np.mean(valence_imp2, axis=0))]
            }
        },
        'arousal': {
            'view1': {
                'max_importance': float(np.max(arousal_imp1)),
                'min_importance': float(np.min(arousal_imp1)),
                'mean_importance': float(np.mean(arousal_imp1)),
                'most_important_band': freq_bands[np.argmax(np.mean(arousal_imp1, axis=1))],
                'most_important_channel': channel_orders[0][np.argmax(np.mean(arousal_imp1, axis=0))]
            },
            'view2': {
                'max_importance': float(np.max(arousal_imp2)),
                'min_importance': float(np.min(arousal_imp2)),
                'mean_importance': float(np.mean(arousal_imp2)),
                'most_important_band': freq_bands[np.argmax(np.mean(arousal_imp2, axis=1))],
                'most_important_channel': channel_orders[1][np.argmax(np.mean(arousal_imp2, axis=0))]
            }
        }
    }
    
    with open('./results/32ch_2DCNN_gauss/cnn_analysis/importance_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=4)
    
    print("Analysis completed. Results saved in ./results/32ch_2DCNN_gauss/cnn_analysis/")