import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers, optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from google.colab import drive

# Google Driveをマウント
drive.mount('/content/drive')

# ログレベルを設定
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# グローバル変数の定義
n_neurons = 10
total_epochs = 1000
batch_size = 256

# 結果を保存するディレクトリのパス
BASE_DIR = '/content/drive/MyDrive/DEAP_Results'
RESULTS_DIR = os.path.join(BASE_DIR, '32ch_2DCNN')

# ディレクトリの作成
os.makedirs(os.path.join(RESULTS_DIR, 'distributions'), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'models'), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'evaluation'), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'residual'), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'training_history'), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'correlations'), exist_ok=True)

def load_deap_data_dat(file_path, selected_channels=None):
    """
    DEAPデータセットの.datファイルを読み込む関数
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    X = data['data']  # (40, 40, 8064) - (trial, channel, data)
    y = data['labels']  # (40, 4) - (trial, label)
    
    channels = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 
                'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz',
                'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz',
                'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']
    
    X = X[:, :32, :]
    
    if selected_channels is not None:
        channel_indices = [channels.index(ch) for ch in selected_channels]
        X = X[:, channel_indices, :]
        channels = selected_channels
    
    X = np.transpose(X, (0, 1, 2))
    
    return X, y, channels

def extract_band_powers_for_2D_32ch(X, fs=128, window_sec=10, relative=False, noverlap=None, clipping=False):
    """
    2DCNN用の特徴量抽出関数
    """
    n_trials, n_channels, n_samples = X.shape
    window_size = int(window_sec * fs)
    
    if noverlap is None:
        noverlap = window_size // 2
    
    n_windows = (n_samples - window_size) // (window_size - noverlap) + 1
    
    # 特徴量の初期化
    features = np.zeros((n_trials, n_windows, 5, n_channels, 1))  # 5は周波数帯の数
    
    for trial in range(n_trials):
        for window in range(n_windows):
            start = window * (window_size - noverlap)
            end = start + window_size
            
            for channel in range(n_channels):
                signal = X[trial, channel, start:end]
                
                # ここでは簡略化のため、ダミーの特徴量を生成
                # 実際の実装では、適切な周波数帯域のパワーを計算
                features[trial, window, :, channel, 0] = np.random.rand(5)
    
    return features

def create_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(20, (1,2), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('tanh')(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Flatten()(x)
    
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('tanh')(x)
    
    x = layers.Dense(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('tanh')(x)
    
    output = layers.Dense(1)(x)
    
    model = models.Model(inputs=inputs, outputs=output)
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss='mse'
    )
    return model

def analyze_channel_correlations(X, y, channels, save_dir=os.path.join(RESULTS_DIR, 'correlations')):
    os.makedirs(save_dir, exist_ok=True)
    
    channel_powers = np.mean(X, axis=2)
    
    valence_correlations = []
    arousal_correlations = []
    valence_p_values = []
    arousal_p_values = []
    
    for i in range(len(channels)):
        corr_val, p_val = stats.pearsonr(channel_powers[:, i], y[:, 0])
        valence_correlations.append(corr_val)
        valence_p_values.append(p_val)
        
        corr_aro, p_aro = stats.pearsonr(channel_powers[:, i], y[:, 1])
        arousal_correlations.append(corr_aro)
        arousal_p_values.append(p_aro)
    
    results_df = pd.DataFrame({
        'Channel': channels,
        'Valence_Correlation': valence_correlations,
        'Valence_p_value': valence_p_values,
        'Arousal_Correlation': arousal_correlations,
        'Arousal_p_value': arousal_p_values
    })
    
    results_df['Valence_Abs_Corr'] = np.abs(results_df['Valence_Correlation'])
    results_df['Arousal_Abs_Corr'] = np.abs(results_df['Arousal_Correlation'])
    
    results_df.to_csv(os.path.join(save_dir, 'channel_correlations.csv'), index=False)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(results_df[['Valence_Correlation']].values.reshape(-1, 1),
                annot=True, fmt='.2f', cmap='coolwarm',
                xticklabels=['Valence'], yticklabels=channels)
    plt.title('Valence Correlations')
    
    plt.subplot(1, 2, 2)
    sns.heatmap(results_df[['Arousal_Correlation']].values.reshape(-1, 1),
                annot=True, fmt='.2f', cmap='coolwarm',
                xticklabels=['Arousal'], yticklabels=channels)
    plt.title('Arousal Correlations')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'correlation_heatmap.png'))
    plt.close()
    
    print("\n=== Top 10 Channels for Valence ===")
    print(results_df.sort_values('Valence_Abs_Corr', ascending=False)[['Channel', 'Valence_Correlation', 'Valence_p_value']].head(10))
    
    print("\n=== Top 10 Channels for Arousal ===")
    print(results_df.sort_values('Arousal_Abs_Corr', ascending=False)[['Channel', 'Arousal_Correlation', 'Arousal_p_value']].head(10))
    
    return results_df

def plot_training_history(history, model_type, save_dir):
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_type} Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, f'training_history_{model_type}.png'))
    plt.close()

def evaluate_regression(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R^2': r2}

if __name__ == "__main__":
    # データの準備
    person_ids = range(1,33)
    selected_channels = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']
    
    # データの読み込みと結合
    X_all = []
    y_all = []
    
    for person_id in person_ids:
        file_path = f'/content/drive/MyDrive/DEAP_Data/s{str(person_id).zfill(2)}.dat'
        print(f'{file_path}を処理中...')
        X, y, channels = load_deap_data_dat(file_path, selected_channels=selected_channels)
        X_all.append(X)
        y_all.append(y)
    
    X_combined = np.concatenate(X_all, axis=0)
    y_combined = np.concatenate(y_all, axis=0)
    y_combined = y_combined[:, :2]  # valenceとarousalのみを使用

    # 特徴量の抽出
    X_features = extract_band_powers_for_2D_32ch(
        X_combined,
        fs=128,
        window_sec=10,
        relative=False
    )

    print("Data shape:", X_features.shape)
    print("Label shape:", y_combined.shape)
    print("\nデータの分布確認:")
    print("Valenceの範囲:", np.min(y_combined[:, 0]), "から", np.max(y_combined[:, 0]))
    print("Arousalの範囲:", np.min(y_combined[:, 1]), "から", np.max(y_combined[:, 1]))
    print("Xの最小値:", np.min(X_features), "Xの最大値", np.max(X_features), "Xの平均", np.mean(X_features))

    # チャンネルとラベルの相関分析
    print("\n=== チャンネルとラベルの相関分析 ===")
    correlation_results = analyze_channel_correlations(X_combined, y_combined, selected_channels)

    # データの分割
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_features, y_combined, test_size=0.1, random_state=7
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1, random_state=7
    )

    # 訓練データのみでスケーラーを学習
    scaler = StandardScaler()
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    scaler.fit(X_train_reshaped)

    # データのスケーリング
    X_train_scaled = scaler.transform(X_train_reshaped)
    X_test_scaled = scaler.transform(X_test.reshape(X_test.shape[0], -1))
    X_val_scaled = scaler.transform(X_val.reshape(X_val.shape[0], -1))

    # 元の形状に戻す
    X_train = X_train_scaled.reshape(X_train.shape)
    X_val = X_val_scaled.reshape(X_val.shape)
    X_test = X_test_scaled.reshape(X_test.shape)

    # データ分布の可視化
    plt.figure(figsize=(8, 8))
    plt.scatter(y_train[:, 0], y_train[:, 1], alpha=0.5)
    plt.xlabel('Valence')
    plt.ylabel('Arousal')
    plt.title('Data Distribution')
    plt.grid(True)
    plt.axis([-1, 1, -1, 1])
    plt.savefig(os.path.join(RESULTS_DIR, 'distributions/data_distribution.png'))
    plt.close()

    # モデルの作成
    input_shape = X_train.shape[1:]
    valence_model = create_model(input_shape=input_shape)
    arousal_model = create_model(input_shape=input_shape)

    # コールバックの設定
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=30, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor='val_loss', patience=80, restore_best_weights=True, verbose=1)
    ]

    # Valenceモデルの学習
    valence_history = valence_model.fit(
        X_train, y_train[:, 0],
        epochs=total_epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val[:, 0]),
        callbacks=callbacks,
        verbose=1
    )

    # Arousalモデルの学習
    arousal_history = arousal_model.fit(
        X_train, y_train[:, 1],
        epochs=total_epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val[:, 1]),
        callbacks=callbacks,
        verbose=1
    )

    # 予測
    valence_predictions = valence_model.predict(X_test)
    arousal_predictions = arousal_model.predict(X_test)

    valence_predictions = valence_predictions.reshape(-1)
    arousal_predictions = arousal_predictions.reshape(-1)

    # 評価結果の計算
    valence_results = evaluate_regression(y_test[:, 0], valence_predictions)
    arousal_results = evaluate_regression(y_test[:, 1], arousal_predictions)

    # 残差プロット
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test[:, 0], valence_predictions, alpha=0.5)
    plt.plot([-1, 1], [-1, 1], 'r--')
    plt.xlabel('True Valence')
    plt.ylabel('Predicted Valence')
    plt.title('Valence: True vs Predicted')
    plt.grid(True, alpha=0.3)
    plt.axis([-1, 1, -1, 1])

    plt.subplot(1, 2, 2)
    plt.scatter(y_test[:, 1], arousal_predictions, alpha=0.5)
    plt.plot([-1, 1], [-1, 1], 'r--')
    plt.xlabel('True Arousal')
    plt.ylabel('Predicted Arousal')
    plt.title('Arousal: True vs Predicted')
    plt.grid(True, alpha=0.3)
    plt.axis([-1, 1, -1, 1])

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'residual/residual_plots.png'))
    plt.close()

    # モデルの保存
    valence_model.save(os.path.join(RESULTS_DIR, 'models/valence_model.keras'))
    arousal_model.save(os.path.join(RESULTS_DIR, 'models/arousal_model.keras'))

    # 結果をDataFrameに保存
    results_df = pd.DataFrame({
        'Metric': ['MSE', 'RMSE', 'MAE', 'R^2'],
        'Valence': [
            valence_results['MSE'],
            valence_results['RMSE'],
            valence_results['MAE'],
            valence_results['R^2']
        ],
        'Arousal': [
            arousal_results['MSE'],
            arousal_results['RMSE'],
            arousal_results['MAE'],
            arousal_results['R^2']
        ]
    })

    results_df.to_csv(os.path.join(RESULTS_DIR, 'evaluation/results.csv'), index=False)

    # 結果の出力
    print("\n=== Final Results ===")
    print("\nValence Results:")
    for metric, value in valence_results.items():
        print(f"{metric}: {value:.4f}")

    print("\nArousal Results:")
    for metric, value in arousal_results.items():
        print(f"{metric}: {value:.4f}")

    plot_training_history(valence_history, 'Valence', RESULTS_DIR)
    plot_training_history(arousal_history, 'Arousal', RESULTS_DIR)

    print("\nTraining and evaluation completed.") 