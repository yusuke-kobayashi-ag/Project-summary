import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler,RobustScaler, MinMaxScaler
from dataset import make_temporal_dataset

# GPU使用状況の詳細表示
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

# メモリ使用の最適化（必要に応じて）
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled")
    except RuntimeError as e:
        print(e)

# グローバル変数の定義
n_neurons = 10
sigma = 0.25
total_epochs = 1000
batch_size = 256


# ディレクトリの作成
os.makedirs('./results/32ch_2DCNN_gauss_temp/distributions', exist_ok=True)
os.makedirs('./results/32ch_2DCNN_gauss_temp/models', exist_ok=True)
os.makedirs('./results/32ch_2DCNN_gauss_temp/evaluation', exist_ok=True)
os.makedirs('./results/32ch_2DCNN_gauss_temp/residual', exist_ok=True)
os.makedirs('./results/32ch_2DCNN_gauss_temp/training_history', exist_ok=True)

def create_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Conv部分
    x = layers.ConvLSTM2D(32, (2,2), padding='same', 
                         kernel_regularizer=regularizers.l2(0.01),
                         return_sequences=True
                         )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    #x = layers.Dropout(0.1)(x)
    x = layers.MaxPooling3D((1,1,2))(x)
    # Conv部分
    x = layers.ConvLSTM2D(64, (4,4), padding='same', 
                         kernel_regularizer=regularizers.l2(0.01),

                         )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    #x = layers.Dropout(0.1)(x)
    x = layers.MaxPooling2D((1,2))(x)

    x = layers.Flatten()(x)
    
    # Dense部分
    x = layers.Dense(128, 
                     kernel_regularizer=regularizers.l2(0.01)
                     )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Dense(64, 
                     kernel_regularizer=regularizers.l2(0.01)
                     )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.1)(x)
    
    output = layers.Dense(1)(x)
    
    model = models.Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss='mse'
    )
    
    return model



def evaluate_regression(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R^2': r2}

def oversample_extreme_values(X, y, threshold=0, multiplier=3, noise_scale=0.1):
    # 極値のインデックスを取得
    extreme_indices_valence = np.where(np.abs(y[:, 0]) >= threshold)[0]
    extreme_indices_arousal = np.where(np.abs(y[:, 1]) >= threshold)[0]
    extreme_indices = np.unique(np.concatenate([extreme_indices_valence, extreme_indices_arousal]))
    
    # オーバーサンプリング
    X_extreme = np.repeat(X[extreme_indices], multiplier, axis=0)
    y_extreme = np.repeat(y[extreme_indices], multiplier, axis=0)
    
    # データの振幅に基づくノイズレベルの計算
    # 各チャンネルごとの標準偏差を計算
    channel_stds = np.std(X_extreme, axis=tuple(range(1, X_extreme.ndim)))
    
    # ノイズの生成（チャンネルごとに異なる大きさ）
    noise = np.zeros_like(X_extreme)
    for i in range(len(channel_stds)):
        noise_level = channel_stds[i] * noise_scale
        channel_noise = np.random.normal(0, noise_level, X_extreme[i].shape)
        noise[i] = channel_noise
    
    # ノイズの追加
    X_extreme = X_extreme + noise
    
    # データの結合とシャッフル
    X_balanced = np.concatenate([X, X_extreme], axis=0)
    y_balanced = np.concatenate([y, y_extreme], axis=0)
    
    indices = np.random.permutation(len(X_balanced))
    X_balanced = X_balanced[indices]
    y_balanced = y_balanced[indices]
    
    # 情報の出力
    print(f"Original data shape: {X.shape}")
    print(f"Balanced data shape: {X_balanced.shape}")
    print(f"Extreme samples for Valence (|v| >= {threshold}): {len(extreme_indices_valence)}")
    print(f"Extreme samples for Arousal (|a| >= {threshold}): {len(extreme_indices_arousal)}")
    print(f"Total unique extreme samples: {len(extreme_indices)}")
    print(f"Noise scale: {noise_scale} of channel std")
    print(f"Channel std range: {channel_stds.min():.6f} to {channel_stds.max():.6f}")
    
    return X_balanced, y_balanced

def plot_training_history(history, model_type, save_dir):
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_type} Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_dir}/training_history_{model_type}.png')
    plt.close()

# メインの実行部分
if __name__ == "__main__":
    # データの準備
    person_ids = range(1,33)
    #selected_channels = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']
    selected_channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

    
    # データセット作成
    X, y = make_temporal_dataset(
        person_ids=person_ids, 
        window_sec=2, 
        step_sec=2,
        relative=False, 
        selected_channels=selected_channels,
        log_transform=True,
        plot_distributions=False,
    )

    # データ分割
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.1, random_state=77
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1, random_state=77
    )

    X_train, y_train = oversample_extreme_values(X_train, y_train, threshold=0, multiplier=5, noise_scale=0.1)

    # スケーリング
    scaler = StandardScaler()
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    scaler.fit(X_train_reshaped)
    
    X_train = scaler.transform(X_train_reshaped).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(X_val.shape[0], -1)).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)

    # 検証データの保存
    save_dir = './results/32ch_2DCNN_gauss_temp/validation_data'
    os.makedirs(save_dir, exist_ok=True)
    
    np.save(os.path.join(save_dir, f'X_val_neuron{n_neurons}_sigma{sigma}.npy'), X_val)
    np.save(os.path.join(save_dir, f'y_val_neuron{n_neurons}_sigma{sigma}.npy'), y_val)

    # コールバックの設定
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    ]

    # Valenceモデルの学習
    input_shape = X_train.shape[1:]
    valence_model = create_model(input_shape)
    valence_history = valence_model.fit(
        X_train, y_train[:, 0],
        epochs=total_epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val[:, 0]),
        callbacks=callbacks,
        verbose=1
    )

    # コールバックの設定
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    ]

    # Arousalモデルの学習
    input_shape = X_train.shape[1:]
    arousal_model = create_model(input_shape)
    arousal_history = arousal_model.fit(
        X_train, y_train[:, 1],
        epochs=total_epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val[:, 1]),
        callbacks=callbacks,
        verbose=1
    )

    # 予測と評価
    valence_predictions = valence_model.predict(X_test)
    arousal_predictions = arousal_model.predict(X_test)

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
    plt.savefig(f'./results/32ch_2DCNN_gauss_temp/residual/neuron{n_neurons}_sigma{sigma}_residual_plots.png')
    plt.close()

    # モデルの保存
    valence_model.save(f'./results/32ch_2DCNN_gauss_temp/models/neuron{n_neurons}_sigma{sigma}_valence_model.keras')
    arousal_model.save(f'./results/32ch_2DCNN_gauss_temp/models/neuron{n_neurons}_sigma{sigma}_arousal_model.keras')

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

    # 結果の保存
    results_df.to_csv(f'./results/32ch_2DCNN_gauss_temp/evaluation/neuron{n_neurons}_sigma{sigma}_results.csv', index=False)

    # 結果の出力
    print("\n=== Final Results ===")
    print("\nValence Results:")
    for metric, value in valence_results.items():
        print(f"{metric}: {value:.4f}")

    print("\nArousal Results:")
    for metric, value in arousal_results.items():
        print(f"{metric}: {value:.4f}")

    print("\nTraining and evaluation completed.")

    plot_training_history(valence_history, 'Valence', './results/32ch_2DCNN_gauss_temp')
    plot_training_history(arousal_history, 'Arousal', './results/32ch_2DCNN_gauss_temp')
    # 検証データの保存
    np.save('./data/X_val.npy', X_val)
    np.save('./data/y_val.npy', y_val)