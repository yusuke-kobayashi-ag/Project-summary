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
from dataset import make_data_set

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # または '2'

# Metal GPUの設定
try:
    tf.config.experimental.set_visible_devices([], 'GPU')
    print('Using Apple Silicon (M2) with Metal backend')
except:
    print('Using CPU only')


total_epochs = 1000
batch_size = 64

# ディレクトリの作成
os.makedirs('./results/14ch_1DCNN/distributions', exist_ok=True)
os.makedirs('./results/14ch_1DCNN/models', exist_ok=True)
os.makedirs('./results/14ch_1DCNN/evaluation', exist_ok=True)
os.makedirs('./results/14ch_1DCNN/residual', exist_ok=True)
os.makedirs('./results/14ch_1DCNN/training_history', exist_ok=True)

def create_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv1D(filters=32, kernel_size=3, activation="tanh", padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    #x = layers.Dropout(0.1)(x)
    
    x = layers.Conv1D(filters=64, kernel_size=3, activation="tanh", padding='same')(x)
    x = layers.BatchNormalization()(x)
    #x = layers.Dropout(0.1)(x)

    x = layers.Conv1D(filters=128, kernel_size=3, activation="tanh", padding='same')(x)
    x = layers.BatchNormalization()(x)
    #x = layers.Dropout(0.1)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="tanh")(x)
    #x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation="tanh")(x)
    #x = layers.Dropout(0.1)(x)
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

def oversample_extreme_values(X, y, threshold=0.7, multiplier=3, noise_level=0.01):
    extreme_indices_valence = np.where(np.abs(y[:, 0]) >= threshold)[0]
    extreme_indices_arousal = np.where(np.abs(y[:, 1]) >= threshold)[0]
    extreme_indices = np.unique(np.concatenate([extreme_indices_valence, extreme_indices_arousal]))
    
    X_extreme = np.repeat(X[extreme_indices], multiplier, axis=0)
    y_extreme = np.repeat(y[extreme_indices], multiplier, axis=0)
    
    noise = np.random.normal(0, noise_level, X_extreme.shape)
    X_extreme = X_extreme + noise
    
    X_balanced = np.concatenate([X, X_extreme], axis=0)
    y_balanced = np.concatenate([y, y_extreme], axis=0)
    
    indices = np.random.permutation(len(X_balanced))
    X_balanced = X_balanced[indices]
    y_balanced = y_balanced[indices]
    
    print(f"Original data shape: {X.shape}")
    print(f"Balanced data shape: {X_balanced.shape}")
    print(f"Extreme samples for Valence (|v| >= {threshold}): {len(extreme_indices_valence)}")
    print(f"Extreme samples for Arousal (|a| >= {threshold}): {len(extreme_indices_arousal)}")
    print(f"Total unique extreme samples: {len(extreme_indices)}")
    print(f"Added noise level: {noise_level}")
    
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
    selected_channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    #selected_channels = ['Fp1','Fp2']
    X, y = make_data_set(
        person_ids=person_ids, 
        model_type='1DCNN',
        window_sec=10, 
        relative=False, 
        selected_channels=selected_channels,
        log_transform=True,
        use_segments=False,  # セグメント化を有効化
        overlap_sec=5,
        plot_distributions=True
    )

    print("Data shape:", X.shape)
    print("Label shape:", y.shape)
    print("\nデータの分布確認:")
    print("Valenceの範囲:", np.min(y[:, 0]), "から", np.max(y[:, 0]))
    print("Arousalの範囲:", np.min(y[:, 1]), "から", np.max(y[:, 1]))
    print("Xの最小値:", np.min(X), "Xの最大値", np.max(X), "Xの平均", np.mean(X))

    # データの分割
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=7
    )

    # 訓練データのみでスケーラーを学習
    scaler = StandardScaler()
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    scaler.fit(X_train_reshaped)

    # 訓練データと検証データそれぞれを変換
    X_train_scaled = scaler.transform(X_train_reshaped)
    X_val_scaled = scaler.transform(X_val.reshape(X_val.shape[0], -1))

    # 元の形状に戻す
    X_train = X_train_scaled.reshape(X_train.shape)
    X_val = X_val_scaled.reshape(X_val.shape)
    
    # データ分布の可視化（検証データを使用）
    plt.figure(figsize=(8, 8))
    plt.scatter(y_val[:, 0], y_val[:, 1], alpha=0.5)
    plt.xlabel('Valence')
    plt.ylabel('Arousal')
    plt.title('Validation Data Distribution')
    plt.grid(True)
    plt.axis([-1, 1, -1, 1])
    plt.savefig('./results/14ch_1DCNN/distributions/validation_data_distribution.png')
    plt.close()

    # モデルの作成
    input_shape = X_train.shape[1:]
    valence_model = create_model(input_shape)
    arousal_model = create_model(input_shape)

    # コールバックの設定
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1)
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

    # 予測（検証データを使用）
    valence_predictions = valence_model.predict(X_val)
    arousal_predictions = arousal_model.predict(X_val)

    # reshape predictions if needed
    valence_predictions = valence_predictions.reshape(-1)
    arousal_predictions = arousal_predictions.reshape(-1)

    # 評価結果の計算（検証データを使用）
    valence_results = evaluate_regression(y_val[:, 0], valence_predictions)
    arousal_results = evaluate_regression(y_val[:, 1], arousal_predictions)

    # 残差プロット（検証データを使用）
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_val[:, 0], valence_predictions, alpha=0.5)
    plt.plot([-1, 1], [-1, 1], 'r--')
    plt.xlabel('True Valence')
    plt.ylabel('Predicted Valence')
    plt.title('Valence: True vs Predicted (Validation)')
    plt.grid(True, alpha=0.3)
    plt.axis([-1, 1, -1, 1])

    plt.subplot(1, 2, 2)
    plt.scatter(y_val[:, 1], arousal_predictions, alpha=0.5)
    plt.plot([-1, 1], [-1, 1], 'r--')
    plt.xlabel('True Arousal')
    plt.ylabel('Predicted Arousal')
    plt.title('Arousal: True vs Predicted (Validation)')
    plt.grid(True, alpha=0.3)
    plt.axis([-1, 1, -1, 1])

    plt.tight_layout()
    plt.savefig('./results/14ch_1DCNN/residual/validation_residual_plots.png')
    plt.close()

    # モデルの保存
    valence_model.save('./results/14ch_1DCNN/models/valence_model.keras')
    arousal_model.save('./results/14ch_1DCNN/models/arousal_model.keras')

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

    plot_training_history(valence_history, 'Valence', './results/14ch_1DCNN/training_history')
    plot_training_history(arousal_history, 'Arousal', './results/14ch_1DCNN/training_history')

    # 結果の保存
    results_df.to_csv('./results/14ch_1DCNN/evaluation/results.csv', index=False)

    # 結果の出力
    print("\n=== Final Results ===")
    print("\nValence Results:")
    for metric, value in valence_results.items():
        print(f"{metric}: {value:.4f}")

    print("\nArousal Results:")
    for metric, value in arousal_results.items():
        print(f"{metric}: {value:.4f}")

    print("\nTraining and evaluation completed.")