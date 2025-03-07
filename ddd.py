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


# グローバル変数の定義
n_neurons = 10
sigma = 0.25
total_epochs = 1000
batch_size = 64

# ディレクトリの作成
os.makedirs('./results/32ch_2DCNN_gauss/distributions', exist_ok=True)
os.makedirs('./results/32ch_2DCNN_gauss/models', exist_ok=True)
os.makedirs('./results/32ch_2DCNN_gauss/evaluation', exist_ok=True)
os.makedirs('./results/32ch_2DCNN_gauss/residual', exist_ok=True)
os.makedirs('./results/32ch_2DCNN_gauss/training_history', exist_ok=True)


def create_model(input_shape, sigma):
    inputs = layers.Input(shape=input_shape)
    
    # Conv部分の改善
    x = layers.Conv2D(20, (1,2), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('tanh')(x)  # tanhから変更
    x = layers.Dropout(0.1)(x)
    x = layers.Conv2D(20, (1,2), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('tanh')(x)  # tanhから変更
    x = layers.Dropout(0.1)(x)
    
    x = layers.Flatten()(x)
    
    # Dense部分の改善
    x = layers.Dense(128)(x)  # ユニット数削減
    x = layers.BatchNormalization()(x)
    x = layers.Activation('tanh')(x)
    #x = layers.Dropout(0.2)(x)  # dropout率増加
    
    x = layers.Dense(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('tanh')(x)
    #x = layers.Dropout(0.2)(x)
    output = layers.Dense(n_neurons, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=output)

    centers = tf.constant(np.linspace(-1, 1, n_neurons), dtype=tf.float32)[None, :]
    
    def loss(y_true, y_pred):
        importance_weights = 1.0 + tf.abs(y_true)
        
        # TensorFlowでの正規分布計算
        diff = (centers - y_true[:, None]) / sigma
        target_distribution = tf.exp(-0.5 * tf.square(diff)) / (sigma * tf.sqrt(2.0 * np.pi))
        
        # 正規化
        max_prob = 1.0 / (sigma * tf.sqrt(2.0 * np.pi))
        target_distribution = target_distribution / max_prob
        
        # 確率分布に変換
        target_distribution = target_distribution / tf.reduce_sum(target_distribution, axis=1, keepdims=True)
        
        # KLダイバージェンス
        sample_losses = tf.keras.losses.KLDivergence()(target_distribution, y_pred)
        
        return tf.reduce_mean(sample_losses)
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss=loss
    )
    return model

def decode_prediction(pred_dist):
    centers = np.linspace(-1, 1, n_neurons)
    if len(pred_dist.shape) == 1:
        pred_dist = pred_dist[np.newaxis, :]
    power = 2
    emphasized_dist = np.power(pred_dist, power)
    normalized_dist = emphasized_dist / np.sum(emphasized_dist, axis=1, keepdims=True)
    predictions = np.sum(normalized_dist * centers, axis=1)
    return predictions

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
    selected_channels = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']
    
    # データセット作成
    X, y = make_data_set(
        person_ids=person_ids, 
        model_type='2DCNN',
        window_sec=10, 
        relative=False, 
        selected_channels=selected_channels,
        log_transform=True,
        use_segments=False,
        overlap_sec=5,
        plot_distributions=True,
        gauss=True,
        clipping=False
    
    )

    # データ分割部分を以下に変更
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05,random_state=77)

    # スケーリング部分を簡略化
    scaler = StandardScaler()
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    scaler.fit(X_train_reshaped)

    X_train = scaler.transform(X_train_reshaped).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(X_val.shape[0], -1)).reshape(X_val.shape)


    # 検証データの保存
    save_dir = './results/32ch_2DCNN_gauss/validation_data'
    os.makedirs(save_dir, exist_ok=True)
    
    np.save(os.path.join(save_dir, f'X_val_neuron{n_neurons}_sigma{sigma}.npy'), X_val)
    np.save(os.path.join(save_dir, f'y_val_neuron{n_neurons}_sigma{sigma}.npy'), y_val)

    # コールバックの設定
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=30, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor='val_loss', patience=80, restore_best_weights=True, verbose=1)
    ]

    # Valenceモデルの学習
    input_shape = X_train.shape[1:]
    valence_model = create_model(input_shape, sigma)
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
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=30, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor='val_loss', patience=80, restore_best_weights=True, verbose=1)
    ]

    # Arousalモデルの学習
    input_shape = X_train.shape[1:]
    arousal_model = create_model(input_shape, sigma)
    arousal_history = arousal_model.fit(
        X_train, y_train[:, 1],
        epochs=total_epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val[:, 1]),
        callbacks=callbacks,
        verbose=1
    )

    # 評価部分を以下に変更
    valence_pred_dist = valence_model.predict(X_val)
    arousal_pred_dist = arousal_model.predict(X_val)

    valence_predictions = decode_prediction(valence_pred_dist)
    arousal_predictions = decode_prediction(arousal_pred_dist)

    valence_results = evaluate_regression(y_val[:, 0], valence_predictions)
    arousal_results = evaluate_regression(y_val[:, 1], arousal_predictions)

    # 残差プロット
    # 残差プロット部分を修正
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
    plt.savefig(f'./results/32ch_2DCNN_gauss/residual/neuron{n_neurons}_sigma{sigma}_validation_plots.png')
    plt.close()

    # モデルの保存
    valence_model.save(f'./results/32ch_2DCNN_gauss/models/neuron{n_neurons}_sigma{sigma}_valence_model.keras')
    arousal_model.save(f'./results/32ch_2DCNN_gauss/models/neuron{n_neurons}_sigma{sigma}_arousal_model.keras')

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
    results_df.to_csv(f'./results/32ch_2DCNN_gauss/evaluation/neuron{n_neurons}_sigma{sigma}_results.csv', index=False)

    # 結果の出力
    print("\n=== Final Results ===")
    print("\nValence Results:")
    for metric, value in valence_results.items():
        print(f"{metric}: {value:.4f}")

    print("\nArousal Results:")
    for metric, value in arousal_results.items():
        print(f"{metric}: {value:.4f}")

    print("\nTraining and evaluation completed.")
    plot_training_history(valence_history, 'Valence', './results/32ch_2DCNN_gauss')
    plot_training_history(arousal_history, 'Arousal', './results/32ch_2DCNN_gauss')
    # 検証データの保存
    np.save('./data/X_val.npy', X_val)
    np.save('./data/y_val.npy', y_val)
