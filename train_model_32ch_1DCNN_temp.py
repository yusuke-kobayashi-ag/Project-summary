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
from dataset import make_temporal_dataset_1d, make_temporal_dataset_2d
import numpy as np                      
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

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
os.makedirs('./results/32ch_1DCNN_gauss_temp/distributions', exist_ok=True)
os.makedirs('./results/32ch_1DCNN_gauss_temp/models', exist_ok=True)
os.makedirs('./results/32ch_1DCNN_gauss_temp/evaluation', exist_ok=True)
os.makedirs('./results/32ch_1DCNN_gauss_temp/residual', exist_ok=True)
os.makedirs('./results/32ch_1DCNN_gauss_temp/training_history', exist_ok=True)

def create_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Conv部分
    x = layers.ConvLSTM1D(32, 3, padding='same', 
                         kernel_regularizer=regularizers.l2(0.01),
                         return_sequences=True
                         )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    #x = layers.Dropout(0.1)(x)
    x = layers.MaxPooling2D((1,2))(x)
    # Conv部分
    x = layers.ConvLSTM1D(64, 4, padding='same', 
                         kernel_regularizer=regularizers.l2(0.01),

                         )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    #x = layers.Dropout(0.1)(x)
    x = layers.MaxPooling1D(2)(x)

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



def oversample_extreme_values_(X, y, threshold=0, multiplier=3, noise_scale=0.1):
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

def oversample_extreme_values_2d(X, y, epochs=100, batch_size=32, latent_dim=100):
    """
    GANを使用したEEGデータのオーバーサンプリング関数
    
    Args:
        X: 入力データ
        y: ラベル
        epochs: 最大エポック数
        batch_size: バッチサイズ
        latent_dim: 潜在空間の次元
    """
    X = X.astype('float32')
    y = y.astype('float32')
    timesteps = X.shape[1]

    # Generator
    gen_input = layers.Input(shape=(latent_dim + 2,))
    x = layers.Dense(timesteps * 5 * 14 * 32)(gen_input)
    x = layers.Reshape((timesteps, 5, 14, 32))(x)
    x = layers.Conv3D(64, (3, 3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv3D(32, (3, 3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    gen_output = layers.Conv3D(1, (3, 3, 3), padding='same', activation='tanh')(x)
    generator = models.Model(gen_input, gen_output)

    # Discriminator
    disc_input = layers.Input(shape=(timesteps, 5, 14, 1))
    x = layers.Conv3D(32, (3, 3, 3), padding='same')(disc_input)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv3D(64, (3, 3, 3), padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv3D(1, (3, 3, 3), padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64)(x)
    x = layers.LeakyReLU(0.2)(x)
    disc_output = layers.Dense(1, activation='sigmoid')(x)
    discriminator = models.Model(disc_input, disc_output)

    # オプティマイザ
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    
    # 損失関数
    cross_entropy = tf.keras.losses.BinaryCrossentropy()

    @tf.function
    def train_step(real_samples, real_labels):
        noise = tf.random.normal([batch_size, latent_dim], dtype=tf.float32)
        gen_input = tf.concat([noise, real_labels], axis=1)
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_samples = generator(gen_input, training=True)
            
            real_output = discriminator(real_samples, training=True)
            fake_output = discriminator(generated_samples, training=True)
            
            gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
            real_loss = cross_entropy(tf.ones_like(real_output), real_output)
            fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
            disc_loss = real_loss + fake_loss
            
        gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        
        g_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
        d_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))
        
        return gen_loss, disc_loss

        # 学習状態の監視用変数
    best_combined_loss = float('inf')
    patience_counter = 0
    best_generator_weights = None
    history = {'d_loss': [], 'g_loss': []}

    # 学習ループ
    for epoch in range(epochs):
        idx = np.random.randint(0, X.shape[0], batch_size)
        real_samples = X[idx]
        real_labels = y[idx]
        
        g_loss, d_loss = train_step(real_samples, real_labels)
        
        history['d_loss'].append(float(d_loss))
        history['g_loss'].append(float(g_loss))
        
        # Early Stopping判定
        current_combined_loss = float(g_loss + d_loss)
        
        if current_combined_loss < best_combined_loss - 0.01:
            best_combined_loss = current_combined_loss
            patience_counter = 0
            best_generator_weights = generator.get_weights()
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")
        
        # Early Stopping条件をチェック
        if patience_counter >= 50:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            print(f"Best combined loss: {best_combined_loss:.4f}")
            generator.set_weights(best_generator_weights)
            break
        
        # Mode collapseの検出
        if float(d_loss) < 0.1 or float(g_loss) > 2.0:
            print(f"\nPossible mode collapse detected at epoch {epoch}")
            print(f"Reverting to best weights")
            generator.set_weights(best_generator_weights)
            break

    # 学習の要約を表示
    print("\nTraining Summary:")
    print(f"Total epochs: {epoch + 1}")
    print(f"Final D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")
    print(f"Best combined loss: {best_combined_loss:.4f}")

    # 最良の重みがある場合は使用
    if best_generator_weights is not None:
        generator.set_weights(best_generator_weights)

    # データ生成
    n_generate = int(3*len(X))
    noise = np.random.normal(0, 1, (n_generate, latent_dim))
    n_repeats = int(np.ceil(n_generate / len(y)))
    y_repeated = np.tile(y, (n_repeats, 1))[:n_generate]
    
    gen_input = np.concatenate([noise, y_repeated], axis=1)
    generated_samples = generator.predict(gen_input)
    
    # データの結合とシャッフル
    X_balanced = np.concatenate([X, generated_samples], axis=0)
    y_balanced = np.concatenate([y, y_repeated], axis=0)
    
    indices = np.random.permutation(len(X_balanced))
    X_balanced = X_balanced[indices]
    y_balanced = y_balanced[indices]
    
    print(f"Original data shape: {X.shape}")
    print(f"Balanced data shape: {X_balanced.shape}")
    print(f"Generated samples: {n_generate}")
    
    return X_balanced, y_balanced

def oversample_extreme_values_1D(X, y, epochs=100, batch_size=32, latent_dim=100):
    """
    1D形式のデータに対応したGANによるオーバーサンプリング関数
    
    Args:
        X: 入力データ (n_samples, n_timesteps, n_features, 1)
        y: ラベル
        epochs: 最大エポック数
        batch_size: バッチサイズ
        latent_dim: 潜在空間の次元
    """
    X = X.astype('float32')
    y = y.astype('float32')
    timesteps = X.shape[1]
    n_features = X.shape[2]

    # Generator
    gen_input = layers.Input(shape=(latent_dim + 2,))
    x = layers.Dense(timesteps * n_features * 4)(gen_input)
    x = layers.Reshape((timesteps, n_features, 4))(x)
    
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Conv2D(32, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    gen_output = layers.Conv2D(1, (3, 3), padding='same', activation='tanh')(x)
    generator = models.Model(gen_input, gen_output)

    # Discriminator
    disc_input = layers.Input(shape=(timesteps, n_features, 1))
    
    x = layers.Conv2D(32, (3, 3), padding='same')(disc_input)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(32, (3, 3), padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64)(x)
    x = layers.LeakyReLU(0.2)(x)
    disc_output = layers.Dense(1, activation='sigmoid')(x)
    discriminator = models.Model(disc_input, disc_output)

    # オプティマイザ
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    
    # 損失関数
    cross_entropy = tf.keras.losses.BinaryCrossentropy()

    @tf.function
    def train_step(real_samples, real_labels):
        noise = tf.random.normal([batch_size, latent_dim], dtype=tf.float32)
        gen_input = tf.concat([noise, real_labels], axis=1)
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_samples = generator(gen_input, training=True)
            
            real_output = discriminator(real_samples, training=True)
            fake_output = discriminator(generated_samples, training=True)
            
            gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
            real_loss = cross_entropy(tf.ones_like(real_output), real_output)
            fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
            disc_loss = real_loss + fake_loss
            
        gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        
        g_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
        d_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))
        
        return gen_loss, disc_loss

    # 学習状態の監視用変数
    best_combined_loss = float('inf')
    patience_counter = 0
    best_generator_weights = None
    history = {'d_loss': [], 'g_loss': []}

    # 学習ループ
    for epoch in range(epochs):
        idx = np.random.randint(0, X.shape[0], batch_size)
        real_samples = X[idx]
        real_labels = y[idx]
        
        g_loss, d_loss = train_step(real_samples, real_labels)
        
        history['d_loss'].append(float(d_loss))
        history['g_loss'].append(float(g_loss))
        
        # Early Stopping判定
        current_combined_loss = float(g_loss + d_loss)
        
        if current_combined_loss < best_combined_loss - 0.01:
            best_combined_loss = current_combined_loss
            patience_counter = 0
            best_generator_weights = generator.get_weights()
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")
        
        # Early Stopping条件をチェック
        if patience_counter >= 50:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            print(f"Best combined loss: {best_combined_loss:.4f}")
            generator.set_weights(best_generator_weights)
            break
        
        # Mode collapseの検出
        if float(d_loss) < 0.1 or float(g_loss) > 2.0:
            print(f"\nPossible mode collapse detected at epoch {epoch}")
            print(f"Reverting to best weights")
            generator.set_weights(best_generator_weights)
            break

    # 学習の要約を表示
    print("\nTraining Summary:")
    print(f"Total epochs: {epoch + 1}")
    print(f"Final D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")
    print(f"Best combined loss: {best_combined_loss:.4f}")

    # 最良の重みがある場合は使用
    if best_generator_weights is not None:
        generator.set_weights(best_generator_weights)

    # データ生成
    n_generate = int(3*len(X))
    noise = np.random.normal(0, 1, (n_generate, latent_dim))
    n_repeats = int(np.ceil(n_generate / len(y)))
    y_repeated = np.tile(y, (n_repeats, 1))[:n_generate]
    
    gen_input = np.concatenate([noise, y_repeated], axis=1)
    generated_samples = generator.predict(gen_input)
    
    # データの結合とシャッフル
    X_balanced = np.concatenate([X, generated_samples], axis=0)
    y_balanced = np.concatenate([y, y_repeated], axis=0)
    
    indices = np.random.permutation(len(X_balanced))
    X_balanced = X_balanced[indices]
    y_balanced = y_balanced[indices]
    
    print(f"Original data shape: {X.shape}")
    print(f"Balanced data shape: {X_balanced.shape}")
    print(f"Generated samples: {n_generate}")
    
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
    X, y = make_temporal_dataset_1d(
        person_ids=person_ids, 
        window_sec=5, 
        step_sec=5,
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

    #X_train, y_train = oversample_extreme_values(X_train, y_train, threshold=0, multiplier=5, noise_scale=1)
    X_train, y_train = oversample_extreme_values_1D(X_train, y_train)

    # スケーリング
    scaler = StandardScaler()
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    scaler.fit(X_train_reshaped)
    
    X_train = scaler.transform(X_train_reshaped).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(X_val.shape[0], -1)).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)

    # 検証データの保存
    save_dir = './results/32ch_1DCNN_gauss_temp/validation_data'
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
    plt.savefig(f'./results/32ch_1DCNN_gauss_temp/residual/neuron{n_neurons}_sigma{sigma}_residual_plots.png')
    plt.close()

    # モデルの保存
    valence_model.save(f'./results/32ch_1DCNN_gauss_temp/models/neuron{n_neurons}_sigma{sigma}_valence_model.keras')
    arousal_model.save(f'./results/32ch_1DCNN_gauss_temp/models/neuron{n_neurons}_sigma{sigma}_arousal_model.keras')

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
    results_df.to_csv(f'./results/32ch_1DCNN_gauss_temp/evaluation/neuron{n_neurons}_sigma{sigma}_results.csv', index=False)

    # 結果の出力
    print("\n=== Final Results ===")
    print("\nValence Results:")
    for metric, value in valence_results.items():
        print(f"{metric}: {value:.4f}")

    print("\nArousal Results:")
    for metric, value in arousal_results.items():
        print(f"{metric}: {value:.4f}")

    print("\nTraining and evaluation completed.")

    plot_training_history(valence_history, 'Valence', './results/32ch_1DCNN_gauss_temp')
    plot_training_history(arousal_history, 'Arousal', './results/32ch_1DCNN_gauss_temp')
    # 検証データの保存
    np.save('./data/X_val.npy', X_val)
    np.save('./data/y_val.npy', y_val)