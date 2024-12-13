import numpy as np
from sklearn.preprocessing import StandardScaler
from dataset_segment import make_segment_dataset
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
from datetime import datetime

def create_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(filters=50, kernel_size=(2,2), activation="tanh", padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="tanh")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="tanh")(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1)(x)
    
    model = models.Model(inputs=inputs, outputs=output)
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss='mse'
    )
    return model

def evaluate_and_plot(model, X_test, y_true, emotion_type, save_dir):
    """モデルの評価と結果の可視化"""
    # 予測
    y_pred = model.predict(X_test).flatten()
    
    # 評価指標の計算
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # 結果の保存
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    print(f"\n{emotion_type} Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Residual Plot
    plt.figure(figsize=(10, 6))
    residuals = y_true - y_pred
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'{emotion_type} Residual Plot')
    plt.savefig(os.path.join(save_dir, f'{emotion_type.lower()}_residual_plot.png'))
    plt.close()
    
    # Actual vs Predicted Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_true, y=y_pred)
    plt.plot([-1, 1], [-1, 1], 'r--')  # 理想的な予測線
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{emotion_type} Actual vs Predicted')
    plt.savefig(os.path.join(save_dir, f'{emotion_type.lower()}_actual_vs_predicted.png'))
    plt.close()
    
    # Distribution Plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=y_true, label='Actual', color='blue')
    sns.kdeplot(data=y_pred, label='Predicted', color='red')
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.title(f'{emotion_type} Distribution Comparison')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'{emotion_type.lower()}_distribution.png'))
    plt.close()
    
    return metrics

if __name__ == "__main__":
    # 結果保存用のディレクトリ作成
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f'./results/evaluation_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)

    # データの準備
    train_person_ids = range(1, 29)
    test_person_ids = range(29, 33)
    selected_channels = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 
                        'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 
                        'P4', 'P8', 'PO4', 'O2']
    
    # データセット作成
    X_train, y_train, X_test, y_test = make_segment_dataset(
        train_person_ids=train_person_ids,
        test_person_ids=test_person_ids,
        model_type='2DCNN',
        window_sec=10,
        overlap_sec=5,
        relative=False,
        selected_channels=selected_channels,
        log_transform=True,
        plot_distributions=True
    )

    # スケーリング
    scaler = StandardScaler()
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
    
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_test_scaled = scaler.transform(X_test_reshaped)
    
    X_train = X_train_scaled.reshape(X_train.shape)
    X_test = X_test_scaled.reshape(X_test.shape)

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
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=callbacks
    )

    valence_metrics = evaluate_and_plot(
        valence_model, X_test, y_test[:, 0], 
        'Valence', save_dir
    )

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=30, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor='val_loss', patience=80, restore_best_weights=True, verbose=1)
    ]
    
    # Arousalモデルの学習
    arousal_history = arousal_model.fit(
        X_train, y_train[:, 1],
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=callbacks
    )

    arousal_metrics = evaluate_and_plot(
        arousal_model, X_test, y_test[:, 1], 
        'Arousal', save_dir
    )

    # 学習曲線のプロット
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(valence_history.history['loss'], label='train')
    plt.plot(valence_history.history['val_loss'], label='validation')
    plt.title('Valence Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(arousal_history.history['loss'], label='train')
    plt.plot(arousal_history.history['val_loss'], label='validation')
    plt.title('Arousal Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_curves.png'))
    plt.close()
    