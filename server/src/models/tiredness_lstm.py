"""
Tiredness model: an LSTM over the sequence of per-frame geometric features.

Following Ghoddoosian, Galib & Athitsos (2019), sequential blink / eye-closure
features carry the drowsiness signal and a small recurrent network captures their
temporal structure cheaply. Deliberately small so it runs in the browser.

Input:  (window_len, n_frame_features)   e.g. (50, 6) for 5 s at 10 fps
Output: single sigmoid in [0,1] = tiredness probability
"""

from __future__ import annotations


def build_tiredness_model(window_len: int, n_features: int,
                          lstm_units: int = 32, dropout: float = 0.3):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Input(shape=(window_len, n_features)),
        layers.Masking(mask_value=0.0),          # padded frames are all-zero
        layers.LSTM(lstm_units, return_sequences=False),
        layers.Dropout(dropout),
        layers.Dense(16, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(1, activation="sigmoid", name="tiredness"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    return model
