"""
Boredom model: MobileNetV2 image backbone fused with window-level geometric
features.

Boredom's signal is weaker and more person-dependent than drowsiness
(Santoni, Basaruddin & Junus, 2023), so a purely geometric approach is less
reliable. A compact MobileNetV2 backbone (Sandler et al., 2018), pretrained and
fine-tuned on DAiSEE, provides a learned facial representation, concatenated with
the window-level geometric/temporal features before the classification head.

MobileNetV2 is chosen over heavier backbones (e.g. EfficientNet) because the model
must ultimately run client-side.

Two inputs:
  image:    (H, W, 3) face crop, preprocessed for MobileNetV2
  features: (n_window_features,) window-level geometric vector
Output: single sigmoid = boredom probability
"""

from __future__ import annotations


def build_boredom_model(image_size: int = 160,
                        n_window_features: int = 10,
                        dropout: float = 0.3,
                        train_backbone: bool = False):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    # --- image branch: MobileNetV2 backbone ---
    img_in = layers.Input(shape=(image_size, image_size, 3), name="image")
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False, weights="imagenet", pooling="avg",
    )
    backbone.trainable = train_backbone      # freeze for transfer learning first
    x_img = backbone(img_in, training=False)
    x_img = layers.Dropout(dropout)(x_img)
    x_img = layers.Dense(64, activation="relu")(x_img)

    # --- feature branch: window-level geometric features ---
    feat_in = layers.Input(shape=(n_window_features,), name="features")
    x_feat = layers.Dense(32, activation="relu")(feat_in)
    x_feat = layers.Dropout(dropout)(x_feat)

    # --- fuse ---
    x = layers.Concatenate()([x_img, x_feat])
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid", name="boredom")(x)

    model = models.Model(inputs=[img_in, feat_in], outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    return model


def preprocess_face(face_bgr, image_size: int = 160):
    """Crop-agnostic preprocessing: resize + MobileNetV2 preprocess. Expects a
    BGR face image; converts to RGB and scales to [-1, 1]."""
    import cv2
    import tensorflow as tf
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (image_size, image_size))
    return tf.keras.applications.mobilenet_v2.preprocess_input(rgb.astype("float32"))
