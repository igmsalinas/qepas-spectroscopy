"""Deep-learning architecture builders for QEPAS signals."""

from __future__ import annotations

from typing import Callable

import tensorflow as tf
from keras import layers, models, regularizers


def _norm_layer(norm_name: str, name: str | None = None):
    if norm_name == "batch":
        return layers.BatchNormalization(name=name)
    elif norm_name == "layer":
        return layers.LayerNormalization(name=name)
    elif norm_name == "group":
        return layers.GroupNormalization(groups=2, name=name)
    elif norm_name is None or norm_name == "none":
        return lambda x: x
    else:
        raise ValueError(f"Unknown normalization: {norm_name}")


def _activation_layer(name: str = "relu") -> Callable:
    if name == "relu":
        return layers.ReLU()
    elif name == "gelu":
        return layers.Activation("gelu")
    elif name == "swish":
        return layers.Activation("swish")
    else:
        raise ValueError(f"Unknown activation: {name}")


def build_simple_cnn(
    input_shape: tuple[int, int],
    scalar_dim: int,
    conv_blocks: int = 2,
    initial_filters: int = 32,
    kernel_size: int = 7,
    strides: int = 2,
    scalar_units: int = 16,
    dense_blocks: int = 2,
    dense_units: int = 64,
    dropout: float = 0.3,
    l2: float = 1e-4,
    normalization: str = "batch",
    activation: str = "relu",
    num_outputs: int = 2,
) -> models.Model:
    """Simple stacked 1-D CNN with global pooling."""
    signals_input = layers.Input(shape=input_shape, name="signals")
    x = signals_input
    for i in range(conv_blocks):
        filters = initial_filters * (2 ** i)
        x = layers.Conv1D(
            filters,
            kernel_size,
            strides=strides,
            padding="same",
            kernel_regularizer=regularizers.l2(l2),
            name=f"conv_{i}",
        )(x)
        x = _norm_layer(normalization, name=f"norm_{i}")(x)
        x = _activation_layer(activation)(x)

    x = layers.GlobalAveragePooling1D(name="gap")(x)

    scalar_input = layers.Input(shape=(scalar_dim,), name="scalars")
    s = layers.Dense(scalar_units, activation=activation, name="scalar_dense")(scalar_input)
    s = layers.Dropout(dropout)(s)

    c = layers.concatenate([x, s], name="concat")
    for i in range(dense_blocks):
        c = layers.Dense(
            dense_units // (2 ** i) if i > 0 else dense_units,
            activation=activation,
            kernel_regularizer=regularizers.l2(l2),
            name=f"dense_{i}",
        )(c)
        c = layers.Dropout(dropout, name=f"drop_{i}")(c)

    output = layers.Dense(num_outputs, activation="linear", name="concentrations")(c)
    return models.Model(inputs=[signals_input, scalar_input], outputs=output, name="SimpleCNN")


def build_resnet1d_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int,
    strides: int,
    l2: float,
    normalization: str,
    activation: str,
    name: str,
) -> tf.Tensor:
    """Residual block with optional downsampling."""
    shortcut = x
    if strides > 1 or x.shape[-1] != filters:
        shortcut = layers.Conv1D(
            filters,
            1,
            strides=strides,
            padding="same",
            kernel_regularizer=regularizers.l2(l2),
            name=f"{name}_shortcut",
        )(shortcut)
        shortcut = _norm_layer(normalization, name=f"{name}_shortcut_norm")(shortcut)

    x = layers.Conv1D(
        filters,
        kernel_size,
        strides=strides,
        padding="same",
        kernel_regularizer=regularizers.l2(l2),
        name=f"{name}_conv1",
    )(x)
    x = _norm_layer(normalization, name=f"{name}_norm1")(x)
    x = _activation_layer(activation)(x)

    x = layers.Conv1D(
        filters,
        kernel_size,
        padding="same",
        kernel_regularizer=regularizers.l2(l2),
        name=f"{name}_conv2",
    )(x)
    x = _norm_layer(normalization, name=f"{name}_norm2")(x)

    x = layers.Add(name=f"{name}_add")([x, shortcut])
    x = _activation_layer(activation)(x)
    return x


def build_resnet1d(
    input_shape: tuple[int, int],
    scalar_dim: int,
    blocks: int = 3,
    base_filters: int = 32,
    kernel_size: int = 7,
    scalar_units: int = 16,
    dense_units: int = 64,
    dropout: float = 0.3,
    l2: float = 1e-4,
    normalization: str = "batch",
    activation: str = "relu",
    num_outputs: int = 2,
) -> models.Model:
    """1-D ResNet with residual blocks."""
    signals_input = layers.Input(shape=input_shape, name="signals")
    x = layers.Conv1D(
        base_filters,
        kernel_size,
        padding="same",
        kernel_regularizer=regularizers.l2(l2),
        name="stem",
    )(signals_input)
    x = _norm_layer(normalization, name="stem_norm")(x)
    x = _activation_layer(activation)(x)

    for i in range(blocks):
        strides = 2 if i > 0 else 1
        filters = base_filters * (2 ** i)
        x = build_resnet1d_block(
            x,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            l2=l2,
            normalization=normalization,
            activation=activation,
            name=f"resblock_{i}",
        )

    x = layers.GlobalAveragePooling1D(name="gap")(x)

    scalar_input = layers.Input(shape=(scalar_dim,), name="scalars")
    s = layers.Dense(scalar_units, activation=activation, name="scalar_dense")(scalar_input)
    s = layers.Dropout(dropout)(s)

    c = layers.concatenate([x, s], name="concat")
    c = layers.Dense(dense_units, activation=activation, kernel_regularizer=regularizers.l2(l2), name="dense_0")(c)
    c = layers.Dropout(dropout, name="drop_0")(c)
    c = layers.Dense(max(16, dense_units // 2), activation=activation, name="dense_1")(c)

    output = layers.Dense(num_outputs, activation="linear", name="concentrations")(c)
    return models.Model(inputs=[signals_input, scalar_input], outputs=output, name="ResNet1D")


def build_tcn(
    input_shape: tuple[int, int],
    scalar_dim: int,
    nb_filters: int = 32,
    kernel_size: int = 7,
    nb_stacks: int = 2,
    dilations: list[int] | None = None,
    scalar_units: int = 16,
    dense_units: int = 64,
    dropout: float = 0.3,
    l2: float = 1e-4,
    activation: str = "relu",
    normalization: str = "batch",
    num_outputs: int = 2,
) -> models.Model:
    """Temporal Convolutional Network with dilated causal convolutions."""
    if dilations is None:
        dilations = [1, 2, 4, 8]

    signals_input = layers.Input(shape=input_shape, name="signals")
    x = signals_input
    for s in range(nb_stacks):
        for d in dilations:
            x = layers.Conv1D(
                nb_filters,
                kernel_size,
                dilation_rate=d,
                padding="causal",
                kernel_regularizer=regularizers.l2(l2),
                name=f"tcn_stack{s}_dil{d}",
            )(x)
            x = layers.BatchNormalization(name=f"tcn_norm{s}_dil{d}")(x)
            x = _activation_layer(activation)(x)
            x = layers.SpatialDropout1D(dropout)(x)

    x = layers.GlobalAveragePooling1D(name="gap")(x)

    scalar_input = layers.Input(shape=(scalar_dim,), name="scalars")
    s = layers.Dense(scalar_units, activation=activation, name="scalar_dense")(scalar_input)
    s = layers.Dropout(dropout)(s)

    c = layers.concatenate([x, s], name="concat")
    c = layers.Dense(dense_units, activation=activation, kernel_regularizer=regularizers.l2(l2), name="dense_0")(c)
    c = layers.Dropout(dropout, name="drop_0")(c)

    output = layers.Dense(num_outputs, activation="linear", name="concentrations")(c)
    return models.Model(inputs=[signals_input, scalar_input], outputs=output, name="TCN")


def build_lstm(
    input_shape: tuple[int, int],
    scalar_dim: int,
    lstm_units: int = 64,
    lstm_layers: int = 2,
    scalar_units: int = 16,
    dense_units: int = 64,
    dropout: float = 0.3,
    l2: float = 1e-4,
    bidirectional: bool = False,
    activation: str = "relu",
    normalization: str = "batch",
    num_outputs: int = 2,
) -> models.Model:
    """LSTM/GRU-based sequence model."""
    signals_input = layers.Input(shape=input_shape, name="signals")
    x = signals_input
    for i in range(lstm_layers):
        return_sequences = i < lstm_layers - 1
        lstm = layers.LSTM(
            lstm_units // (2 ** i) if i > 0 else lstm_units,
            return_sequences=return_sequences,
            dropout=dropout,
            recurrent_dropout=dropout / 2,
            kernel_regularizer=regularizers.l2(l2),
            name=f"lstm_{i}",
        )
        x = lstm(x) if not bidirectional else layers.Bidirectional(lstm, name=f"bilstm_{i}")(x)
        x = layers.LayerNormalization()(x)

    scalar_input = layers.Input(shape=(scalar_dim,), name="scalars")
    s = layers.Dense(scalar_units, activation=activation, name="scalar_dense")(scalar_input)
    s = layers.Dropout(dropout)(s)

    c = layers.concatenate([x, s], name="concat")
    c = layers.Dense(dense_units, activation=activation, kernel_regularizer=regularizers.l2(l2), name="dense_0")(c)
    c = layers.Dropout(dropout, name="drop_0")(c)

    output = layers.Dense(num_outputs, activation="linear", name="concentrations")(c)
    return models.Model(inputs=[signals_input, scalar_input], outputs=output, name="LSTM")


def build_multiscale_cnn(
    input_shape: tuple[int, int],
    scalar_dim: int,
    filters: int = 32,
    kernel_sizes: list[int] | None = None,
    scalar_units: int = 16,
    dense_units: int = 64,
    dropout: float = 0.3,
    l2: float = 1e-4,
    normalization: str = "batch",
    activation: str = "relu",
    num_outputs: int = 2,
) -> models.Model:
    """Multi-scale CNN with parallel branches of different kernel sizes."""
    if kernel_sizes is None:
        kernel_sizes = [3, 7, 15]

    signals_input = layers.Input(shape=input_shape, name="signals")
    branches = []
    for k in kernel_sizes:
        b = layers.Conv1D(
            filters,
            k,
            padding="same",
            kernel_regularizer=regularizers.l2(l2),
            name=f"branch_k{k}",
        )(signals_input)
        b = _norm_layer(normalization, name=f"branch_norm_k{k}")(b)
        b = _activation_layer(activation)(b)
        b = layers.GlobalMaxPooling1D(name=f"gmp_k{k}")(b)
        branches.append(b)

    x = layers.concatenate(branches, name="branches_concat")
    x = layers.Dense(filters * len(kernel_sizes), activation=activation, name="branch_fusion")(x)
    x = layers.Dropout(dropout)(x)

    scalar_input = layers.Input(shape=(scalar_dim,), name="scalars")
    s = layers.Dense(scalar_units, activation=activation, name="scalar_dense")(scalar_input)
    s = layers.Dropout(dropout)(s)

    c = layers.concatenate([x, s], name="concat")
    c = layers.Dense(dense_units, activation=activation, kernel_regularizer=regularizers.l2(l2), name="dense_0")(c)
    c = layers.Dropout(dropout, name="drop_0")(c)

    output = layers.Dense(num_outputs, activation="linear", name="concentrations")(c)
    return models.Model(inputs=[signals_input, scalar_input], outputs=output, name="MultiScaleCNN")


def build_transformer1d(
    input_shape: tuple[int, int],
    scalar_dim: int,
    embed_dim: int = 64,
    num_heads: int = 4,
    ff_dim: int = 128,
    num_blocks: int = 2,
    scalar_units: int = 16,
    dense_units: int = 64,
    dropout: float = 0.3,
    l2: float = 1e-4,
    num_outputs: int = 2,
) -> models.Model:
    """1-D Transformer encoder over the time dimension."""
    signals_input = layers.Input(shape=input_shape, name="signals")
    x = layers.Conv1D(embed_dim, 7, padding="same", kernel_regularizer=regularizers.l2(l2), name="embedding")(signals_input)
    x = layers.LayerNormalization(name="embed_norm")(x)

    for i in range(num_blocks):
        attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout,
            name=f"attn_{i}",
        )(x, x)
        x = layers.LayerNormalization(name=f"norm1_{i}")(x + attn)
        ff = layers.Dense(ff_dim, activation="relu", name=f"ff_{i}_0")(x)
        ff = layers.Dropout(dropout)(ff)
        ff = layers.Dense(embed_dim, name=f"ff_{i}_1")(ff)
        x = layers.LayerNormalization(name=f"norm2_{i}")(x + ff)

    x = layers.GlobalAveragePooling1D(name="gap")(x)

    scalar_input = layers.Input(shape=(scalar_dim,), name="scalars")
    s = layers.Dense(scalar_units, activation="relu", name="scalar_dense")(scalar_input)
    s = layers.Dropout(dropout)(s)

    c = layers.concatenate([x, s], name="concat")
    c = layers.Dense(dense_units, activation="relu", kernel_regularizer=regularizers.l2(l2), name="dense_0")(c)
    c = layers.Dropout(dropout, name="drop_0")(c)

    output = layers.Dense(num_outputs, activation="linear", name="concentrations")(c)
    return models.Model(inputs=[signals_input, scalar_input], outputs=output, name="Transformer1D")
