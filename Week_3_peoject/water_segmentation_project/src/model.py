import logging
from pathlib import Path
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.layers import (
    Activation,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Input,
    Layer,
    MaxPool2D,
    Dropout,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UNetMultiChannel:
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (128, 128, 12),
        num_classes: int = 1,
        base_filters: int = 64,
        kernel_initializer: str = "he_normal",
        padding: str = "valid",
        dropout: float = 0.5,
        batch_norm: bool = True,
    ) -> None:
        self.input_shape: Tuple[int, int, int] = input_shape
        self.num_classes: int = num_classes
        self.base_filters: int = base_filters
        self.kernel_initializer: str = kernel_initializer
        self.padding: str = padding
        self.dropout: float = dropout
        self.batch_norm: bool = batch_norm
        self.model: Model = self.build_model()

    def __str__(self) -> str:
        details = [
            f"UNetMultiChannel(",
            f"  input_shape={self.input_shape},",
            f"  num_classes={self.num_classes},",
            f"  base_filters={self.base_filters},",
            f"  kernel_initializer='{self.kernel_initializer}',",
            f"  padding='{self.padding}'",
            f")",
            f"Model name: {self.model.name}",
            f"Total params: {self.model.count_params():,}",
            f"Input shape: {self.model.input_shape}",
            f"Output shape: {self.model.output_shape}",
        ]
        return "\n".join(details)

    def conv_block(
        self,
        x: Layer,
        filters: int,
        kernel_size: int = 3,
        padding: Optional[str] = None,
        activation: str = "relu",
        kernel_initializer: Optional[str] = None,
        batch_norm: bool = True,
    ) -> Layer:

        kernel_initializer = kernel_initializer or self.kernel_initializer
        padding = padding if padding is not None else self.padding
        x = Conv2D(
            filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer
        )(x)
        x = Activation(activation)(x)
        if batch_norm:
            x = layers.BatchNormalization()(x)

        x = Conv2D(
            filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer
        )(x)
        x = Activation(activation)(x)
        if batch_norm:
            x = layers.BatchNormalization()(x)
        return x

    def encoder_block(
        self,
        x: Layer,
        filters: int,
        kernel_initializer: Optional[str] = None,
        padding: Optional[str] = None,
    ) -> Tuple[Layer, Layer]:

        kernel_initializer = kernel_initializer or self.kernel_initializer
        padding = padding if padding is not None else self.padding
        c = self.conv_block(
            x, filters, kernel_initializer=kernel_initializer, padding=padding
        )
        p = MaxPool2D((2, 2))(c)
        return c, p

    def decoder_block(
        self,
        x: Layer,
        skip: Layer,
        filters: int,
        kernel_initializer: Optional[str] = None,
        padding: Optional[str] = None,
    ) -> Layer:
        kernel_initializer = kernel_initializer or self.kernel_initializer
        padding = padding if padding is not None else self.padding
        us = Conv2DTranspose(
            filters,
            (2, 2),
            strides=2,
            padding=padding,
            kernel_initializer=kernel_initializer,
        )(x)
        concat = Concatenate()([us, skip])
        return self.conv_block(
            concat, filters, kernel_initializer=kernel_initializer, padding=padding
        )

    def build_model(self) -> Model:
        logger.info("Building UNet model...")
        inputs = Input(shape=self.input_shape)

        # Encoder
        c1, p1 = self.encoder_block(inputs, self.base_filters)
        c2, p2 = self.encoder_block(p1, self.base_filters * 2)
        c3, p3 = self.encoder_block(p2, self.base_filters * 4)
        c4, p4 = self.encoder_block(p3, self.base_filters * 8)

        # Bottleneck
        b = self.conv_block(p4, self.base_filters * 16)
        if self.dropout:
            b = layers.Dropout(self.dropout)(b)
        if self.batch_norm:
            b = layers.BatchNormalization()(b)

        # Decoder
        d1 = self.decoder_block(b, c4, self.base_filters * 8)
        d2 = self.decoder_block(d1, c3, self.base_filters * 4)
        d3 = self.decoder_block(d2, c2, self.base_filters * 2)
        d4 = self.decoder_block(d3, c1, self.base_filters)

        # Output
        outputs = Conv2D(
            self.num_classes,
            (1, 1),
            activation="sigmoid",
            kernel_initializer=self.kernel_initializer,
            padding=self.padding,
        )(d4)

        model = Model(inputs, outputs, name="UNet_MultiChannel")
        logger.info("Model built successfully.")
        return model

    def summary(self) -> None:
        logger.info("Model summary:")
        self.model.summary()

    def get_model(self) -> Model:
        return self.model


if __name__ == "__main__":
    logger.info("Starting UNetMultiChannel test run...")
    unet = UNetMultiChannel(input_shape=(128, 128, 12), padding="same")
    print("=" * 100)
    print(unet)
    print("=" * 100)
    unet.summary()
