import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

(ds_train, ds_test), ds_info = tfds.load(
    'fashion_mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

def display_bottleneck_features(model, dataset, num_images=5, num_channels=8):
    encoder = models.Model(inputs=model.input, outputs=model.get_layer(index=3).output)

    for noisy_imgs, _ in dataset.take(1):
        encoded_outputs = encoder.predict(noisy_imgs[:num_images])

        for i in range(num_images):
            plt.figure(figsize=(15, 2))
            for j in range(num_channels):
                ax = plt.subplot(1, num_channels, j + 1)
                plt.imshow(encoded_outputs[i, :, :, j], cmap='viridis')
                plt.title(f"Ch {j+1}")
                plt.axis("off")
            plt.suptitle(f"Bottleneck Features (Image {i+1})")
            plt.tight_layout()
            plt.show()


def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.5)
    noisy_image = tf.clip_by_value(image + noise, 0.0, 1.0)
    return noisy_image, image

BATCH_SIZE = 128

train_dataset = ds_train.map(preprocess).cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = ds_test.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

from tensorflow.keras import layers, models

def build_autoencoder():
    input_img = layers.Input(shape=(28, 28, 1))

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = models.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder

autoencoder = build_autoencoder()
autoencoder.summary()

history = autoencoder.fit(train_dataset, epochs=10, validation_data=test_dataset)

def display_denoised_images(model, dataset, num_images=10):
    for noisy_imgs, clean_imgs in dataset.take(1):
        preds = model.predict(noisy_imgs)
        plt.figure(figsize=(20, 4))
        for i in range(num_images):
            ax = plt.subplot(3, num_images, i + 1)
            plt.imshow(noisy_imgs[i].numpy().squeeze(), cmap='gray')
            plt.title("Noisy")
            plt.axis("off")

            ax = plt.subplot(3, num_images, i + 1 + num_images)
            plt.imshow(clean_imgs[i].numpy().squeeze(), cmap='gray')
            plt.title("Clean")
            plt.axis("off")

            ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
            plt.imshow(preds[i].squeeze(), cmap='gray')
            plt.title("Denoised")
            plt.axis("off")
        plt.show()

display_denoised_images(autoencoder, test_dataset)
display_bottleneck_features(autoencoder, test_dataset)