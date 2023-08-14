import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Generator model
def build_generator(latent_dim):
    model = tf.keras.Sequential()
    model.add(layers.Dense(256, input_dim=latent_dim, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(28 * 28, activation='sigmoid'))
    model.add(layers.Reshape((28, 28)))
    return model

# Discriminator model
def build_discriminator(latent_dim):
    model = tf.keras.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='linear'))
    return model

# GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = tf.keras.Sequential([generator, discriminator])
    return model

# Training loop
def train_gan(gan, dataset, latent_dim, epochs=10, batch_size=72):  # Updated batch size
    generator, discriminator = gan.layers
    for epoch in range(epochs):
        for batch in dataset:
            # Training the discriminator
            noise = tf.random.normal(shape=(batch_size, latent_dim))
            generated_images = generator(noise)
            real_images = batch
            combined_images = tf.concat([generated_images, real_images], axis=0)
            labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
            labels += 0.05 * tf.random.uniform(labels.shape)  # Adding noise to labels
            discriminator.trainable = True
            d_loss = discriminator.train_on_batch(combined_images, labels)

            # Training the generator
            noise = tf.random.normal(shape=(batch_size, latent_dim))
            labels = tf.ones((batch_size, 1))
            discriminator.trainable = False
            g_loss = gan.train_on_batch(noise, labels)

        print(f"Epoch {epoch + 1}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}")

        # Generate and plot images at the end of each epoch
        if epoch % 5 == 0:
            generate_and_plot(generator, latent_dim)

# Generating and plotting images
def generate_and_plot(generator, latent_dim, n_samples=10):
    noise = tf.random.normal(shape=(n_samples, latent_dim))
    generated_images = generator.predict(noise)

    plt.figure(figsize=(10, 1))
    for i in range(n_samples):
        plt.subplot(1, n_samples, i+1)
        plt.imshow(generated_images[i], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load and preprocess the dataset (e.g., MNIST)
    (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.0

    # Limit the dataset size to 4992 samples (divisible evenly by 72, the batch size)
    x_train = x_train[:4992]

    # Set the latent dimension (noise vector size)
    latent_dim = 100

    # Build the models (same as before)
    generator = build_generator(latent_dim)
    discriminator = build_discriminator(latent_dim)

    # Compile the models (same as before)
    discriminator.compile(optimizer='adam', loss='mean_squared_error')
    gan = build_gan(generator, discriminator)
    gan.compile(optimizer='adam', loss='mean_squared_error')

    # Create the dataset and set the correct batch size
    batch_size = 72
    dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(4992).batch(batch_size, drop_remainder=True)

    # Train the GAN (same as before)
    train_gan(gan, dataset, latent_dim, epochs=50, batch_size=batch_size)



