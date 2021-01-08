import tensorflow as tf

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

class MNIST_DCGAN(object):
    def __init__(self, buffer_size = 60000, batch_size = 256, epochs = 50):
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.BUFFER_SIZE = buffer_size
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.noise_dim = 100
        self.num_examples_to_generate = 16
        self.checkpoint_dir = './training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")

        self.generator = self.make_generator_model()
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator = self.make_discriminator_model()
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                         discriminator_optimizer=self.discriminator_optimizer,
                                         generator=self.generator,
                                         discriminator=self.discriminator)
        self.seed = tf.random.normal([self.num_examples_to_generate, self.noise_dim])
        self.gen_train_loss =  tf.keras.metrics.Mean(name='gen_train_loss')
        self.disc_train_loss = tf.keras.metrics.Mean(name='disc_train_loss')


    def make_generator_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((7, 7, 256)))
        assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 7, 7, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 14, 14, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(
            layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 28, 28, 1)

        return model

    def make_discriminator_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                input_shape=[28, 28, 1]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        self.gen_train_loss(gen_loss)
        self.disc_train_loss(disc_loss)


    def train(self, dataset, epochs):
        for epoch in range(epochs):
            # Reset the metrics at the start of the next epoch
            self.gen_train_loss.reset_states()
            self.disc_train_loss.reset_states()

            start = time.time()

            for image_batch in dataset:
                self.train_step(image_batch)


            # Produce images for the GIF as we go
            self.generate_and_save_images(self.generator,
                                     epoch + 1,
                                     self.seed)

            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)
        print(
            f'Epoch {epoch + 1}, '
            f'GLoss: {self.gen_train_loss.result()}, '
            f'DLoss: {self.disc_train_loss.result()}, '
        )

        # Generate after the final epoch
        self.generate_and_save_images(self.generator,
                                 epochs,
                                 self.seed)

    def generate_and_save_images(self, model, epoch, test_input):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        if not os.path.exists('images'):
            os.makedirs('images')
        predictions = model(test_input, training=False)

        fig = plt.figure(figsize=(4, 4))
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')
        plt.savefig('images/image_at_epoch_{:04d}.png'.format(epoch))

    # Display a single image using the epoch number
    def display_image(epoch_no):
        return PIL.Image.open('images/image_at_epoch_{:04d}.png'.format(epoch_no))


if __name__ == '__main__':
    print("Tensorflow Version:", tf.__version__)

    BUFFER_SIZE = 60000
    BATCH_SIZE = 128
    EPOCHS = 2

    # Get the dataset
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    # Train the model
    mnist_dcgan = MNIST_DCGAN(buffer_size = BUFFER_SIZE, batch_size = BATCH_SIZE, epochs = EPOCHS)
    mnist_dcgan.train(train_dataset, epochs=2)

    # Restore last checkpoint and display images
    mnist_dcgan.checkpoint.restore(tf.train.latest_checkpoint(mnist_dcgan.checkpoint_dir))
    # mnist_dcgan.display_image(mnist_dcgan.EPOCHS)

    # Generate gif from each epoch
    anim_file = 'dcgan.gif'
    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob('images/image*.png')
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)



