import glob
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.applications import VGG19

def build_resnet_block(x, filters, kernel_size=3, stride=1, padding='same'):
    """
    Residual block
    """
    res = layers.Conv2D(filters, kernel_size, strides=stride, padding=padding)(x)
    res = layers.BatchNormalization()(res)
    res = layers.LeakyReLU(alpha=0.2)(res)
    return res

def build_generator(input_shape):
    """
    Build the generator network
    """
    x_in = layers.Input(shape=input_shape)
    
    # Initial Convolution
    x = layers.Conv2D(64, 9, strides=1, padding='same', activation='relu')(x_in)
    
    # Residual Blocks
    for _ in range(16):
        x = build_resnet_block(x, 64)
    
    # Post-residual block
    x = layers.Conv2D(64, 3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Skip connection
    x = layers.Add()([x, x_in])
    
    # Upsampling Blocks
    x = layers.UpSampling2D(size=2)(x)
    x = layers.Conv2D(256, 3, strides=1, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    x = layers.UpSampling2D(size=2)(x)
    x = layers.Conv2D(256, 3, strides=1, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    # Output layer
    x = layers.Conv2D(3, 9, strides=1, padding='same', activation='tanh')(x)
    
    generator = Model(x_in, x, name='generator')
    return generator

def build_discriminator(input_shape):
    """
    Build the discriminator network
    """
    x_in = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(64, 3, strides=1, padding='same')(x_in)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    x = layers.Conv2D(64, 3, strides=2, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    
    x = layers.Conv2D(128, 3, strides=1, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    
    x = layers.Conv2D(128, 3, strides=2, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    
    x = layers.Conv2D(256, 3, strides=1, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    
    x = layers.Conv2D(256, 3, strides=2, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    
    x = layers.Conv2D(512, 3, strides=1, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    
    x = layers.Conv2D(512, 3, strides=2, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    x_out = layers.Dense(1, activation='sigmoid')(x)
    
    discriminator = Model(x_in, x_out, name='discriminator')
    return discriminator

def build_vgg(input_shape):
    """
    Build VGG network to extract image features
    """
    vgg = VGG19(weights="imagenet", include_top=False, input_shape=input_shape)
    vgg.trainable = False
    model = Model(inputs=vgg.input, outputs=vgg.get_layer('block5_conv4').output, name='vgg')
    return model

def build_adversarial_model(generator, discriminator, vgg, input_shape):
    """
    Build and compile the adversarial model
    """
    input_high_resolution = layers.Input(shape=input_shape)
    input_low_resolution = layers.Input(shape=(input_shape[0] // 4, input_shape[1] // 4, input_shape[2]))
    
    # Generate high-resolution images from low-resolution images
    generated_high_resolution_images = generator(input_low_resolution)
    
    # Extract feature maps of the generated images using VGG
    features = vgg(generated_high_resolution_images)
    
    # Make the discriminator network as non-trainable
    discriminator.trainable = False
    
    # Get the probability of generated high-resolution images
    probs = discriminator(generated_high_resolution_images)
    
    # Create and compile the adversarial model
    adversarial_model = Model([input_low_resolution, input_high_resolution], [probs, features], name='adversarial')
    adversarial_model.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[1e-3, 1], optimizer=Adam(0.0002, 0.5))
    
    return adversarial_model

def sample_images(data_dir, batch_size, high_resolution_shape, low_resolution_shape):
    """
    Sample a batch of images from the dataset directory
    """
    # Make a list of all images inside the data directory
    all_images = glob.glob(os.path.join(data_dir, '*.*'))
    
    # Choose a random batch of images
    images_batch = np.random.choice(all_images, size=batch_size)
    
    low_resolution_images = []
    high_resolution_images = []
    
    for img in images_batch:
        # Load and preprocess the image
        img = load_img(img, target_size=high_resolution_shape)
        img_hr = img_to_array(img)
        img_lr = img_to_array(img.resize(low_resolution_shape))
        
        # Normalize images to the range [-1, 1]
        img_hr = (img_hr - 127.5) / 127.5
        img_lr = (img_lr - 127.5) / 127.5
        
        high_resolution_images.append(img_hr)
        low_resolution_images.append(img_lr)
    
    return np.array(high_resolution_images), np.array(low_resolution_images)

def save_images(low_resolution_image, original_image, generated_image, path):
    """
    Save low-resolution, high-resolution(original), and generated high-resolution images in a single image
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(low_resolution_image)
    axes[0].set_title("Low Resolution")
    axes[0].axis('off')
    
    axes[1].imshow(original_image)
    axes[1].set_title("Original")
    axes[1].axis('off')
    
    axes[2].imshow(generated_image)
    axes[2].set_title("Generated")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def train(epochs, batch_size, data_dir, tensorboard_log_dir):
    # Input shape for images
    input_shape = (256, 256, 3)
    
    # Build and compile the generator, discriminator, and VGG models
    generator = build_generator(input_shape)
    discriminator = build_discriminator(input_shape)
    vgg = build_vgg(input_shape)
    
    adversarial_model = build_adversarial_model(generator, discriminator, vgg, input_shape)
    
    # Set up Tensorboard for logging
    tensorboard = TensorBoard(log_dir=tensorboard_log_dir)
    tensorboard.set_model(generator)
    tensorboard.set_model(discriminator)
    
    for epoch in range(epochs):
        print("Epoch: {}".format(epoch))
        
        # Train the discriminator
        # ...
        
        # Train the generator
        # ...
        
        # Save images and log data
        if epoch % 100 == 0:
            high_resolution_images, low_resolution_images = sample_images(data_dir, batch_size, input_shape, (64, 64, 3))
            generated_images = generator.predict_on_batch(low_resolution_images)
            
            for i, (low_res, orig, gen) in enumerate(zip(low_resolution_images, high_resolution_images, generated_images)):
                save_images(low_res, orig, gen, "results/img_{}_{}".format(epoch, i))
        
        # Save models
        if (epoch + 1) % 5000 == 0:
            generator.save("generator_model_{}.h5".format(epoch + 1))
            discriminator.save("discriminator_model_{}.h5".format(epoch + 1))

if __name__ == '__main__':
    data_dir = "/path/to/dataset/directory"
    epochs = 30000
    batch_size = 1
    tensorboard_log_dir = "logs/"  # Set your log directory path
    train(epochs, batch_size, data_dir, tensorboard_log_dir)
