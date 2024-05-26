from IPython import display
import os
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
import tensorflow as tf
import tensorflow_probability as tfp
import time
# !pip install git+https://github.com/tensorflow/docs
import tensorflow_docs.vis.embed as embed
import gc
from tf_utils.train_vae import *
from tf_utils.manage_data import *
from tf_utils.process_images import *
from models.cvae import *
  
  
def generate_and_save_images(model, epoch, test_sample):
  mean, logvar = model.encode(test_sample)
  z = model.reparameterize(mean, logvar)
  predictions = model.sample(z)
  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
    plt.subplot(int(np.sqrt(predictions.shape[0])), int(np.sqrt(predictions.shape[0])), i + 1)
    plt.imshow(predictions[i, :, :, :])
    plt.axis('off')

  # tight_layout minimizes the overlap between 2 sub-plots
  plt.savefig('data_out/image_at_epoch_{:04d}.png'.format(epoch))
  # plt.show()


def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))


def plot_latent_images(model, n, digit_size=28):
  """Plots n x n digit images decoded from the latent space."""

  norm = tfp.distributions.Normal(0, 1)
  grid_x = norm.quantile(np.linspace(0.05, 0.95, n))
  grid_y = norm.quantile(np.linspace(0.05, 0.95, n))
  image_width = digit_size*n
  image_height = image_width
  image = np.zeros((image_height, image_width))

  for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
      z = np.array([[xi, yi]])
      x_decoded = model.sample(z)
      digit = tf.reshape(x_decoded[0], (digit_size, digit_size))
      image[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = digit.numpy()

  plt.figure(figsize=(10, 10))
  plt.imshow(image, cmap='Greys_r')
  plt.axis('Off')
  # plt.show()


def inference_image(model, image):
  reshaped_image = tf.expand_dims(image, axis=0)
  mean, logvar = model.encode(reshaped_image)
  z = model.reparameterize(mean, logvar)
  predictions = model.sample(z)
  return predictions[0, :, :, :]


##########################
#   GENERAL PARAMETERS   #
##########################
batch_size = 4
epochs = 100
latent_dim = 128 # set the dimensionality of the latent space to a plane for visualization later
num_examples_to_generate = 1
img_shape = 200
img_channels = 3
load_model = False # Load model according previous saved checkpoints
###########################################

#### Download dataset #####
# (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
####### Load a video #######
train_images = extract_video_frames("data_in/video_cortito.mp4", img_shape=img_shape, img_channels=img_channels)
### Load a set of images ###
# train_images = load_img_dataset("data_in/selulitis/", img_shape, img_channels = img_channels)
# train_images = load_img_dataset("data_in/landscapes/", img_shape, img_channels = img_channels)
#############################

# Preprocess data
train_images = shuffle_dataset(train_images)
train_images, test_images = split_dataset(train_images)
train_images = normalize_images(train_images)
test_images = normalize_images(test_images)

train_size = len(train_images)
test_size = len(test_images)
print("Shape of train images:", np.shape(train_images))
print("Shape of test images:", np.shape(test_images))

# Read random images
random_images = load_img_dataset("data_in/test_images_random", img_shape, img_channels = img_channels)
random_images = normalize_images(random_images)
print("Shape of random images:", np.shape(random_images))


# Shuffle and create TF Dataset
train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                 .shuffle(train_size).batch(batch_size))
del train_images
gc.collect()
test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                .shuffle(test_size).batch(batch_size))
del test_images
gc.collect()
                
# Define optimizer
optimizer = tf.keras.optimizers.Adam(1e-4)


# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, latent_dim])

# Create model
model = CVAE(latent_dim, img_shape, img_channels, load_model = load_model)
print("Encoder summary:\n") 
model.encoder.summary()
print("Decoder summary:\n")
model.decoder.summary()
encoder_checkpoint_path = "training_1/cp-encoder-{epoch:04d}.weights.h5"
decoder_checkpoint_path = "training_1/cp-decoder-{epoch:04d}.weights.h5"

# Save the weights using the `checkpoint_path` format
# model.encoder.save_weights(encoder_checkpoint_path.format(epoch=0))
# model.decoder.save_weights(decoder_checkpoint_path.format(epoch=0))

# Pick a sample of the test set for generating output images
assert batch_size >= num_examples_to_generate
for test_batch in test_dataset.take(1):
  test_sample = test_batch[0:num_examples_to_generate, :, :, :]
save_img(test_sample[0],"test_sample_original")

generate_and_save_images(model, 0, test_sample)

for epoch in range(1, epochs + 1):
  start_time = time.time()
  for train_x in train_dataset:
    train_step(model, train_x, optimizer)
  end_time = time.time()

  loss = tf.keras.metrics.Mean()
  for test_x in test_dataset:
    loss(compute_loss(model, test_x))
  elbo = -loss.result()
  display.clear_output(wait=False)
  print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
        .format(epoch, elbo, end_time - start_time))
  generate_and_save_images(model, epoch, test_sample)
  
  #plt.imshow(display_image(epoch))
  #plt.axis('off')  # Display images

# Save the weights using the `checkpoint_path` format
# model.encoder.save_weights(encoder_checkpoint_path.format(epoch=epochs))
# model.decoder.save_weights(decoder_checkpoint_path.format(epoch=epochs))

# Make some predictions
save_img(inference_image(model, test_sample[0]),"test_sample_predicted_trained")
save_img(inference_image(model, random_images[0]),"chino_predicted_trained")
save_img(inference_image(model, random_images[1]),"test_image_predicted_trained")


anim_file = 'data_out/cvae.gif'
with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('data_out/image*.png')
  filenames = sorted(filenames)
  for filename in filenames:
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)
  
embed.embed_file(anim_file)