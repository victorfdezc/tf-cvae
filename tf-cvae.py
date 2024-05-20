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

def preprocess_images(images, img_shape, channels):
  images = (images.reshape((images.shape[0], img_shape, img_shape, channels)) / 255.).astype('float32')
  # return np.where(images > .5, 1.0, 0.0).astype('float32')
  return images

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


def extract_frames(video_path, img_channels):
  video_reader = imageio.get_reader(video_path)
  frames = []
  for frame in video_reader:
    # Convert frame to PIL Image
    pil_frame = Image.fromarray(frame)
    # Convert to grayscale
    if img_channels==1: pil_frame = pil_frame.convert('L')
    frames.append(pil_frame)
  return frames

def read_image(image_path, image_shape, channels=3):
  # Read an image file
  image = PIL.Image.open(image_path)
  image.resize((image_shape, image_shape), Image.Resampling.LANCZOS)
  if channels == 1: 
    image = image.convert('L')
    image = np.array(image)
  elif channels == 3: 
    # Get only RGB channels
    image = np.array(image)
    image = image[:,:,:channels]
  return np.array(image)

def read_dataset(dataset_path, image_shape=256, channels=3):
  # Check if the path exists
  if not os.path.exists(dataset_path):
      print("The specified path does not exist.")
      return
  # Check if the path is a directory
  if not os.path.isdir(dataset_path):
      print("The specified path is not a directory.")
      return
  # List all files in the directory
  files = os.listdir(dataset_path)
  
  dataset = []
  for file in files:
    dataset.append(read_image(os.path.join(dataset_path, file), image_shape, channels))
  return dataset

def resize_image(image, image_shape):
  # Extend dimension if grayscale
  if len(np.shape(image)) == 2:  image = image[..., np.newaxis]
  ## TODO: FLOAT16? OR FLOAT32?
  return tf.image.resize(image, [image_shape, image_shape])

def resize_dataset_images(dataset, image_shape):
  resized_dataset = []
  for image in dataset:
    resized_dataset.append(resize_image(image, image_shape))
  return np.array(resized_dataset)

def normalize_images(images):
  # Normalize the image by 255
  return np.array(images) / 255.0

def inference_image(model, image):
  reshaped_image = tf.expand_dims(image, axis=0)
  mean, logvar = model.encode(reshaped_image)
  z = model.reparameterize(mean, logvar)
  predictions = model.sample(z)
  return predictions[0, :, :, :]

def save_image(image,img_name):
  fig = plt.figure()
  plt.imshow(image)
  plt.axis('off')
  plt.savefig(img_name + '.png')
  

def split_frames(frames, num_images_set1):
    set1 = frames[:num_images_set1]
    set2 = frames[num_images_set1:]
    return set1, set2


batch_size = 4
epochs = 1
# set the dimensionality of the latent space to a plane for visualization later
latent_dim = 128
num_examples_to_generate = 1
img_shape = 200
img_channels = 3


# (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

frames = extract_frames("data_in/video_cortito.mp4", img_channels)
print("Number of frames:", np.shape(frames))
train_images, test_images = split_frames(frames, 2400)

# read images
# train_images = read_dataset("data_in/selulitis/", img_channels)
# train_images, test_images = split_frames(train_images, 1)
# train_images = read_dataset("data_in/landscapes/", img_shape, channels = img_channels)
# train_images, test_images = split_frames(train_images, int(len(train_images)/4*3))

# Get the data prepared
train_size = len(train_images)
test_size = len(test_images)
train_images = np.array(train_images)
test_images = np.array(test_images)
print("Number of frames in train_images:", train_size)
print("Number of frames in test_images:", test_size)
print("Shape of train_images:", np.shape(train_images))
print("Shape of test_images:", np.shape(test_images))


# Preprocess data
train_images = resize_dataset_images(train_images, img_shape)
train_images = preprocess_images(train_images,img_shape, img_channels)
test_images = resize_dataset_images(test_images, img_shape)
test_images = preprocess_images(test_images,img_shape, img_channels)

# Get the data prepared
train_size = len(train_images)
test_size = len(test_images)
train_images = np.array(train_images)
test_images = np.array(test_images)
print("Number of frames in train_images:", train_size)
print("Number of frames in test_images:", test_size)
print("Shape of train_images:", np.shape(train_images))
print("Shape of test_images:", np.shape(test_images))

# fig = plt.figure()
# plt.imshow(test_images[0, :, :, :].astype('float32'))
# plt.axis('off')
# plt.savefig('test_image.png')

# Read random images
random_images = read_dataset("data_in/test_images_random",channels=img_channels)
random_images = resize_dataset_images(random_images, img_shape)
random_images = preprocess_images(random_images,img_shape, img_channels)
print("Shape of chino:", np.shape(random_images))


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
model = CVAE(latent_dim, img_shape, img_channels, load_model = False)
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
save_image(test_sample[0],"test_sample_original")

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
save_image(inference_image(model, test_sample[0]),"test_sample_predicted_trained")
save_image(inference_image(model, random_images[0]),"chino_predicted_trained")
save_image(inference_image(model, random_images[1]),"test_image_predicted_trained")


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