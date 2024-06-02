import time
import gc
from tf_utils.manage_data import *
from tf_utils.process_images import *
from models.cvae import *
  


##########################
#   GENERAL PARAMETERS   #
##########################
batch_size = 4
epochs = 1
latent_dim = 128 # set the dimensionality of the latent space to a plane for visualization later
num_examples_to_generate = 1
image_shape = 200
image_channels = 3
load_model = True # Load model according previous saved checkpoints
###########################################

#### Download dataset #####
# (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
####### Load a video #######
train_images = extract_video_frames("data_in/video_cortito.mp4", img_shape=image_shape, img_channels=image_channels)
### Load a set of images ###
# train_images = load_image_dataset("data_in/selulitis/", image_shape, image_channels = image_channels)
# train_images = load_image_dataset("data_in/landscapes/", image_shape, image_channels = image_channels)
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
random_images = load_image_dataset("data_in/test_images_random", image_shape, img_channels = image_channels)
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
model = CVAE(latent_dim, image_shape, image_channels, load_model = load_model)

# Pick a sample of the test set for generating output images
assert batch_size >= num_examples_to_generate
for test_batch in test_dataset.take(1):
  test_sample = test_batch[0:num_examples_to_generate, :, :, :]
save_image(test_sample[0],"test_sample_original")

generated_images = model.inference_batch_images(test_sample)
save_image_matrix(generated_images, 'data_out/image_at_epoch_{:04d}.png'.format(0))

for epoch in range(1, epochs + 1):
  start_time = time.time()
  for train_x in train_dataset:
    model.train_step(train_x, optimizer)
  end_time = time.time()

  loss = tf.keras.metrics.Mean()
  for test_x in test_dataset:
    loss(model.compute_loss(test_x))
  elbo = -loss.result()

  print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
        .format(epoch, elbo, end_time - start_time))
  generated_images = model.inference_batch_images(test_sample)
  save_image_matrix(generated_images, 'data_out/image_at_epoch_{:04d}.png'.format(epoch))
  

# Save the weights using the `checkpoint_path` format
if not load_model:
  model.encoder.save_weights("training_1/cp-encoder-{epoch:04d}.weights.h5".format(epoch=epochs))
  model.decoder.save_weights("training_1/cp-decoder-{epoch:04d}.weights.h5".format(epoch=epochs))

# Make some predictions
save_image(model.inference_image(test_sample[0]),"test_sample_predicted_trained")
save_image(model.inference_image(random_images[0]),"chino_predicted_trained")
save_image(model.inference_image(random_images[1]),"test_image_predicted_trained")

'''
i=0
for image in model.sample():
  save_image(model.sample()[0],"model_sample_random"+str(i))
  i+=1
'''
save_gif('data_out/cvae', re_images_name='data_out/image*.png')