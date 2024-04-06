# sudo docker build -t my_tensorflow_image .
sudo docker run --gpus all -v $(pwd):/app -v $(pwd)/tf-cvae.py:/app/tf-cvae.py my_tensorflow_image
