sudo docker build -t my_tensorflow_image .
sudo docker run --gpus all -v $(pwd):/app my_tensorflow_image
