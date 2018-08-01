# Raccoon-detection
Object detection of raccoons in Tensorflow API

Download /model/research repository from https://github.com/tensorflow/models
Set this repository in your tensorflow(terminate/Anaconda) directory

The code is made to reproduce the course--https://blog.csdn.net/chenmaolin88/article/details/79357263
# Training
sudo -i
cd /path/to/tensorflow/models/research
(root)# TF_CPP_MIN_LOG_LEVEL="2" CUDA_VISIBLE_DEVICES="1" python object_detection/train.py --logtostderr --pipeline_config_path=/home/amax/SIAT/Raccoon_detection/ssd_mobilenet_v1_coco_2017_11_17/ssd_mobilenet_v1_raccoon.config --train_dir=/home/amax/SIAT/Raccoon_detection/train

# Validation
In the same directory

CUDA_VISIBLE_DEVICES="0" TF_CPP_MIN_LOG_LEVEL="2" \
python object_detection/eval.py \
--logtostderr \
--pipeline_config_path='/path/to/ssd_mobilenet_v1_coco_2017_11_17/ssd_mobilenet_v1_raccoon.config' \
--checkpoint_dir='/path/to/Raccoon_detection/train' \
--eval_dir='/path/to/Raccoon_detection/val'

# Tensorboard
tensorboard --logdir ./SIAT/Raccoon_detection

# frozen graph
cd ./object_detection
python export_inference_graph.py \
--pipeline_config_path=/path/to/ssd_mobilenet_v1_coco_2017_11_17/ssd_mobilenet_v1_raccoon.config \
--trained_checkpoint_prefix=/path/to/Raccoon_detection/train/model.ckpt-42883 \
--output_directory=/home/amax/SIAT/Raccoon_detection/train
