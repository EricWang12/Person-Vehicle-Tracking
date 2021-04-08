## Setup guide


### Installing docker

```bash

# Update the apt package index and install packages to allow apt to use a repository over HTTPS

sudo apt-get update 

sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker’s official GPG key:
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg



#  install the latest version of Docker Engine and containerd:

sudo apt-get install docker-ce docker-ce-cli containerd.io

# Verify that Docker Engine is installed correctly
sudo docker run hello-world

```


Nvidia Docker

```bash
nvidia-smi

# Setup the stable repository and the GPG key:

distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list


# install nvidia docker
sudo apt-get install -y nvidia-docker2

sudo systemctl restart docker


# verify
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```


### Run the MAL inference

```bash
sudo docker run --gpus all -v .:/workspace --rm --ipc=host -it nvcr.io/nvidia/pytorch:19.10-py3

cd PVT

# Live above goes into nvidia docker and then run:
python setup.py clean --all install
python setup.py build develop


# Also: dependencies of deepsort
pip install -U -r requirements.txt


# In the future in order to get back to docker  you can use:
docker attach sharp_mcnulty 
```


### Downloading weights


```bash

# deepsort weight
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1_qwTWdzT9dWNudpusgKavj_4elGgbkUN' -O ckpt.t7 -P  retinanet/deep_sort_pytorch/deep_sort/deep/checkpoint/

```

ResNet50 Weight for MAL:
https://cmu.app.box.com/s/f70ewy7fh66bsb551v44hfskehgz07z3



and put it in models/resnet50/. Then change the name in models/transfer_model.py 

(I made slight adjustment to original transfer_model for better path control)


Then convert for  ODTK 

```bash
python models/transfer_model.py
```

### Download the dataset 

You can download the VisDrone From [HERE](https://github.com/VisDrone/VisDrone-Dataset)



you can also use [process-dataset.py](./process-dataset.py) to convert and resize VisDrone


### Run the model

```bash
# vanilla original model on COCO
CUDA_VISIBLE_DEVICES=0 retinanet infer --config "./configs/MAL_R-50-FPN_e2e.yaml"  --images ./COCO-DATASET-2017/val2017/   --annotations ./COCO-DATASET-2017/annotations/instances_val2017.json --batch=1


# changed inference

python retinanet/main.py infer --config "./configs/MAL_R-50-FPN_e2e.yaml"  --images $YOUR-VISDRONE-SEQUENCE-FOLDER   --annotations $CONVERTED-JSON-ANNOTATION

```


### MISC

convert frames to 16fps video

```bash

ffmpeg -r 16 -i uav0000073_04464_v-%07d.jpg  -qscale 0 -vcodec mpeg4 -y ../../MOT-output.mp4

```