# Interactive Curation of Datasets for Training and Refining Generative Models

The main contributors of this repository include Wenjie Ye, [Yue Dong](http://yuedong.shading.me) and [Pieter Peers](http://www.cs.wm.edu/~ppeers/).

## Introduction

This repository provides a reference implementation for the PG 2019 paper "Interactive Curation of Datasets for Training and Refining Generative Models".

More information (including a copy of the paper) can be found at *****.

## Citation
If you use our code or models, please cite:

```
@article{Ye:2019:ICD, 
 author = {Ye, Wenjie and Dong, Yue and Peers, Pieter},
 title = {Interactive Curation of Datasets for Training and Refining Generative Models},
 year = {2019},
 journal = {Computer Graphics Forum},
 volume = {38},
 number = {7},
 pages = {***--***},
 }
```

----------------------------------------------------------------
## Usage

### System requirements
- A Linux system with python Tensorflow-GPU environment. 

### Preparations
##### Prepare necessary code and data.
- Clone this repository.
- If you want to experiment with FFHQ face images:

    Download FFHQ dataset from https://github.com/NVlabs/ffhq-dataset. Only 1024x1024 images are needed. Put all the image files into a single folder. 

    Download pretrained StyleGAN FFHQ model named "stylegan-ffhq-1024x1024.pkl" from https://github.com/NVlabs/stylegan and put it into "data" folder. 

    Download pretrained FaceNet model named "20180402-114759" from https://github.com/davidsandberg/facenet and put it into "data" folder.

- If you want to experiment with bedroom images:

    Download pretrained StyleGAN bedroom model named "stylegan-bedrooms-256x256.pkl" from https://github.com/NVlabs/stylegan and put it into "data" folder. 

    Download "vgg16.py" from https://github.com/machrisaa/tensorflow-vgg and put it into the repository root directory. 

    Download pretrained vgg16 model from https://github.com/machrisaa/tensorflow-vgg and put it into "data" folder.

- If you want to experiment with wood, metal and stone texture images:

    Download pretrained GAN models from ***** and put them into "data" folder. 

##### Configuration
- Modify the "config.py" file.

    Change the variable "outputroot" to a directory to which you want to write output files. 

    Change the variable "ffhq_path" to the directory where you put all the FFHQ images (if downloaded). 

- (Conditional) If during running the pre-compiled "expand.so" works incorrect, please compile it yourself. 

      g++ ./expand.cpp -o expand.so

### Run the system.
##### Step 1: Start the dataset curation system. 
Run "main_ui.py" to start the system. Example:

      python main_ui.py -datatype face -enable_simul 1 -gpuid 0 -experiment_name curation_face

Arguments:
- datatype: which data source to experiment on. Valid values are "face", "bedroom", "wood", "metal", "stone". 
- enable_simul: whether to enable simultaneous labeling and training/selecting, or to perform labeling, training, selecting sequentially. 
- gpuid: which GPU card to use. Multiple card is supported. Use comma to separate.
- experiment_name: a name to identify the experiment. It will be used as the name of output folder.

All Arguments are optional and can be omitted. 

By default, the interactive system will run on port 5001, which can be changed in "main_ui.py".

Wait until the system initialzation finishes. At that point, you should see "Running on https://0.0.0.0:5001/ (Press CTRL+C to quit)" in the console. 

Note: FFHQ data source needs heavy precomputation in the first run and will take a long time. 

##### Step 2: Interactive label the images. 
Use a modern browser to visit "https://**#computer#**:5001/home", follow the UI to label the images. 

**#computer#** should be substituted with the server name or IP address if the system is running on a remote server, or "localhost" if it is running locally. 

After several rounds, you can stop labelling and go to the next step. After the training of each round, a user-intent classifier model will be saved in the output folder. 

##### Step 3: Finetune the original GAN model. 
For FFHQ face or bedroom (StyleGAN models), an example for running the finetuning is:

      python filter_and_train.py -datatype face -classifier_model PATH_TO_CLASSIFIER_MODEL -gpuid 0

For texture models, an example for running the finetuning is:

      python filter_and_train_texture.py -datatype wood -classifier_model PATH_TO_CLASSIFIER_MODEL -gpuid 0

Argument "classifier_model" is the path to the user-intent classifier model obtained in step 2, which should have the form ".../model.ckpt".

The system will generate the dataset for finetuning and run the GAN finetuning. After it finishes, you will get the finetuned model and some samples in a folder created under the "outputroot" directory. 

## Acknowledgement
[StyleGAN](https://github.com/NVlabs/stylegan) related code in this repository is provided by NVIDIA under Creative Commons Attribution-NonCommercial 4.0 International License (http://creativecommons.org/licenses/by-nc/4.0/). We made modifications on the original code. 

[FaceNet](https://github.com/davidsandberg/facenet) related code in this repository is provided by davidsandberg under MIT License.

A part of texture GAN related code is provided by Xiao Li. 

[VGG](https://github.com/machrisaa/tensorflow-vgg) related code is provided by machrisaa. 

This repository is provided for non-commercial use only, without any warranty. If you use this repository, you also need to agree to the license of the preceding code providers. 

## Contact
You can contact Wenjie Ye (ywjleft@gmail.com) if you have any problems.

## Reference
[1] YE W., DONG Y., PEERS P.: Interactive curation of datasets for training and refining generative models. Computer Graphics Forum 38, 7 (2019). 

[2] KARRAS T., LAINE S., AILA T.: A style-based generator architecture for generative adversarial networks. In CVPR (2019). 

[3] SCHROFF F., KALENICHENKO D., PHILBIN J.: Facenet: A unified embedding for face recognition and clustering. In CVPR (2015), pp. 815–823.

[4] SIMONYAN K., ZISSERMAN A.: Very deep convolutional networks for large-scale image recognition. In ICLR (2015)
