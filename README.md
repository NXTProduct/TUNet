# TUNet - Official Implementation
**TUNet: A Block-online Bandwidth Extension Model based on Transformers and Self-supervised Pretraining**

Viet-Anh Nguyen, Anh H. T. Nguyen, and Andy W. H. Khong

[![arXiv](https://img.shields.io/badge/arXiv-2110.13492-<COLOR>.svg)](https://arxiv.org/abs/2110.13492)



# 1. Installation
## Setup
### Clone the repo
```
$ git clone https://github.com/kuleshov/audio-super-res.git
$ cd TUNet
```
### Install dependencies
* Our implementation requires the `libsndfile` and `libsamplerate` libraries for the Python packages `soundfile` and `samplerate`, respectively. On Ubuntu, they can be easily installed using `apt-get`:
    ```
    $ apt-get update && apt-get install libsndfile-dev libsamplerate-dev
    ```
*  Create a Python 3.8 environment. Conda is recommended:
    ```
    $ conda create -n tunet python=3.8
    $ conda activate tunet
    ```

* Install the requirements:
    ```
    $ pip install -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html
    ```
**Note**: the argument `-f https://download.pytorch.org/whl/cu113/torch_stable.html` is provided to install `torch==1.10.0+cu113`  (Pytorch 1.10, CUDA 11.3) inside the `requirements.txt` . Choose an appropriate CUDA version to your GPUs and change/remove the argument according to [PyTorch documentation](https://pytorch.org/get-started/locally/)
# 2. Data preparation
In our paper, we conduct experiments on the [VCTK](https://datashare.ed.ac.uk/handle/10283/3443) and [VIVOS](https://ailab.hcmus.edu.vn/vivos) datasets. You may use either one or both.

* Download and extract the datasets:
    ```
    $ wget http://www.udialogue.org/download/VCTK-Corpus.tar.gz -O data/vctk/
    $ wget https://ailab.hcmus.edu.vn/assets/vivos.tar.gz -O data/vivos/
    $ tar -zxvf data/vctk/VCTK-Corpus.tar.gz 
    $ tar -zxvf data/vivos/vivos.tar.gz 
    ```

    After extracting the datasets, your `./data` directory should look like this:

    ```
    .
    |--data
        |--vctk
            |--raw
                |--VCTK-Corpus
                    |--wav48
                        |--p225
                            |--p225_001.wav
                            ...

         |--vivos
                |--train
                    |--waves
                        |--VIVOSSPK01
                            |--VIVOSSPK12_R001.wav
                            ...                
                |--test
                    |--waves
                        |--VIVOSDEV01
                            |--VIVOSDEV01_R001.wav
                            ...                                    
    ```
* In order to load the datasets, text files that contain training and testing audio paths are required. We have prepared `train.txt` and `test.txt` files in `./data/vctk` and `./data/vivos` directories.

# 3. Run the code
## Configuration
`config.py` is the most important file. Here, you can find all the configurations related to experiment setups, datasets, models, training, testing, etc. Although the config file has been explained thoroughly, we recommend reading our paper to fully understand each parameter.

## Training
* Adjust training hyperparameters in `config.py` 

    **Note:** `batch_size` in this implementation is different from the batch size in the paper. Specifically, we infer "batch size" in our paper as the number of **frames** per batch, whereas in this repo, `batch_size` is the number of **audio files** per batch. The DataLoader loads batches of audio files then chunks into frames on the fly. Since audio duration is variable, the number of frames per batch varies around 12*`batch_size` .
* Run `main.py`:
    ```
    $ python main.py --mode train
    ```
* Each run will create a version in `./lightning_logs`, where the model checkpoint and hyperparameters are saved. In case you want to continue training from one of these versions, just set the argument `--version` of the above command to your desired version number. For example:
    ```
    # resume from version 0
    $ python main.py --mode train --version 0
    ```
* To monitor the training curves as well as inspect model output visualization, run the tensorboard:
    ```
    $ tensorboard --logdir=./lightning_logs --bind_all
    ```
    ![image.png](https://images.viblo.asia/8da3b9e0-d9e8-470a-ae49-f3d8962fe130.png)
    ![image.png](https://images.viblo.asia/75e40509-c36a-4055-af73-36ffd777ba87.png)

## Evaluation
* Modify `config.py` to change evaluation setup if necessary.
* Run `main.py` with a version number to be evaluated:
    ```
    $ python main.py --mode eval --version 0
    ```
     This will give the mean and standard deviation of LSD, LSD-HF, and SI-SDR, respectively. During the evaluation, several output samples are saved to `config.LOG.sample_path` for sanity testing.

## Audio generation
* In order to generate output audios, you need to either put your input samples into `./test_samples` or modify `config.TEST.in_dir` to your input directory. 
* Run `main.py`:
    ```
    python main.py --mode test --version 0
    ```
    The generated audios are saved to `config.TEST.out_dir`.

## Configure a new dataset
Our implementation currently works with the VCTK and VIVOS datasets but can be easily extensible to a new one.
* Firstly, you need to prepare `train.txt` and `test.txt`. See `./data/vivos/train.txt` and `./data/vivos/test.txt` for example.
* Secondly, add a new dictionary to `config.DATA.data_dir`:
    ```
    {
    'root': 'path/to/data/directory',
    'train': 'path/to/train.txt',
    'test': 'path/to/test.txt'
    }
    ```
    **Important:** Make sure each line in `train.txt` and `test.txt` joining with `'root'` is a valid path to its corresponding audio file.

# 4. Citation
```
@misc{nguyen2021tunet,
      title={TUNet: A Block-online Bandwidth Extension Model based on Transformers and Self-supervised Pretraining}, 
      author={Viet-Anh Nguyen and Anh H. T. Nguyen and Andy W. H. Khong},
      year={2021},
      eprint={2110.13492},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
