import argparse
import os

import librosa
import numpy as np
import pytorch_lightning as pl
import soundfile as sf
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from config import CONFIG
from dataset import CustomDataset
from models.tunet import TUNet
from utils.tblogger import TensorBoardLoggerExpanded
from utils.utils import evaluate_dataset, frame, mkdir_p, overlap_add

parser = argparse.ArgumentParser()

parser.add_argument('--version', default=None,
                    help='version to resume')
parser.add_argument('--mode', default='train',
                    help='training or testing mode')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(CONFIG.gpus)
assert args.mode in ['train', 'eval', 'test'], "--mode should be 'train', 'eval' or 'test'"


def resume(train_dataset, val_dataset, version):
    print("Version", version)
    model_path = os.path.join(CONFIG.LOG.log_dir, 'version_{}/checkpoints/'.format(str(version)))
    config_path = os.path.join(CONFIG.LOG.log_dir, 'version_{}/'.format(str(version)) + 'hparams.yaml')
    model_name = [x for x in os.listdir(model_path) if x.endswith(".ckpt")][0]
    ckpt_path = model_path + model_name
    checkpoint = TUNet.load_from_checkpoint(ckpt_path, hparams_file=config_path, train_dataset=train_dataset,
                                            val_dataset=val_dataset)
    return checkpoint


def train():
    train_dataset = CustomDataset('train')
    val_dataset = CustomDataset('val')

    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', verbose=True,
                                          filename='tunet-{epoch:02d}-{val_loss:.4f}', save_weights_only=False)
    gpus = CONFIG.gpus.split(',')
    logger = TensorBoardLoggerExpanded(CONFIG.DATA.sr)
    if args.version is not None:
        model = resume(train_dataset, val_dataset, args.version)
    else:
        model = TUNet(train_dataset, val_dataset)
    trainer = pl.Trainer(logger=logger,
                         gradient_clip_val=CONFIG.TRAIN.clipping_val,
                         gpus=len(gpus),
                         max_epochs=CONFIG.TRAIN.epochs,
                         accelerator="ddp" if len(gpus) > 1 else None,
                         stochastic_weight_avg=True,
                         callbacks=[checkpoint_callback])

    print(model.hparams)
    print(
        'Dataset: {}, Train files: {}, Val files {}'.format(CONFIG.DATA.dataset, len(train_dataset), len(val_dataset)))
    trainer.fit(model)


def evaluate(model):
    sample_path = os.path.join(CONFIG.LOG.sample_path, "version_" + str(args.version))
    testset = CustomDataset('test')
    test_loader = DataLoader(testset, batch_size=1, num_workers=CONFIG.TRAIN.workers)
    res = evaluate_dataset(model, test_loader, sample_path, False)
    print("Version {} -- LSD: {} LSD-HF: {} SI-SDR: {}".format(args.version, res[0], res[1], res[2]))


def test(model):
    in_dir = CONFIG.TEST.in_dir
    out_dir = CONFIG.TEST.out_dir
    files = os.listdir(in_dir)
    mkdir_p(out_dir)
    window_size = CONFIG.DATA.window_size
    stride = CONFIG.DATA.stride

    for file in files:
        sig, sr = librosa.load(os.path.join(in_dir, file), sr=CONFIG.DATA.sr)
        d = max(len(sig) // stride + 1, 2) * stride
        sig = np.hstack((sig, np.zeros(d - len(sig))))
        x = frame(sig, window_size, stride)[:, np.newaxis, :]
        x = torch.Tensor(x).cuda(device=0)
        pred = model(x)
        pred = np.squeeze(pred.detach().cpu().numpy(), 1)
        audio = overlap_add(pred, window_size, stride)
        sf.write(os.path.join(out_dir, 'recon_' + file), audio, samplerate=sr, subtype='PCM_16')


if __name__ == '__main__':

    if args.mode == 'train':
        train()
    else:
        model = resume(None, None, args.version)
        print(model.hparams)
        model.summarize()
        model.eval().cuda(device=0)
        model.freeze()
        if args.mode == 'eval':
            evaluate(model)
        else:
            test(model)
