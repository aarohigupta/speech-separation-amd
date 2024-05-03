### Important for ensuring the proper numpy version
# import subprocess
# commands = 'pip uninstall numpy; pip uninstall numba; python3 -m pip install numba'
# output = subprocess.run(commands, shell=True, capture_output=True, text=True)
# print(f"Output of subprocess: {output}")

import argparse
from numpy.random import default_rng
import numpy as np
import torch
import torch.nn as nn
import os
from pathlib import Path
import random
import librosa
from torch.utils.data import Dataset
import boto3
import audioread
from smart_open import open
import logging
import sys
import torch.distributed as dist
import json

### Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

### Constant variables
rng = default_rng()
sample_rate=8000 
training_signal_len = 40000


### Dataset helpers
def standardize_length(sample, training_signal_len):
    '''
    Returns `sample` such that the length is equal to `training_signal_len`.
    If the sample is too short, it will repeat the start of the input until enough samples
    have been added.
    '''
    len_diff = training_signal_len - len(sample)
    if len_diff < 0: # The sample is too long
        return sample[:training_signal_len]
    elif len_diff > 0: # The sample is too short
        sample_dup = np.tile(sample, (len_diff // len(sample)) + 1)
        return np.concatenate([sample, sample_dup[:len_diff]])
    else:
        return sample
    
def dynamic_mix_data_prep(voice_dir, noise_dir, training_signal_len, sample_rate):
    """
    This function defines the compute graph for dynamic mixing.
    """

    # Randomly chooses files from voice and noise url lists
    voice_dir_len = 2703
    noise_dir_len = 2000
    clean_file_idx, noise_file_idx = rng.choice(voice_dir_len, size=2, replace=False)
    clean1_file, clean2_file = voice_dir[clean_file_idx], voice_dir[noise_file_idx]
    noise_file = noise_dir[np.random.randint(noise_dir_len)]
    
    sources = []
    first_lvl = None
    spk_files = [clean2_file, noise_file, clean1_file]

    for i, spk_file in enumerate(spk_files):
        with open(spk_file, 'rb') as file:
            y, sr = librosa.load(file, sr=sample_rate, duration=training_signal_len/sample_rate)
            y = standardize_length(y, training_signal_len)

        # Normalize audio
        tmp = librosa.util.normalize(y)

        # Layer on a stack
        if i == 0 or i == 1:
            gain = np.clip(random.normalvariate(-27.43, 2.57), -45, 0)
            tmp *= 10 ** (gain / 20)  # Convert dB to amplitude
            first_lvl = gain
        else:
            gain = np.clip(first_lvl + random.normalvariate(-2.51, 2.66), -45, 0)
            tmp *= 10 ** (gain / 20)  # Convert dB to amplitude
            
        sources.append(tmp)

    # Mix the sources together
    mixture = sum(sources)

    # Calculate the maximum amplitude among the tensors
    max_amp = max(np.abs(mixture).max(), *[np.abs(s).max() for s in sources])

    # Calculate scale value and apply it to the array
    mix_scaling = 1 / max_amp * 0.9
    mixture *= mix_scaling
    
    # Save the clean audio also
    with open(clean1_file, 'rb') as file:
        clean, _ = librosa.load(file, sr=sample_rate, duration=training_signal_len/sample_rate)
        clean = standardize_length(clean, training_signal_len)
        
    clean *= 10 ** (gain / 20)  # Apply the same gain as mixture

    return mixture, clean, sample_rate

def get_clean_and_noise_urls():
    '''
    Useful for interacting with S3 within a Sagemaker notebook instance. 
    This enables us to store only the urls for resources within S3 as 
    opposed to passing around all audio files.
    
    Returns (Clean URLs, Noise URLs)
    '''

    # Initialize the S3 client
    s3_client = boto3.client('s3')

    # Specify the S3 bucket name
    bucket_name = 'amdmic-librispeech'

    # Define the paths in S3
    clean_speech_s3_path = "clean_speech/clean_speech/"
    noise_s3_path = "noise/"

    # Store urls of our dataset
    clean = np.array([])
    noise = np.array([])

    # Load clean speech urls
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=clean_speech_s3_path):
        for content in page.get('Contents', []):
            object_key = content['Key']
            url = f"s3://{bucket_name}/{object_key}"
            clean = np.append(clean, url)

    # Load noise urls
    for page in paginator.paginate(Bucket=bucket_name, Prefix=noise_s3_path):
        for content in page.get('Contents', []):
            object_key = content['Key']
            url = f"s3://{bucket_name}/{object_key}"
            noise = np.append(noise, url)

    return clean, noise

class NoisyAudioDataset(Dataset):
    def __init__(self, transform=None, target_transform=None, num_samples=100):
        self.transform = transform
        self.target_transform = target_transform
        self.num_samples = num_samples
        
        clean_dir, noise_dir = get_clean_and_noise_urls()
        self.voice_dir = clean_dir
        self.noise_dir = noise_dir
        logger.info(f"Initializing NoisyAudioDataset with {len(self.voice_dir)} voice and {len(self.noise_dir)} noise samples")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        mixture, clean, sr = dynamic_mix_data_prep(self.voice_dir, self.noise_dir, training_signal_len, sample_rate)
        std_mix = torch.tensor(standardize_length(mixture, training_signal_len))
        std_clean = torch.tensor(standardize_length(clean, training_signal_len))
        
        return std_mix, std_clean

### Model
class GlobalLayerNorm(nn.Module):
    '''
       Calculate Global Layer Normalization
       dim: (int or list or torch.Size) –
            input shape from an expected input of size
       eps: a value added to the denominator for numerical stability.
       elementwise_affine: a boolean value that when set to True,
           this module has learnable per-element affine parameters
           initialized to ones (for weights) and zeros (for biases).
    '''

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.dim, 1))
            self.bias = nn.Parameter(torch.zeros(self.dim, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # x = N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x L
        # gln: mean,var N x 1 x 1
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__class__.__name__))

        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x-mean)**2, (1, 2), keepdim=True)
        # N x C x L
        if self.elementwise_affine:
            x = self.weight*(x-mean)/torch.sqrt(var+self.eps)+self.bias
        else:
            x = (x-mean)/torch.sqrt(var+self.eps)
        return x


class CumulativeLayerNorm(nn.LayerNorm):
    '''
       Calculate Cumulative Layer Normalization
       dim: you want to norm dim
       elementwise_affine: learnable per-element affine parameters
    '''

    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm, self).__init__(
            dim, elementwise_affine=elementwise_affine)

    def forward(self, x):
        # x: N x C x L
        # N x L x C
        x = torch.transpose(x, 1, 2)
        # N x L x C == only channel norm
        x = super().forward(x)
        # N x C x L
        x = torch.transpose(x, 1, 2)
        return x


def select_norm(norm, dim):
    if norm == 'gln':
        return GlobalLayerNorm(dim, elementwise_affine=True)
    if norm == 'cln':
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    else:
        return nn.BatchNorm1d(dim)


class Conv1D(nn.Conv1d):
    '''
       Applies a 1D convolution over an input signal composed of several input planes.
    '''

    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        # x: N x C x L
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__class__.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x)
        return x


class ConvTrans1D(nn.ConvTranspose1d):
    '''
       This module can be seen as the gradient of Conv1d with respect to its input.
       It is also known as a fractionally-strided convolution
       or a deconvolution (although it is not an actual deconvolution operation).
    '''

    def __init__(self, *args, **kwargs):
        super(ConvTrans1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__class__.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x)
        return x

class Conv1D_Block(nn.Module):
    '''
       Consider only residual links
    '''

    def __init__(self, in_channels=256, out_channels=512,
                 kernel_size=3, dilation=1, norm='gln', causal=False):
        super(Conv1D_Block, self).__init__()
        # conv 1 x 1
        self.conv1x1 = Conv1D(in_channels, out_channels, 1)
        self.PReLU_1 = nn.PReLU()
        self.norm_1 = select_norm(norm, out_channels)
        # not causal don't need to padding, causal need to pad+1 = kernel_size
        self.pad = (dilation * (kernel_size - 1)) // 2 if not causal else (
            dilation * (kernel_size - 1))
        # depthwise convolution
        self.dwconv = Conv1D(out_channels, out_channels, kernel_size,
                             groups=out_channels, padding=self.pad, dilation=dilation)
        self.PReLU_2 = nn.PReLU()
        self.norm_2 = select_norm(norm, out_channels)
        self.Sc_conv = nn.Conv1d(out_channels, in_channels, 1, bias=True)
        self.causal = causal

    def forward(self, x):
        # x: N x C x L
        # N x O_C x L
        c = self.conv1x1(x)
        # N x O_C x L
        c = self.PReLU_1(c)
        c = self.norm_1(c)
        # causal: N x O_C x (L+pad)
        # noncausal: N x O_C x L
        c = self.dwconv(c)
        # N x O_C x L
        if self.causal:
            c = c[:, :, :-self.pad]
        c = self.Sc_conv(c)
        return x+c


class ConvTasNet(nn.Module):
    '''
       ConvTasNet module
       N	Number of ﬁlters in autoencoder
       L	Length of the ﬁlters (in samples)
       B	Number of channels in bottleneck and the residual paths’ 1 × 1-conv blocks
       Sc	Number of channels in skip-connection paths’ 1 × 1-conv blocks
       H	Number of channels in convolutional blocks
       P	Kernel size in convolutional blocks
       X	Number of convolutional blocks in each repeat
       R	Number of repeats
    '''

    def __init__(self,
                 N=512,
                 L=16,
                 B=128,
                 H=512,
                 P=3,
                 X=8,
                 R=3,
                 norm="gln",
                 num_spks=2,
                 activate="relu",
                 causal=False):
        super(ConvTasNet, self).__init__()
        # n x 1 x T => n x N x T
        self.encoder = Conv1D(1, N, L, stride=L // 2, padding=0)
        # n x N x T  Layer Normalization of Separation
        self.LayerN_S = select_norm('cln', N)
        # n x B x T  Conv 1 x 1 of  Separation
        self.BottleN_S = Conv1D(N, B, 1)
        # Separation block
        # n x B x T => n x B x T
        self.separation = self._Sequential_repeat(
            R, X, in_channels=B, out_channels=H, kernel_size=P, norm=norm, causal=causal)
        # n x B x T => n x 2*N x T
        self.gen_masks = Conv1D(B, num_spks*N, 1)
        # n x N x T => n x 1 x L
        self.decoder = ConvTrans1D(N, 1, L, stride=L//2)
        # activation function
        active_f = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'softmax': nn.Softmax(dim=0)
        }
        self.activation_type = activate
        self.activation = active_f[activate]
        self.num_spks = num_spks

    def _Sequential_block(self, num_blocks, **block_kwargs):
        '''
           Sequential 1-D Conv Block
           input:
                 num_block: how many blocks in every repeats
                 **block_kwargs: parameters of Conv1D_Block
        '''
        Conv1D_Block_lists = [Conv1D_Block(
            **block_kwargs, dilation=(2**i)) for i in range(num_blocks)]

        return nn.Sequential(*Conv1D_Block_lists)

    def _Sequential_repeat(self, num_repeats, num_blocks, **block_kwargs):
        '''
           Sequential repeats
           input:
                 num_repeats: Number of repeats
                 num_blocks: Number of block in every repeats
                 **block_kwargs: parameters of Conv1D_Block
        '''
        repeats_lists = [self._Sequential_block(
            num_blocks, **block_kwargs) for i in range(num_repeats)]
        return nn.Sequential(*repeats_lists)

    def forward(self, x):
        if x.dim() >= 3:
            raise RuntimeError(
                "{} accept 1/2D tensor as input, but got {:d}".format(
                    self.__class__.__name__, x.dim()))
        if x.dim() == 1:
            x = torch.unsqueeze(x, 0)
        # x: n x 1 x L => n x N x T
        w = self.encoder(x)
        # n x N x L => n x B x L
        e = self.LayerN_S(w)
        e = self.BottleN_S(e)
        # n x B x L => n x B x L
        e = self.separation(e)
        # n x B x L => n x num_spk*N x L
        m = self.gen_masks(e)
        # n x N x L x num_spks
        m = torch.chunk(m, chunks=self.num_spks, dim=1)
        # num_spks x n x N x L
        m = self.activation(torch.stack(m, dim=0))
        d = [w*m[i] for i in range(self.num_spks)]
        # decoder part num_spks x n x L
        s = [self.decoder(d[i], squeeze=True) for i in range(self.num_spks)]
        return s
    
### Dataloaders

def _get_train_data_loader(num_samples, batch_size, is_distributed, **kwargs):
    """
    Get train dataloader
    """
    logger.info("Get train data loader")
    dataset = NoisyAudioDataset(num_samples)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed else None
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        shuffle=train_sampler is None, 
        **kwargs)

def _get_val_data_loader(num_samples, batch_size, **kwargs):
    """
    Get validation dataloader
    """
    dataset = NoisyAudioDataset(num_samples)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, **kwargs)

def train(args):
    if torch.cuda.device_count() > 0:
        logger.info(f"Let's use {torch.cuda.device_count()} GPUs!!")

    #### initialise distributed training
    is_distributed = len(args.hosts) > 1 and args.backend is not None
    logger.debug("Distributed training - {}".format(is_distributed))

    if is_distributed:
        # Initialize the distributed environment.
        world_size = len(args.hosts)
        os.environ["WORLD_SIZE"] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        os.environ["RANK"] = str(host_rank)
        dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)
        logger.info(
            "Initialized the distributed environment: '{}' backend on {} nodes. ".format(
                args.backend, dist.get_world_size()
            )
            + "Current host rank is {}. Number of gpus: {}".format(dist.get_rank(), args.num_gpus)
        )
        
    use_cuda = args.num_gpus > 0
    logger.debug("Number of gpus available - {}".format(args.num_gpus))
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    # Create the dataset
    train_loader = _get_train_data_loader(args.train_sample_size, args.batch_size, is_distributed, **kwargs)
    test_loader = _get_val_data_loader(args.test_sample_size, args.batch_size, **kwargs)

    # log dataset information
    logger.debug(
        "Processes {}/{} ({:.0f}%) of train data".format(
            len(train_loader.sampler),
            len(train_loader.dataset),
            100.0 * len(train_loader.sampler) / len(train_loader.dataset),
        )
    )

    logger.debug(
        "Processes {}/{} ({:.0f}%) of test data".format(
            len(test_loader.sampler),
            len(test_loader.dataset),
            100.0 * len(test_loader.sampler) / len(test_loader.dataset),
        )
    )

    # Create model
    model = ConvTasNet().to(device)
    if is_distributed:
        model = nn.parallel.DistributedDataParallel(model)
    else:
        model = nn.DataParallel(model)

    criterion = torch.nn.MSELoss()
    # TODO in real run: change to Adam
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        nonfinite_count = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs[0], targets)

            # loss thresholding
            loss = loss[loss > args.threshold]
            if loss.nelement() > 0:
                loss = loss.mean() # average on mini-batch

            # Backward pass, gradient clipping, optimization
            if loss.nelement() > 0 and torch.isfinite(loss):
                loss.backward()
                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
                train_loss += loss.item()
            else:
                nonfinite_count += 1
                logger.debug(f"Non-finite loss, skipping iteration, has happened {nonfinite_count} times so far")
                loss.data = torch.tensor(0.0).to(device)

            if batch_idx % args.log_interval == 0:
                logger.debug(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(inputs),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
            # manage memory
            del inputs, targets, outputs, loss
            torch.cuda.empty_cache() # try to not do this every time because it slows down training
                
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                test_loss += criterion(outputs[0], targets).item()

        test_loss /= len(test_loader)
        logger.debug("\nTest set: Average loss: {:.4f}\n".format(test_loss))

        # checkpoint every save_checkpoint_interval
        if epoch % args.save_checkpoint_interval == 0:
            num = epoch // args.save_checkpoint_interval
            model_path = os.path.join(args.model_dir, f"checkpoint_{num}.pth")
            torch.save(model.cpu().state_dict(), model_path)
            logger.debug("Model saved at {}".format(model_path))

    # save the final model
    model_path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), model_path)
    logger.debug("Final model saved at {}".format(model_path))

    return model_path

def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvTasNet().to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pth")))
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        metavar="N",
        help="input batch size for training (default: 64)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        metavar="N",
        help="number of epochs to train (default: 50)",
    )

    # num_gps, seed, log_interval, threshold, clip_grad_norm, save_checkpoint_interval
    parser.add_argument("--num-gpus", 
                        type=int, 
                        default=os.environ["SM_NUM_GPUS"]
                       )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        metavar="S",
        help="random seed (default: 1)",
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        metavar="N",
        help="how many batches to wait before logging training status",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=-30,
        metavar="N",
        help="threshold value for loss",
    )

    parser.add_argument(
        "--clip-grad-norm",
        type=float,
        default=5.0,
        metavar="N",
        help="value used to clip the gradient norm",
    )

    parser.add_argument(
        "--save-checkpoint-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many epochs to wait before saving model",
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ["SM_MODEL_DIR"],
        metavar="N",
        help="path to save model checkpoints",
    )
    
    parser.add_argument(
        "--train-sample-size",
        type=int,
        default=100,
        metavar="N",
        help="number of samples to train on",
    )

    parser.add_argument(
        "--test-sample-size",
        type=int,
        default=20,
        metavar="N",
        help="number of samples to test on",
    )

    # distributed training arguments
    parser.add_argument(
        "--hosts",
        type=list,
        default=json.loads(os.environ["SM_HOSTS"]),
    )

    parser.add_argument(
        "--current-host",
        type=str,
        default=os.environ["SM_CURRENT_HOST"],
    )

    parser.add_argument(
        "--backend",
        type=str,
        default=None,
    )
    
    train(parser.parse_args())