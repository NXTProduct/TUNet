class CONFIG:
    gpus = "0"  # List of gpu devices

    # Bandwidth extension and masked speech modeling experiment config
    class TASK:
        task = 'bwe'  # Task to execute. Should either be 'msm' or 'bwe'
        assert task in ['msm', 'bwe'], "task should either be 'msm' or 'bwe'"

        mask_chunk = 256  # Size of masked chunks for MSM. Should be a power of two
        mask_ratio = 0.2  # MSM masking ratio in range (0, 1)

        '''
        BWE downsampling method. Should be either 'cheby', 'augment' or resampy supported methods.
            'cheby' uses the Scipy's decimation based on the Chebyshev Type-I lowpass filter.
            'augment' uses the Chebyshev Type-I filters with random orders and ripples.
        '''
        downsampling = 'cheby'

        # resampy supported methods
        resampy = ['kaiser_best', 'kaiser_fast', 'fft', 'polyphase', 'linear', 'zero_order_hold', 'sinc_best',
                   'sinc_medium', 'sinc_fastest', 'soxr_vhq', 'soxr_hq', 'soxr_mq', 'soxr_lq', 'soxr_qq']
        assert downsampling in ['augment', 'cheby'] + resampy, 'Invalid downsampling method'
        orders = range(1, 11)  # the Chebyshev Type-I orders
        ripples = [1e-9, 1e-6, 1e-3, 1, 5]  # the Chebyshev Type-I ripples

    class TRAIN:
        batch_size = 80  # number of audio files per batch
        lr = 3e-4  # learning rate
        epochs = 150  # max training epochs
        workers = 16  # number of dataloader workers
        val_split = 0.1  # validation set proportion
        loss_type = 2  # training loss types. 1: MSE loss, 2: MSE and multi-resolution STFT loss
        assert loss_type in [1, 2], 'Invalid loss_type'
        mse_weight = 10000  # weight of the MSE loss
        clipping_val = 1.0  # gradient clipping value
        patience = 3  # learning rate scheduler's patience
        factor = 0.5  # learning rate reduction factor

    # Model config
    class MODEL:
        tfilm = True  # enable/disable TFiLM layers
        n_blocks = 64  # number of blocks of TFiLM layers.
        bottleneck_type = 'performer'  # bottleneck module. Should either be 'performer', 'lstm' or None
        assert bottleneck_type in ['performer', 'lstm', None], "Invalid bottleneck_type"
        kernel_sizes = [66, 18, 8]  # kernel sizes of each convolution/deconvolution layers
        strides = [4, 4, 4]  # strides of each convolution/deconvolution layers
        out_channels = [64, 128, 256]  # numbers of filters of each convolution/deconvolution layers

        # Performer bottleneck config
        class TRANSFORMER:
            dim_head = 32
            depth = 3
            heads = 2

    # Dataset config
    class DATA:
        dataset = 'vivos'  # dataset to use. Should either be 'vctk' or 'vivos'
        '''
        Dictionary that specifies paths to root directories and train/test text files of each datasets.
        'root' is the path to the dataset and each line of the train.txt/test.txt files should contains the path to an
        audio file from 'root'. 
        '''
        data_dir = {'vctk': {'root': 'data/vctk/wav48',
                             'train': "data/vctk/train.txt",
                             'test': "data/vctk/test.txt"},
                    'vivos': {'root': 'data/vivos',
                              'train': 'data/vivos/train.txt',
                              'test': 'data/vivos/test.txt'}}
        assert dataset in data_dir.keys(), 'Unknown dataset.'
        sr = 16000  # target audio sampling rate
        ratio = 2  # downsampling ratio
        window_size = 8192  # size of the sliding window
        stride = 4096  # stride of the sliding window. Should be divisible to 'mask_chunk' if the task is MSM.

    class LOG:
        log_dir = 'lightning_logs'  # checkpoint and log directory
        sample_path = 'audio_samples'  # path to save generated audio samples in evaluation.

    class TEST:
        in_dir = 'test_samples'  # path to test audio inputs
        out_dir = 'test_samples'  # path to generated outputs
