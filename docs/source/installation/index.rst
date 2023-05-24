.. _install icefall:

Installation
============

.. caution::

   99% users who have issues about the installation are using conda.

.. caution::

   99% users who have issues about the installation are using conda.

.. caution::

   99% users who have issues about the installation are using conda.

.. hint::

   We suggest that you use ``pip install`` to install PyTorch.

   You can use the following command to create a virutal environment in Python:

    .. code-block:: bash

        python3 -m venv ./my_env
        source ./my_env/bin/activate

``icefall`` depends on `k2 <https://github.com/k2-fsa/k2>`_ and
`lhotse <https://github.com/lhotse-speech/lhotse>`_.

We recommend that you use the following steps to install the dependencies.

- (0) Install CUDA toolkit and cuDNN (Required if you want to use CUDA)
- (1) Install PyTorch and torchaudio
- (2) Install k2
- (3) Install lhotse

.. caution::

  Installation order matters.

(0) Install CUDA toolkit and cuDNN
----------------------------------

Please refer to
`<https://k2-fsa.github.io/k2/installation/cuda-cudnn.html>`_
to install CUDA and cuDNN.

(1) Install PyTorch and torchaudio
----------------------------------

Please refer `<https://pytorch.org/>`_ to install PyTorch
and torchaudio.

.. hint::

   You can also go to  `<https://download.pytorch.org/whl/torch_stable.html>`_
   to download pre-compiled wheels and install them.

.. caution::

   Please install torch and torchaudio at the same time.


(2) Install k2
--------------

Please refer to `<https://k2-fsa.github.io/k2/installation/index.html>`_
to install ``k2``.

.. caution::

  Please don't change your installed PyTorch after you have installed k2.

.. note::

   We suggest that you install k2 from source by following
   `<https://k2-fsa.github.io/k2/installation/from_source.html>`_
   or
   `<https://k2-fsa.github.io/k2/installation/for_developers.html>`_.

.. hint::

   Please always install the latest version of k2.

(3) Install lhotse
------------------

Please refer to `<https://lhotse.readthedocs.io/en/latest/getting-started.html#installation>`_
to install ``lhotse``.


.. hint::

    We strongly recommend you to use::

      pip install git+https://github.com/lhotse-speech/lhotse

    to install the latest version of lhotse.

(4) Download icefall
--------------------

``icefall`` is a collection of Python scripts; what you need is to download it
and set the environment variable ``PYTHONPATH`` to point to it.

Assume you want to place ``icefall`` in the folder ``/tmp``. The
following commands show you how to setup ``icefall``:


.. code-block:: bash

  cd /tmp
  git clone https://github.com/k2-fsa/icefall
  cd icefall
  pip install -r requirements.txt
  export PYTHONPATH=/tmp/icefall:$PYTHONPATH

.. HINT::

  You can put several versions of ``icefall`` in the same virtual environment.
  To switch among different versions of ``icefall``, just set ``PYTHONPATH``
  to point to the version you want.


Installation example
--------------------

The following shows an example about setting up the environment.


(1) Create a virtual environment and activate it
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  .. code-block:: bash

  kuangfangjun:fangjun$ python3 -m venv test-icefall
  kuangfangjun:fangjun$ source test-icefall/bin/activate
  (test-icefall) kuangfangjun:fangjun$

(2) Install CUDA toolkit and cuDNN
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before installing CUDA toolkit, we have to decide which version of CUDA to install.

Before deciding which version of CUDA to install, we have to decide which version
of PyTorch to install.

In this example, we will install the following wheel,

  `<https://download.pytorch.org/whl/cu116/torch-1.13.0%2Bcu116-cp38-cp38-linux_x86_64.whl>`_

which means we will use torch 1.13.0, CUDA 11.6, and Python 3.8.

.. hint::

   You can choose a version that is a best fit for you.

So we are going to install CUDA 11.6 in this example, we can follow
`<https://k2-fsa.github.io/k2/installation/cuda-cudnn.html#cuda-11-6>`_
to install it.

.. hint::

   Please also follow the above link to install cuDNN.

.. caution::

   Please ``d o n ' t`` use ``conda install`` to install CUDA toolkit. Otherwise,
   you may be ``S A D`` later.

Before we continue, let us check that we have installed CUDA 11.6 successfully:

.. code-block:: bash

  (test-icefall) kuangfangjun:fangjun$ nvcc --version
  nvcc: NVIDIA (R) Cuda compiler driver
  Copyright (c) 2005-2022 NVIDIA Corporation
  Built on Thu_Feb_10_18:23:41_PST_2022
  Cuda compilation tools, release 11.6, V11.6.112
  Build cuda_11.6.r11.6/compiler.30978841_0

(3) Install torch and torchaudio
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have decided above to install torch 1.13.0 with CUDA 11.6 and Python 3.8,
so we can go to `<https://download.pytorch.org/whl/torch_stable.html>`_
to download the pre-compiled wheels for torch and torchaudio.

From `<https://pytorch.org/audio/main/installation.html#compatibility-matrix>`_,
if we are going to install ``torch 1.13.0``, we have to choose ``torchaudio 0.13.0``.

We can use the following commands to install torch 1.13.0 with CUDA 11.6:

.. code-block:: bash

  (test-icefall) kuangfangjun:fangjun$ wget https://download.pytorch.org/whl/cu116/torch-1.13.0%2Bcu116-cp38-cp38-linux_x86_64.whl
  (test-icefall) kuangfangjun:fangjun$ wget https://download.pytorch.org/whl/cu116/torchaudio-0.13.0%2Bcu116-cp38-cp38-linux_x86_64.whl
  (test-icefall) kuangfangjun:fangjun$ pip install ./torch-1.13.0+cu116-cp38-cp38-linux_x86_64.whl ./torchaudio-0.13.0+cu116-cp38-cp38-linux_x86_64.whl

  Processing ./torch-1.13.0+cu116-cp38-cp38-linux_x86_64.whl
  Processing ./torchaudio-0.13.0+cu116-cp38-cp38-linux_x86_64.whl
  Collecting typing-extensions (from torch==1.13.0+cu116)
    Downloading https://files.pythonhosted.org/packages/85/d2/949d324c348014f0fd2e8e6d8efd3c0adefdcecd28990d4144f2cfc8105e/typing_extensions-4.6.0-py3-none-any.whl
  Installing collected packages: typing-extensions, torch, torchaudio
  Successfully installed torch-1.13.0+cu116 torchaudio-0.13.0+cu116 typing-extensions-4.6.0

Before we continue, let us check that we have installed torch and torchaudio successfully
by using the following command

.. code-block:: bash

  (test-icefall) kuangfangjun:fangjun$ python3 -m torch.utils.collect_env

The output is given below:

.. code-block:: bash

  Collecting environment information...
  PyTorch version: 1.13.0+cu116
  Is debug build: False
  CUDA used to build PyTorch: 11.6
  ROCM used to build PyTorch: N/A

  OS: Ubuntu 18.04.5 LTS (x86_64)
  GCC version: (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0
  Clang version: Could not collect
  CMake version: version 3.21.6
  Libc version: glibc-2.27

  Python version: 3.8.0 (default, Oct 28 2019, 16:14:01)  [GCC 8.3.0] (64-bit runtime)
  Python platform: Linux-5.4.54-2.0.4.std7c.el7.x86_64-x86_64-with-glibc2.27
  Is CUDA available: True
  CUDA runtime version: 11.6.112
  CUDA_MODULE_LOADING set to: LAZY
  GPU models and configuration:
  GPU 0: Tesla V100-PCIE-32GB
  GPU 1: Tesla V100-PCIE-32GB
  GPU 2: Tesla V100-PCIE-32GB
  GPU 3: Tesla V100-PCIE-32GB
  GPU 4: Tesla V100-PCIE-32GB
  GPU 5: Tesla V100-PCIE-32GB
  GPU 6: Tesla V100-PCIE-32GB
  GPU 7: Tesla V100-PCIE-32GB

  Nvidia driver version: 510.47.03
  cuDNN version: /usr/lib/x86_64-linux-gnu/libcudnn.so.7.6.2
  HIP runtime version: N/A
  MIOpen runtime version: N/A
  Is XNNPACK available: True

  Versions of relevant libraries:
  [pip3] torch==1.13.0+cu116
  [pip3] torchaudio==0.13.0+cu116
  [conda] Could not collect

(3) Install k2
~~~~~~~~~~~~~~

The following link:

  `<https://k2-fsa.github.io/k2/installation/index.html>`_

lists several methods to install `k2`_.

In this example, we will install `k2`_ from source by following
`<https://k2-fsa.github.io/k2/installation/from_source.html>`_.

.. code-block:: bash

  (test-icefall) kuangfangjun:fangjun$ git clone https://github.com/k2-fsa/k2.git
  Cloning into 'k2'...
  remote: Enumerating objects: 14565, done.
  remote: Counting objects: 100% (836/836), done.
  remote: Compressing objects: 100% (244/244), done.
  remote: Total 14565 (delta 630), reused 764 (delta 587), pack-reused 13729
  Receiving objects: 100% (14565/14565), 15.52 MiB | 8.00 MiB/s, done.
  Resolving deltas: 100% (10201/10201), done.
  Checking out files: 100% (667/667), done.

  (test-icefall) kuangfangjun:fangjun$ cd k2
  (test-icefall) kuangfangjun:k2$ python3 setup.py install

To check that `k2`_ has been install successfully, please run:

.. code-block:: bash

  (test-icefall) kuangfangjun:k2$ python3 -m k2.version

The output is given below:

.. code-block:: bash

  Collecting environment information...

  k2 version: 1.24.3
  Build type: Release
  Git SHA1: 2fd1aa794f62efc06f27e7d8b886d05189a65b5a
  Git date: Mon May 22 16:30:55 2023
  Cuda used to build k2: 11.6
  cuDNN used to build k2: 8.2.1
  Python version used to build k2: 3.8
  OS used to build k2: Ubuntu 18.04.5 LTS
  CMake version: 3.21.6
  GCC version: 7.5.0
  CMAKE_CUDA_FLAGS:  -Wno-deprecated-gpu-targets   -lineinfo --expt-extended-lambda -use_fast_math -Xptxas=-w  --expt-extended-lambda -gencode arch=compute_70,code=sm_70 -DONNX_NAMESPACE=onnx_c2 -gencode arch=compute_70,code=sm_70 -Xcudafe --diag_suppress=cc_clobber_ignored,--diag_suppress=integer_sign_change,--diag_suppress=useless_using_declaration,--diag_suppress=set_but_not_used,--diag_suppress=field_without_dll_interface,--diag_suppress=base_class_has_different_dll_interface,--diag_suppress=dll_interface_conflict_none_assumed,--diag_suppress=dll_interface_conflict_dllexport_assumed,--diag_suppress=implicit_return_from_non_void_function,--diag_suppress=unsigned_compare_with_zero,--diag_suppress=declared_but_not_referenced,--diag_suppress=bad_friend_decl --expt-relaxed-constexpr --expt-extended-lambda -D_GLIBCXX_USE_CXX11_ABI=0 --compiler-options -Wall  --compiler-options -Wno-strict-overflow  --compiler-options -Wno-unknown-pragmas
  CMAKE_CXX_FLAGS:  -D_GLIBCXX_USE_CXX11_ABI=0 -Wno-unused-variable  -Wno-strict-overflow
  PyTorch version used to build k2: 1.13.0+cu116
  PyTorch is using Cuda: 11.6
  NVTX enabled: True
  With CUDA: True
  Disable debug: True
  Sync kernels : False
  Disable checks: False
  Max cpu memory allocate: 214748364800 bytes (or 200.0 GB)
  k2 abort: False
  __file__: /star-fj/fangjun/test-icefall/lib/python3.8/site-packages/k2-1.24.3.dev20230523+cuda11.6.torch1.13.0-py3.8-linux-x86_64.egg/k2/version/version.py
  _k2.__file__: /star-fj/fangjun/test-icefall/lib/python3.8/site-packages/k2-1.24.3.dev20230523+cuda11.6.torch1.13.0-py3.8-linux-x86_64.egg/_k2.cpython-38-x86_64-linux-gnu.so

(4) Install lhotse
~~~~~~~~~~~~~~~~~~

.. code-block::

  (test-icefall) kuangfangjun:~$ pip install git+https://github.com/lhotse-speech/lhotse

.. caution::

   Make sure you have install ``torch`` and ``torchaudio`` before you install `lhotse`_.
   Otherwise, you will be ``S A D`` later.

If you get the following error while installing `lhotse`_:

.. code-block:: bash

   RuntimeError: Cython required to build dev version of cytoolz.

please run the following command to fix it:

.. code-block:: bash

   pip install cython

If you get the following error while installing `lhotse`_:

.. code-block:: bash

  error: invalid command 'bdist_wheel'

please run the following command to fix it:

.. code-block:: bash

   pip install wheel

Check that you have installed `lhotse`_ successfully:

.. code-block:: bash

  (test-icefall) kuangfangjun:~$ python3 -c "import lhotse; print(lhotse.__version__)"
  1.15.0.dev+git.ed8620d7.clean

(5) Download icefall
~~~~~~~~~~~~~~~~~~~~

In the following, we will install `icefall`_ to the directory ``/tmp``.
You can select any other directory.

.. code-block::

(test-icefall) kuangfangjun:~$ cd /tmp
(test-icefall) kuangfangjun:tmp$ git clone https://github.com/k2-fsa/icefall
(test-icefall) kuangfangjun:tmp$ cd icefall
(test-icefall) kuangfangjun:icefall$ pip install -r requirements.txt

  $ cd /tmp
  $ git clone https://github.com/k2-fsa/icefall
  $ cd icefall
  $ pip install -r requirements.txt


Test Your Installation
----------------------

To test that your installation is successful, let us run
the `yesno recipe <https://github.com/k2-fsa/icefall/tree/master/egs/yesno/ASR>`_
on CPU.

Data preparation
~~~~~~~~~~~~~~~~

.. code-block:: bash
(test-icefall) kuangfangjun:icefall$ export PYTHONPATH=/tmp/icefall:$PYTHONPATH
(test-icefall) kuangfangjun:icefall$ cd /tmp/icefall/egs/yesno/ASR
(test-icefall) kuangfangjun:ASR$ ./prepare.sh

The log of running ``./prepare.sh`` is:

.. code-block::

   2023-05-12 17:55:21 (prepare.sh:27:main) dl_dir: /tmp/icefall/egs/yesno/ASR/download
   2023-05-12 17:55:21 (prepare.sh:30:main) Stage 0: Download data
   /tmp/icefall/egs/yesno/ASR/download/waves_yesno.tar.gz: 100%|_______________________________________________________________| 4.70M/4.70M [06:54<00:00, 11.4kB/s]
   2023-05-12 18:02:19 (prepare.sh:39:main) Stage 1: Prepare yesno manifest
   2023-05-12 18:02:21 (prepare.sh:45:main) Stage 2: Compute fbank for yesno
   2023-05-12 18:02:23,199 INFO [compute_fbank_yesno.py:65] Processing train
   Extracting and storing features: 100%|_______________________________________________________________| 90/90 [00:00<00:00, 212.60it/s]
   2023-05-12 18:02:23,640 INFO [compute_fbank_yesno.py:65] Processing test
   Extracting and storing features: 100%|_______________________________________________________________| 30/30 [00:00<00:00, 304.53it/s]
   2023-05-12 18:02:24 (prepare.sh:51:main) Stage 3: Prepare lang
   2023-05-12 18:02:26 (prepare.sh:66:main) Stage 4: Prepare G
   /project/kaldilm/csrc/arpa_file_parser.cc:void kaldilm::ArpaFileParser::Read(std::istream&):79
   [I] Reading \data\ section.
   /project/kaldilm/csrc/arpa_file_parser.cc:void kaldilm::ArpaFileParser::Read(std::istream&):140
   [I] Reading \1-grams: section.
   2023-05-12 18:02:26 (prepare.sh:92:main) Stage 5: Compile HLG
   2023-05-12 18:02:28,581 INFO [compile_hlg.py:124] Processing data/lang_phone
   2023-05-12 18:02:28,582 INFO [lexicon.py:171] Converting L.pt to Linv.pt
   2023-05-12 18:02:28,609 INFO [compile_hlg.py:48] Building ctc_topo. max_token_id: 3
   2023-05-12 18:02:28,610 INFO [compile_hlg.py:52] Loading G.fst.txt
   2023-05-12 18:02:28,611 INFO [compile_hlg.py:62] Intersecting L and G
   2023-05-12 18:02:28,613 INFO [compile_hlg.py:64] LG shape: (4, None)
   2023-05-12 18:02:28,613 INFO [compile_hlg.py:66] Connecting LG
   2023-05-12 18:02:28,614 INFO [compile_hlg.py:68] LG shape after k2.connect: (4, None)
   2023-05-12 18:02:28,614 INFO [compile_hlg.py:70] <class 'torch.Tensor'>
   2023-05-12 18:02:28,614 INFO [compile_hlg.py:71] Determinizing LG
   2023-05-12 18:02:28,615 INFO [compile_hlg.py:74] <class '_k2.ragged.RaggedTensor'>
   2023-05-12 18:02:28,615 INFO [compile_hlg.py:76] Connecting LG after k2.determinize
   2023-05-12 18:02:28,615 INFO [compile_hlg.py:79] Removing disambiguation symbols on LG
   2023-05-12 18:02:28,616 INFO [compile_hlg.py:91] LG shape after k2.remove_epsilon: (6, None)
   2023-05-12 18:02:28,617 INFO [compile_hlg.py:96] Arc sorting LG
   2023-05-12 18:02:28,617 INFO [compile_hlg.py:99] Composing H and LG
   2023-05-12 18:02:28,619 INFO [compile_hlg.py:106] Connecting LG
   2023-05-12 18:02:28,619 INFO [compile_hlg.py:109] Arc sorting LG
   2023-05-12 18:02:28,619 INFO [compile_hlg.py:111] HLG.shape: (8, None)
   2023-05-12 18:02:28,619 INFO [compile_hlg.py:127] Saving HLG.pt to data/lang_phone


Training
~~~~~~~~

Now let us run the training part:

.. code-block::

  $ export CUDA_VISIBLE_DEVICES=""
  $ ./tdnn/train.py

.. CAUTION::

  We use ``export CUDA_VISIBLE_DEVICES=""`` so that ``icefall`` uses CPU
  even if there are GPUs available.

.. hint::

   In case you get a ``Segmentation fault (core dump)`` error, please use:

      .. code-block:: bash

        export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

   See more at `<https://github.com/k2-fsa/icefall/issues/674>` if you are
   interested.

The training log is given below:

.. code-block::

   2023-05-12 18:04:59,759 INFO [train.py:481] Training started
   2023-05-12 18:04:59,759 INFO [train.py:482] {'exp_dir': PosixPath('tdnn/exp'), 'lang_dir': PosixPath('data/lang_phone'), 'lr': 0.01, 'feature_dim': 23, 'weight_decay': 1e-06, 'start_epoch': 0, 
   'best_train_loss': inf, 'best_valid_loss': inf, 'best_train_epoch': -1, 'best_valid_epoch': -1, 'batch_idx_train': 0, 'log_interval': 10, 'reset_interval': 20, 'valid_interval': 10, 'beam_size': 10, 
   'reduction': 'sum', 'use_double_scores': True, 'world_size': 1, 'master_port': 12354, 'tensorboard': True, 'num_epochs': 15, 'seed': 42, 'feature_dir': PosixPath('data/fbank'), 'max_duration': 30.0,
   'bucketing_sampler': False, 'num_buckets': 10, 'concatenate_cuts': False, 'duration_factor': 1.0, 'gap': 1.0, 'on_the_fly_feats': False, 'shuffle': False, 'return_cuts': True, 'num_workers': 2, 
   'env_info': {'k2-version': '1.24.3', 'k2-build-type': 'Release', 'k2-with-cuda': True, 'k2-git-sha1': '3b7f09fa35e72589914f67089c0da9f196a92ca4', 'k2-git-date': 'Mon May 8 22:58:45 2023', 
   'lhotse-version': '1.15.0.dev+git.6fcfced.clean', 'torch-version': '2.0.0+cu118', 'torch-cuda-available': False, 'torch-cuda-version': '11.8', 'python-version': '3.1', 'icefall-git-branch': 'master', 
   'icefall-git-sha1': '30bde4b-clean', 'icefall-git-date': 'Thu May 11 17:37:47 2023', 'icefall-path': '/tmp/icefall', 
   'k2-path': 'tmp/lib/python3.10/site-packages/k2-1.24.3.dev20230512+cuda11.8.torch2.0.0-py3.10-linux-x86_64.egg/k2/__init__.py', 
   'lhotse-path': 'tmp/lib/python3.10/site-packages/lhotse/__init__.py', 'hostname': 'host', 'IP address': '0.0.0.0'}}
   2023-05-12 18:04:59,761 INFO [lexicon.py:168] Loading pre-compiled data/lang_phone/Linv.pt
   2023-05-12 18:04:59,764 INFO [train.py:495] device: cpu
   2023-05-12 18:04:59,791 INFO [asr_datamodule.py:146] About to get train cuts
   2023-05-12 18:04:59,791 INFO [asr_datamodule.py:244] About to get train cuts
   2023-05-12 18:04:59,852 INFO [asr_datamodule.py:149] About to create train dataset
   2023-05-12 18:04:59,852 INFO [asr_datamodule.py:199] Using SingleCutSampler.
   2023-05-12 18:04:59,852 INFO [asr_datamodule.py:205] About to create train dataloader
   2023-05-12 18:04:59,853 INFO [asr_datamodule.py:218] About to get test cuts
   2023-05-12 18:04:59,853 INFO [asr_datamodule.py:252] About to get test cuts
   2023-05-12 18:04:59,986 INFO [train.py:422] Epoch 0, batch 0, loss[loss=1.065, over 2436.00 frames. ], tot_loss[loss=1.065, over 2436.00 frames. ], batch size: 4
   2023-05-12 18:05:00,352 INFO [train.py:422] Epoch 0, batch 10, loss[loss=0.4561, over 2828.00 frames. ], tot_loss[loss=0.7076, over 22192.90 frames. ], batch size: 4
   2023-05-12 18:05:00,691 INFO [train.py:444] Epoch 0, validation loss=0.9002, over 18067.00 frames.
   2023-05-12 18:05:00,996 INFO [train.py:422] Epoch 0, batch 20, loss[loss=0.2555, over 2695.00 frames. ], tot_loss[loss=0.484, over 34971.47 frames. ], batch size: 5
   2023-05-12 18:05:01,217 INFO [train.py:444] Epoch 0, validation loss=0.4688, over 18067.00 frames.
   2023-05-12 18:05:01,251 INFO [checkpoint.py:75] Saving checkpoint to tdnn/exp/epoch-0.pt
   2023-05-12 18:05:01,389 INFO [train.py:422] Epoch 1, batch 0, loss[loss=0.2532, over 2436.00 frames. ], tot_loss[loss=0.2532, over 2436.00 frames. ], batch size: 4
   2023-05-12 18:05:01,637 INFO [train.py:422] Epoch 1, batch 10, loss[loss=0.1139, over 2828.00 frames. ], tot_loss[loss=0.1592, over 22192.90 frames. ], batch size: 4
   2023-05-12 18:05:01,859 INFO [train.py:444] Epoch 1, validation loss=0.1629, over 18067.00 frames.
   2023-05-12 18:05:02,094 INFO [train.py:422] Epoch 1, batch 20, loss[loss=0.0767, over 2695.00 frames. ], tot_loss[loss=0.118, over 34971.47 frames. ], batch size: 5
   2023-05-12 18:05:02,350 INFO [train.py:444] Epoch 1, validation loss=0.06778, over 18067.00 frames.
   2023-05-12 18:05:02,395 INFO [checkpoint.py:75] Saving checkpoint to tdnn/exp/epoch-1.pt

  ... ...

   2023-05-12 18:05:14,789 INFO [train.py:422] Epoch 13, batch 0, loss[loss=0.01056, over 2436.00 frames. ], tot_loss[loss=0.01056, over 2436.00 frames. ], batch size: 4
   2023-05-12 18:05:15,016 INFO [train.py:422] Epoch 13, batch 10, loss[loss=0.009022, over 2828.00 frames. ], tot_loss[loss=0.009985, over 22192.90 frames. ], batch size: 4
   2023-05-12 18:05:15,271 INFO [train.py:444] Epoch 13, validation loss=0.01088, over 18067.00 frames.
   2023-05-12 18:05:15,497 INFO [train.py:422] Epoch 13, batch 20, loss[loss=0.01174, over 2695.00 frames. ], tot_loss[loss=0.01077, over 34971.47 frames. ], batch size: 5
   2023-05-12 18:05:15,747 INFO [train.py:444] Epoch 13, validation loss=0.01087, over 18067.00 frames.
   2023-05-12 18:05:15,783 INFO [checkpoint.py:75] Saving checkpoint to tdnn/exp/epoch-13.pt
   2023-05-12 18:05:15,921 INFO [train.py:422] Epoch 14, batch 0, loss[loss=0.01045, over 2436.00 frames. ], tot_loss[loss=0.01045, over 2436.00 frames. ], batch size: 4
   2023-05-12 18:05:16,146 INFO [train.py:422] Epoch 14, batch 10, loss[loss=0.008957, over 2828.00 frames. ], tot_loss[loss=0.009903, over 22192.90 frames. ], batch size: 4
   2023-05-12 18:05:16,374 INFO [train.py:444] Epoch 14, validation loss=0.01092, over 18067.00 frames.
   2023-05-12 18:05:16,598 INFO [train.py:422] Epoch 14, batch 20, loss[loss=0.01169, over 2695.00 frames. ], tot_loss[loss=0.01065, over 34971.47 frames. ], batch size: 5
   2023-05-12 18:05:16,824 INFO [train.py:444] Epoch 14, validation loss=0.01077, over 18067.00 frames.
   2023-05-12 18:05:16,862 INFO [checkpoint.py:75] Saving checkpoint to tdnn/exp/epoch-14.pt
   2023-05-12 18:05:16,865 INFO [train.py:555] Done!

Decoding
~~~~~~~~

Let us use the trained model to decode the test set:

.. code-block::

  $ ./tdnn/decode.py

The decoding log is:

.. code-block::

   2023-05-12 18:08:30,482 INFO [decode.py:263] Decoding started
   2023-05-12 18:08:30,483 INFO [decode.py:264] {'exp_dir': PosixPath('tdnn/exp'), 'lang_dir': PosixPath('data/lang_phone'), 'lm_dir': PosixPath('data/lm'), 'feature_dim': 23, 
   'search_beam': 20, 'output_beam': 8, 'min_active_states': 30, 'max_active_states': 10000, 'use_double_scores': True, 'epoch': 14, 'avg': 2, 'export': False, 'feature_dir': PosixPath('data/fbank'), 
   'max_duration': 30.0, 'bucketing_sampler': False, 'num_buckets': 10, 'concatenate_cuts': False, 'duration_factor': 1.0, 'gap': 1.0, 'on_the_fly_feats': False, 'shuffle': False, 'return_cuts': True, 
   'num_workers': 2, 'env_info': {'k2-version': '1.24.3', 'k2-build-type': 'Release', 'k2-with-cuda': True, 'k2-git-sha1': '3b7f09fa35e72589914f67089c0da9f196a92ca4', 'k2-git-date': 'Mon May 8 22:58:45 2023', 
   'lhotse-version': '1.15.0.dev+git.6fcfced.clean', 'torch-version': '2.0.0+cu118', 'torch-cuda-available': False, 'torch-cuda-version': '11.8', 'python-version': '3.1', 'icefall-git-branch': 'master', 
   'icefall-git-sha1': '30bde4b-clean', 'icefall-git-date': 'Thu May 11 17:37:47 2023', 'icefall-path': '/tmp/icefall', 
   'k2-path': '/tmp/lib/python3.10/site-packages/k2-1.24.3.dev20230512+cuda11.8.torch2.0.0-py3.10-linux-x86_64.egg/k2/__init__.py', 
   'lhotse-path': '/tmp/lib/python3.10/site-packages/lhotse/__init__.py', 'hostname': 'host', 'IP address': '0.0.0.0'}}
   2023-05-12 18:08:30,483 INFO [lexicon.py:168] Loading pre-compiled data/lang_phone/Linv.pt
   2023-05-12 18:08:30,487 INFO [decode.py:273] device: cpu
   2023-05-12 18:08:30,513 INFO [decode.py:291] averaging ['tdnn/exp/epoch-13.pt', 'tdnn/exp/epoch-14.pt']
   2023-05-12 18:08:30,521 INFO [asr_datamodule.py:218] About to get test cuts
   2023-05-12 18:08:30,521 INFO [asr_datamodule.py:252] About to get test cuts
   2023-05-12 18:08:30,675 INFO [decode.py:204] batch 0/?, cuts processed until now is 4
   2023-05-12 18:08:30,923 INFO [decode.py:241] The transcripts are stored in tdnn/exp/recogs-test_set.txt
   2023-05-12 18:08:30,924 INFO [utils.py:558] [test_set] %WER 0.42% [1 / 240, 0 ins, 1 del, 0 sub ]
   2023-05-12 18:08:30,925 INFO [decode.py:249] Wrote detailed error stats to tdnn/exp/errs-test_set.txt
   2023-05-12 18:08:30,925 INFO [decode.py:316] Done!

**Congratulations!** You have successfully setup the environment and have run the first recipe in ``icefall``.

Have fun with ``icefall``!

YouTube Video
-------------

We provide the following YouTube video showing how to install ``icefall``.
It also shows how to debug various problems that you may encounter while
using ``icefall``.

.. note::

   To get the latest news of `next-gen Kaldi <https://github.com/k2-fsa>`_, please subscribe
   the following YouTube channel by `Nadira Povey <https://www.youtube.com/channel/UC_VaumpkmINz1pNkFXAN9mw>`_:

      `<https://www.youtube.com/channel/UC_VaumpkmINz1pNkFXAN9mw>`_

..  youtube:: LVmrBD0tLfE
