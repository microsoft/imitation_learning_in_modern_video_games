# Imitation Learning in Modern Video Games
The repository supports training and evaluating behaviour cloning (BC) agents in Minecraft (via the MineRL environment) and Counter Strike: Global Offensive (CS:GO) via the open-source [interface and dataset](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning). The repository accompanies the [Visual Encoders for Data-Efficient Imitation Learning in Modern Video Games](https://arxiv.org/abs/2312.02312) publication.

> [!NOTE]  
> The data pipeline used within this repository has been modified for the open-source release of this codebase. Therefore, training runs using this codebase will not perfectly reproduce the results reported in the paper but we verified that performance of trained agents is comparable.

# Citation
If you use this code in your research, please cite the following paper:
*Schäfer, Lukas, Logan Jones, Anssi Kanervisto, Yuhan Cao, Tabish Rashid, Raluca Georgescu, Dave Bignell, Siddhartha Sen, Andrea Treviño Gavito, and Sam Devlin. "Visual encoders for data-efficient imitation learning in modern video games." Adaptive and Learning Agents Workshop at AAMAS (2025).*

In BibTeX format:
```bibtex
@inproceedings{schafer2025visual,
  title={Visual encoders for data-efficient imitation learning in modern video games},
  author={Sch{\"a}fer, Lukas and Jones, Logan and Kanervisto, Anssi and Cao, Yuhan and Rashid, Tabish and Georgescu, Raluca and Bignell, Dave and Sen, Siddhartha and Gavito, Andrea Trevi{\~n}o and Devlin, Sam},
  booktitle={Adaptive and Learning Agents Workshop, AAMAS},
  year={2025}
}
```

# Installation
We suggest to install the package in a [conda environment](https://www.anaconda.com/). To create a conda environment with the necessary dependencies, run
```bash
conda env create -f environment.yml
```
which will create a conda environment named `bc_from_pixels`. After creating the environment, activate it and install the `pixelbc` package using:
```bash
conda activate bc_from_pixels
pip install -e .
```

## Setup Minecraft MineRL Environment
To run Minecraft, install the required Java/Javac versions:
```bash
sudo add-apt-repository ppa:openjdk-r/ppa
sudo apt-get update
sudo apt-get install openjdk-8-jdk

# Verify installation
java -version # this should output "1.8.X_XXX"
# If you are still seeing a wrong Java version, you may use
# the following line to update it
# sudo update-alternatives --config java
```

Then, install MineRL from the internal repository:
```bash
pip install git+https://github.com/minerllabs/minerl
```
and install xvfb (for virtual buffer):
```bash
sudo apt install xvfb
```
If you get errors with installing MineRL: Make sure you have right JDK version installed: `java -version` and `javac -version` should report `1.8...`. See [MineRL installation docs for more info](https://minerl.readthedocs.io/en/latest/tutorials/index.html).


## Setup CS:GO for Rollouts
To run rollouts in CS:GO, you need to install the game. Since the release of Counter Strike 2, CS:GO has to be installed with its legacy version through Steam under Counter Strike 2. Within [Steam](https://store.steampowered.com/), install the game "Counter-Strike 2". Then, in your library right-click Counter-Strike 2 and select "Betas" and select the "csgo_legacy" version of the game. After the download, you can launch the CS:GO game through Counter-Strike 2 in Steam.

The evaluation is done in a custom aim assist map which can be installed within the Steam workshop [here](https://steamcommunity.com/sharedfiles/filedetails/?id=368026786).

To evaluate agents under settings / conditions consistent with the training data provided within the [CS:GO benchmark repo](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning), make sure to set all settings listed [here](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning?tab=readme-ov-file#troubleshooting) by launching the game, open the settings menu and setting the listed options as stated.

Finally, to allow the rollout script to extract and store meta-information from the goal (such as kills), copy the `gamestate_integration_bcfrompixels.cfg` file provided under `pixelbc/online_rollout/csgo_utils/` in this repo to the `cfg` folder of Steam. This can typically be found under `"D:\Steam\steamapps\common\Counter-Strike Global Offensive\csgo\cfg".` or a similar folder. For more details on the game state integration (GSI) used with this configuration file, see [this developer documentation](https://developer.valvesoftware.com/wiki/Counter-Strike:_Global_Offensive_Game_State_Integration).

# Download Open-Source Datasets

## Minecraft MineRL
To train Minecraft MineRL agents, we use the publicly available VPT contractor data from OpenAI. More details on the data can be found in the [VPT repository](https://github.com/openai/Video-Pre-Training). To download the subset of the data used for this project, run the following command:
```bash
bash pixelbc/scripts/minerl_download_data.sh <PATH>
```
where `<PATH>` is the path to a data split file (e.g. `pixelbc/data/paper_data_split/minerl_6.13_treechop_train_files.txt`). The script will download the video and action data to the `minerl_data` folder.

The MineRL dataset used for this project uses a filtered subset of the original VPT dataset ensuring that each trajectory used for training does chop a tree successfully in a specified timeframe. To filter the trajectories, we use the `pixelbc/scripts/minerl_filter_valid_treechop_trajectories.py` script. The script does not take any arguments.

## Counter-Strike: Global Offensive
To download the training data for CS:GO, we refer to the instructions provided in the [public repository](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning#datasets).


# Training Agents
In this section, we will explain how to train agents using the code in this repository. In all the following, we assume that you have installed the package and dependencies as described above. The fundamental approach to train agents in this project is imitation learning (more specifically behaviour cloning). This approach trains an agent to imitate the behaviour exhibited in a provided dataset of (ideally high-quality) gameplay demonstrations. In the following, we assume that a dataset of recorded gameplay is available to train from.

For this codebase, we assume that the dataset has been downloaded and is available within the filesystem. The configuration will need to specify the path to the dataset as follows:
```yaml
data:
  data_path: <PATH/TO/DATASET>
```
For an example, see the `pixelbc/configs/minerl/default.yaml` configuration file.

The additional configuration parameters are specified to determine the data processing:
```yaml
data:
  game: "csgo"  # type of game, can be either "minerl" or "csgo"
  data_path: <PATH/TO/DATASET>
  image_height: 128  # size to resize images to during pre-processing
  image_width: 128
  framestacking: 1  # number of frames to stack (default: 1 --> no frame stacking)
  num_actions: 15  # number of total actions in the action space
  num_joystick_actions: 4  # number of (continuous) joystick actions in the action space
  num_trigger_actions: 1  # number of trigger actions in the action space
  num_button_actions: 10  # number of button actions in the action space
  discretise_joystick: False  # whether to discretise joystick actions
  discretise_joystick_bins:  # bin boundaries for discretising joystick actions if discretise_joystick is True
    - -1.0
    - -0.845
    - -0.575
    - -0.325
    - 0.0
    - 0.325
    - 0.575
    - 0.845
    - 1.0
  batch_size: 32  # batches of sequences to train on
  sequence_length: 100  # length of sequences to train on
  downsample: 2  # downsampling factor for sequences (e.g. 2 means every second frame is used)
  prefetch_factor: 5  # number of batches to prefetch
  train_num_workers: 20  # number of workers to load training data
  other_num_workers: 1  # number of workers to load validation and/ or testing data
  decord_num_workers: 3  # number of workers to load video data with
```

## Data Split Files
To specify training, validation, and test data, we use data split files. These are simple .txt files with each line containing the name of a file containing the (action) data within a particular dataset. See the `pixelbc/data/paper_data_split` directory for examples of data split files used in the paper.

## Minecraft MineRL: Filter Trajectories for Dataset
The MineRL dataset used for this project uses a filtered subset of the original VPT dataset ensuring that each trajectory used for training does chop a tree successfully in a specified timeframe. To filter the trajectories, we use the `pixelbc/scripts/minerl_filter_valid_treechop_trajectories.py` script. The script does not take any arguments. To modify the path towards the data to filter or where to store the resulting split file, modify the `GLOB_PATTERN` and `OUTPUT_FILE` constants within the script.

## Train a Behaviour Cloning Agent
To start a training run, use the following command:
```bash
python pixelbc/run_train.py <PATH/TO/CONFIG.YAML>
```
where the config provides the respective parameters for pytorch lightning, the data pipeline and model architecture. Note that the config has to specify for which game training is run so the correct processing is applied for the corresponding datasets. This can be done by setting the `data.game` parameter in the config to either `minerl`, or `csgo`.

## Training Parameters
We use [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) to train models with behaviour cloning. This framework abstracts away several details of the training process including the use of modern accelerators (GPUs, TPUs, etc.) and distributed training on hardware with several accelerators available.

First of all, any parameter found in the configuration files can also be overwritten as an argument directly provided to the `run_tran.py` script. E.g. to start a training run from the default configuration but use a different random seed, we can call:
```bash
python pixelbc/run_train.py <pixelbc/configs/minerl/default.yaml> seed=123 data.batch_size=64
```

In this section, we will discuss some hyperparameters which can be specified in training configurations files beyond the ones already discussed above related to the data loading. 

### Resuming Training
To resume a previously started but not finished training run, specify the `resume_from: <PATH>` argument in the training configuration file. This will load the model and optimizer state from the specified checkpoint and continue training from there. Note that the model architecture and the training parameters (e.g. learning rate) need to be the same as in the previous training run. 

If you do not wish to continue training exactly from the same state (including already done progress of training), but only want to load an exciting model checkpoint and start a new training run from there, specify the `checkpoint: <PATH>` argument in the training configuration file. This will load the model state from the specified checkpoint and start a new training run from there. Note that the model architecture and the training parameters (e.g. learning rate) need to be the same as in the previous training run.

### Storing Checkpoints
At regular intervals throughout training, we store checkpoints containing the current agent model and hyperparameters used. To specify how frequent we store such checkpoints, we can use the following parameters
```yaml
log:
  checkpoint_epoch_freq: <INT>
  checkpoint_top_k_validation: <INT>
  checkpoint_last: <BOOL>
```
with the following meanings:
- `checkpoint_epoch_freq`: Frequency of storing checkpoints in epochs. Default: `100`, i.e. every 100 training epochs one checkpoint is stored.
- `checkpoint_top_k_validation:` Number of checkpoints to store based on the validation loss. Default: `1`, i.e. the top 1 checkpoint based on the validation loss is stored.
- `checkpoint_last`: Whether to store the last checkpoint. Default: `True`, i.e. the last checkpoint is always stored.

### Logging Training Metrics
During training, we store several metrics to monitor the training progress. To specify which metrics are stored, we use the following parameters in the configuration files
```yaml
log:
  minimal_metrics: <BOOL>
  device_stats_monitor: <BOOL>
  weights: <BOOL>
  weights_step_freq: <INT>
  weights_log_scalars: <BOOL>
```
with the following meanings:
- `minimal_metrics`: Whether to only log the minimal metrics. See `pixelbc/utils/loggers.py` to modify which metrics are included in this minimal set. This serves to reduce the size of stored tensorboard files.
- `device_stats_monitor`: Whether to log statistics on the device training is run on, including memory usage and CPU/ GPU usage.
- `weights`: Whether to log summary information on the weights of the model. This is unrelated to storing model checkpoints and only refers to inclusion of statistics in the training metric tensorboard!
- `weights_step_freq`: Frequency of logging weights statistics (see above) in terms of number of updates.
- `weights_log_scalars`: Whether to log scalar statistics on the weights of the model.

### Training Hyperparameters
The following parameters in the training configs can be used to specify the hyperparameters of the training process
```yaml
trainer:
  profiler: simple
  max_epochs: 1000
  gradient_clip_algorithm: norm
  gradient_clip_val: 1.0  # set to None to turn off gradient clipping
  precision: 16-mixed
```
with the following meanings:
- `profiler`: Type of time profiler to use and report data from at the end of training. Typically set to `simple` to report a simple string table listing profiling information on each function. See [lightning profiler information](https://lightning.ai/docs/pytorch/stable/tuning/profiler_basic.html) for more details.
- `max_epochs`: Number of epochs of training to complete.
- `gradient_clip_algorithm`: Name of type of gradient clipping to apply during training (if any). Typically set to `norm` to clip gradients with gradient norms.
- `gradient_clip_val`: Value of gradient clipping applied. By default set to `1.0`. Set to `None`/ `null` to use no gradient clipping.
- `precision`: Precision to use during training for the model. By default set to `16-mixed` and overwritten to full precision `32` whenever no accelerator with mixed-precision support is available. Mixed precision is recommended when a supported accelerator is available to reduce VRAM usage and speed-up training.

# Rollout a Trained Agent in the Game
Once we have trained agents, we can rollout these agents in the game. During training, we save checkpoints of models at regular intervals. In this section, we will explain how to load and run these models in the game.

### Minecraft MineRL Rollouts
For Minecraft MineRL, no game has to be launched before calling the rollout script since the minerl environment handles communication with the game. To launch rollouts in Minecraft MineRL, use the `pixelbc/scripts/run_minerl_rollouts.py` script. To run this script, use the following commands:
```bash
cd pixelbc
bash scripts/run_minerl_rollouts.py <PATH/TO/TRAINING_LOGS/SOME_MODEL/MODELS> ... --output_dir <PATH/TO/OUTPUT/DIR>
```
with arguments being a list of paths to logging directories containing `last.ckpt` files as the last checkpoint of training runs.

Note: Rollouts with this script are only supported in UNIX environments (Linux, MacOS).

### Counter-Strike: Global Offensive Rollouts
For CS:GO, we need to start the game. We refer to instructions as documented in the [open-source repository](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning). After, the regular rollout script with the corresponding `--game csgo` argument can be used to launch rollouts in CS:GO.

Note: Rollouts with this script are only supported in Windows.

To run a single rollout of a trained agent in the game, start the CS:GO game and launch the installed custom aim assist map via the console (active console in game settings, open console by pressing `~` key and execute the command `map aimtraindriving_v3 custom`). Then, run the following in a Windows Powershell prompt:
```powershell
python pixelbc\\online_rollout\\rollout.py --game csgo --checkpoint <PATH> --fps <FPS> (<OPTIONAL-ARGUMENTS>)
```
with the following argument options:
- `--checkpoint <PATH>`/ `-ckpt <PATH`: Path to a model checkpoint (as stored by training). This is always required.
- `--fps <INT>`/ `-fps <INT`: FPS at which the model will be prompted for actions (use the same setting as used during training, considering the FPS of the original data and any potentially applied downsampling). By default, you should use `16` for CS:GO evaluation or any integer factor lower (e.g. `4` or `8`) in which case the game will automatically be ran at a slower speed to compensate for the lower fps queries to the model.
- `--game <STR>`/ `-g <STR>`: Name of the game to rollout in. Can be set to either `minerl`, or `csgo`. Default: `minerl`
- `--path <PATH`/ `-p <PATH`: Path to a directory to store evaluation results in. Default: `evaluation`
- `--joystick_action_mode <STR>`/ `-ja <STR`: Specify the mode of action selection for joystick actions. Can be set to either `deterministic` or `stochastic` to take the greedy joystick action or sample from the learned distribution. This is only relevant for models trained with discretised joystick! Default: `deterministic`.
- `--trigger_action_mode <STR>`/ `-ta <STR`: Specify the mode of action selection for trigger actions. Can be set to either `deterministic` or `stochastic`. Default: `stochastic`.
- `--button_action_mode <STR>`/ `-ba <STR`: Specify the mode of action selection for button actions. Can be set to either `deterministic` or `stochastic`. Default: `stochastic`.
- `--ignore_keyboard_inputs`: If provided, ignore keyboard inputs during the rollout which otherwise can be use stop/ reset the rollout (see below). This can be useful when rolling out on a machine while still typing for other purposes on the same machine.

The rollout will start and the agent will start to play the game. The rollout terminates, when the time limit of 5 minutes is reached and by default 3 rollouts will be executed in sequence.


# Pre-Compute Embeddings for Pre-Trained Visual Encoders
Pre-trained visual encoders are frozen during BC training and, thus, are not further trained. This means, their embeddings are fixed and can be pre-computed for all data points in the dataset. Pre-computing these embeddings can be desirable whenever the dataset is comparably small, so pre-computing embeddings is feasible. Once embeddings are computed in this manner, they can be loaded during training and, thus, do not need to be computed on-the-fly during training. This can significantly speed-up training, in particular for larger pre-trained models which are comparably expensive to compute embeddings with. To pre-compute embeddings for a dataset, run
```bash
python pixelbc/scripts/generate_embeddings.py --game <GAME> --encoder_config <PATH> --filelist <PATH> ... --data_base_path <PATH/TO/DATASET> --output_dir <PATH/TO/OUTPUT_DIR> (<OPTIONAL-ARGUMENTS>)
```
with the following arguments:
- `--game <STR>`/ `-g <STR>`: Name of the game to rollout in. Can be set to either `minerl`, or `csgo`.
- `--encoder_config <PATH>`: Path to a configuration file specifying the visual encoder and data pipeline configuration to use. This is required.
- `--filelist <PATH> ...`: List of paths to data split files specifying the data to compute embeddings for. At least one file is required.
- `--data_base_path <PATH>`: Path to the base directory where the data is stored.
- `--output_dir <PATH>`: Path to a directory to store the computed embeddings in. This is required.
- `--encoder_family <STR>`: Family of pre-trained encoder to overwrite encoder defined in provided config. Default: `None`.
- `--encoder_name <STR>`: Name of pre-trained encoder to overwrite encoder defined in provided config. Default: `None`.
- `--num_files <INT>`: Number of files to compute embeddings for. Default: `-1` (compute embeddings for all files in provided filelist).
- `--batch_size <INT>`: Batch size to use for computing embeddings. Default: `2048`.

Embeddings are computed in batches for loaded trajectories. Depending on the pre-trained visual encoder, the computation of embeddings might require significant VRAM when using GPUs. If the computation of embeddings fails due to insufficient VRAM, try reducing the batch size.

Once embeddings for a pre-trained encoder are computed for the dataset, place the embeddings in a `embeddings` directory in the same directory where the data is loaded from. Then, the embeddings can be loaded during training by specifying the `model.train_from_embeddings: True` hyperparameter in the training config.

# Implemented Agent Architectures
In this section, we will briefly outline the agent architectures we implemented and used for agents. The entry point for all agent architectures is `pixelbc/models/bc_model.py`. The general architecture is split into a visual encoder and policy network. The visual encoder receives a previously downscaled image and computes an embedding of the image. The policy network receives an embedding of the image (or a sequence of embeddings) and outputs actions to take. First, we will outline visual encoders before describing the supported policy networks.

## Visual Encoders
Visual encoders are fundamentally split into two categories: pre-trained visual encoders, and end-to-end visual encoders. Pre-trained visual encoders are loaded from publicly available checkpoints and frozen during training. This means, their embeddings are fixed and can be pre-computed for all data points in the dataset (see [above](#pre-compute-embeddings-for-pre-trained-visual-encoders)). End-to-end visual encoders are trained from scratch during training.

### End-To-End Visual Encoders
The entry point for end-to-end trained visual encoders is `pixelbc/models/encoders/encoders.py` with five general architectures being supported:
1. Nature DQN CNN: This is a simple CNN model proposed for DQN to play Atari (see [paper](https://www.nature.com/articles/nature14236)). This model is implemented in `pixelbc/models/encoders/dqn_cnn.py` and its type is described as `nature_cnn`.
2. Impala ResNet: This is a simple ResNet architecture proposed with the Impala RL algorithm (see [paper](https://arxiv.org/abs/1802.01561)). This model is implemented in `pixelbc/models/encoders/impala_resnet.py` and its type is described as `impala_resnet`.
3. Custom ResNet: This is a ResNet architecture based on [this paper](https://arxiv.org/abs/2201.03545) which can be further customised. This model is implemented in `pixelbc/models/encoders/resnet.py` and its type is described as `custom_resnet`.
4. Custom ViT: This is a vision transformer architecture which can be further customised. This model is implemented in `pixelbc/models/encoders/vit.py` and its type is described as `custom_vit`.
5. MLP: This is a simple MLP architecture to encode images. This model uses the general MLP implementation provided in `pixelbc/models/utils/model_utils.py` and its type is described as `mlp`.

Each encoder architecture is wrapped with `ImageEncoderWrapper` (defined in `pixelbc/models/encoders/encoders.py`) which ensures encoder inputs and outputs are of correct shapes and applies a torchvision processing function depending on the encoder. Note, this processing function only handles image colour normalisation and potential image augmentations but does not resize the images! Resizing is done as part of the data processing. Processing pipelines are implemented in `pixelbc/models/utils/image_augmentations.py`.

To use the each type of visual encoders, specify their type in the config parameter as follows:
```yaml
model:
  encoder:
    type: <ENCODER_TYPE>
```

For the custom ResNet architecture, the following parameters can be used to customise the architecture
```yaml
model:
  encoder:
    cnn_encoder_dim: <INT>
    cnn_encoder_start_channels: <INT>
```
by determining the number of channels in the first ResNet block (channel number is doubled in each block) and the encoder dimension the image dimensions are processed down to. The smaller the encoder dimension, the more ResNet blocks are used to process the image down to the specified dimension. We refer to `pixelbc/models/encoders/resnet.py` for more details.

For the custom ViT architecture, the following parameters can be used to customise the architecture
```
model:
  encoder:
    vit_encoder_patch_size: <INT>
    vit_encoder_dim: <INT>
    vit_encoder_num_layers: <INT>
    vit_encoder_num_heads: <INT>
    vit_encoder_mlp_dim: <INT>
```
by determining the patch size (input image width & height has to be divisable by the patch size), the encoder dimension the attention operates on, the number of layers, the number of heads for the attention, and the MLP projection dimension. We refer to `pixelbc/models/encoders/vit.py` for more details.

For the MLP encoder architecture (not recommended), the following parameters can be used to customise the architecture
```
model:
  encoder:
    mlp_encoder_hidden_size: <INT>
    mlp_encoder_num_layers: <INT>
```
by determining the hidden dimension of each layer and number of layers.

### Pre-Trained Visual Encoders
The integration for all pre-trained visual encoders can be found in `pixelbc/models/encoders/pretrained_encoders.py`. Currently, we support the following families of pre-trained visual encoders:
- Meta DINOv2: Models integrated from `torch`, see [repository](https://github.com/facebookresearch/dinov2) and [paper](https://arxiv.org/abs/2304.07193).
- OpenAI CLIP: Models integrated from [`clip` repository](https://github.com/openai/CLIP), see [paper](https://arxiv.org/abs/2103.00020).
- FocalNet: Models integrated from [`timm`](https://huggingface.co/docs/timm/index), see [paper](https://arxiv.org/abs/2203.11926).
- Stable Diffusion 2.1 VAE: Model integrated from [`diffusers` repository](https://huggingface.co/docs/diffusers/index).

Pre-trained encoders are frozen throughout training and are within a wrapper to ensure their inputs and outputs are of correct shapes and to ensure a consistent interface across all encoders. Their respective processing is implemented in `pixelbc/models/utils/image_augmentations.py`. Pre-trained encoders can be used by defining their family and model name in the config as follows:
```yaml
model:
  pretrained_encoder:
    family: <STR>
    name: <STR>
```
with the family being `dino`, `clip`, `focal` or `stablediffusion` for the respective model groups. For names of each particular family, see the `PRETRAINED_ENCODERS` dictionary in `pixelbc/models/encoders/pretrained_encoders.py`.

## Policy Networks
Policy networks receive the embedding computed by the respective visual encoder and output actions. We support three different policy architectures:
- MLP: This is a simple MLP architecture to output actions. This model uses the general MLP implementation provided in `pixelbc/models/utils/model_utils.py` and its type is described as `mlp`. This architecture does not support any form of temporal context/ history and simply outputs actions conditioned on only the current image embedding.
- LSTM: This is a simple LSTM architecture to output actions. This model uses the general LSTM implementation provided in `pixelbc/models/utils/model_utils.py` and its type is described as `lstm`. As a recurrent network, this architecture supports temporal context/ history and outputs actions conditioned on the current image embedding and the previous hidden state of the LSTM which can embed previously received image embeddings. As implemented, this architecture is trained on random subsequences of a trajectory to learn to predict the next action given the current image embedding and previous hidden state. During online rollouts, we reset the hidden state of the LSTM to zero at the beginning of each trajectory and then keep the hidden state accumulating. This discrepancy between training and rollout seemed to not affect performance but should be noted.
- GPT: This is a GPT transformer model akin to GPT-2. This model uses the general GPT implementation provided in `pixelbc/models/utils/pico_gpt.py` and its type is described as `gpt`. Pico-GPT is a stripped down version of the commonly used [Nano-GPT](https://github.com/karpathy/nanoGPT). As a transformer architecture, the model receives sequences of a fixed length as input and outputs actions conditioned on the current image embedding and the previous sequence of image embeddings. We note that we were unable to train successful agents with the GPT transformer architecture in our experiments.

To use a MLP policy network, specify its type in the config parameter as follows:
```yaml
model:
  bc_policy:
      type: mlp
      hidden_size: <INT>
      num_layers: <INT>
```
with `hidden_size` determining the hidden dimension of each layer and `num_layers` determining the number of layers.

To use a LSTM policy network, specify its type in the config parameter as follows:
```yaml
model:
  bc_policy:
      type: lstm
      lstm_hidden_size: 512
      lstm_num_layers: 2
```
with `lstm_hidden_size` determining the hidden dimension of the LSTM and `lstm_num_layers` determining the number of layers. Note, we use a single-layered MLP to first project the image embeddings to the hidden dimension of the LSTM before feeding them into the LSTM. This is done to ensure the input dimension of the LSTM is consistent across different visual encoders. Also, the output of the last LSTM layer is projected with a single-layered MLP to the action space dimension.

To use a GPT policy network, specify its type in the config parameter as follows:
```yaml
model:
  bc_policy:
      type: gpt
      gpt_num_layers: <INT>
      gpt_num_heads: <INT>
      gpt_embedding_dim: <INT>
      gpt_use_positional_encoding: <BOOL>
      gpt_bias: <BOOL>
      gpt_is_causal: <BOOL>
```
with `gpt_num_layers` determining the number of layers, `gpt_num_heads` determining the number of heads for the attention, `gpt_embedding_dim` determining the embedding dimension of the GPT, `gpt_use_positional_encoding` determining whether to use positional encoding (done with the timestep within the embedding being embedded), `gpt_bias` determining whether to use bias in the attention, and `gpt_is_causal` determining whether to use causal attention (only timesteps until a certain point in the sequence can be used to determine its actions, image embeddings after timestep t are not used to compute the action a_t but instead are masked out). Similar to LSTMs, we use a single-layered MLP to project the image embeddings to the embedding dimension of the GPT before feeding them into the GPT and to project the output of the last GPT layer to the action space dimension.

When training a GPT model, it is important to use learning warmup which appears essential for stable training. This can be done by specifying the `gpt_warmup_steps` parameter in the config.


# Create Visualisations
In this section, we will outline scripts to create visualisations.

## Plot Training Metrics
During training, a tensorboard file is stored including various training metrics. To visualise these metrics, we can simply use tensorboard by running
```bash
tensorboard --logdir <PATH>
```
with `<PATH>` being a directory (recursively) containing at least one tensorboard file.

## Create Grad-CAM Visualisations
Gradient-weighted Class Activation Mapping (Grad-CAM) is a technique to inspect activations of models to better understand which parts of the input images are relevant for particular parts of a model's output. To learn more about Grad-CAM, we refer to the [original paper](https://arxiv.org/abs/1610.02391) and [code repository](https://github.com/jacobgil/pytorch-grad-cam/tree/master) used by our implementation.

For such visualisations, we need a trained model checkpoint and an image in the game the model was trained in. To generate a visualisation for a trained model, run
```bash
python pixelbc/plotting/plot_grad_cam.py --checkpoint_path <PATH> --image_path <PATH> (<OPTIONAL-ARGUMENTS>)
```
with the following arguments.
- `--checkpoint_path <PATH>`: Path to a model checkpoint (as stored by training). This is required.
- `--image_path <PATH>`: Path to an image in the game the model was trained in. This is required.
- `--save_path <PATH>`: Path to a directory to store the visualisations in. Default: `None` (visualisations will be shown but not stored).
- `--target_concept <STR>`: This determines the target concept with respect to which activations in the visual encoder are visualised. Currently, we support four target concepts: `embedding`, `actions`, `movement`, and `attack`. These visualise activations with respect to the embedding as output of the visual encoder, the average activations across all actions outputted by the policy of the agent, the average activations across all movement actions outputted by the policy of the agent, and the activations for the attack action outputted by the policy of the agent, respectively. Default: `actions`.

Grad-CAM visualisations look at the activations in particular parts of the visual encoder with respect to the target concept. The particular layers of the visual encoder to look at are specified in `pixelbc/plotting/plot_grad_cam.py` in the `get_target_layers_and_reshape_transform` function. To change the layers to look at, modify the output of this function.

## Report Rollout Performance for Minecraft MineRL and CS:GO
To report the performance of a rollout in Minecraft MineRL or CS:GO, we provide scripts which output performance metrics for each model which can be found in `pixelbc/plotting/csgo_report_results.py` and `pixelbc/plotting/minerl_plot_treechop.py`. For MineRL, we report the success ratios of chopping trees in rollouts, the average, standard deviation and error across success ratios per model and can conduct a t-test to evaluate significance of differences in success ratios between models. Similarly, for CS:GO we report the average, standard deviation and error across the kills-per-minute achieved per model.

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft  trademarks or logos is subject to and must follow  [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.
