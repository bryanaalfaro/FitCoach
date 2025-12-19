# Stream-VLM

<p align="center">
  <img width="640" src="/assets/stream_vlm.png" hspace="50">
</p>

Here we provide the code to evaluate the baseline Stream-VLM model in the paper:

[**Live Fitness Coaching as a Testbed for Situated Interaction (NeurIPS 2024 D&B Track)
**](https://arxiv.org/pdf/2407.08101)

## Abstract

Vision-language models have shown impressive progress in recent years. However, existing models are
largely limited to turn-based interactions, where each turn must be stepped (i.e., prompted) by the
user. Open-ended, asynchronous interactions, where an AI model may proactively deliver timely
responses or feedback based on the unfolding situation in real-time, are an open challenge. In the
paper "Live Fitness Coaching as a Testbed for Situated Interaction (NeurIPS 2024 D&B Track)", we
present the QEVD benchmark and dataset, which explores human-AI interaction in the challenging, yet
controlled, real-world domain of fitness coaching -- a task which intrinsically requires monitoring
live user activity and providing immediate feedback. The benchmark requires vision-language models
to recognize complex human actions, identify possible mistakes, and provide appropriate feedback in
real-time. Our experiments reveal the limitations of existing state-of-the-art vision-language
models for such asynchronous situated interactions. Motivated by this, we propose a simple
end-to-end streaming baseline (Stream-VLM model) that can respond asynchronously to human actions
with appropriate feedback at the appropriate time. This repository contains the code to evaluate
the baseline Stream-VLM model.

## Running the Code

### Getting Started

First, clone the repository using the following command:

```
git clone https://github.com/Qualcomm-AI-research/FitCoach.git $REPO_PATH
cd $REPO_PATH
```

Here, *$REPO_PATH* is the desired location to download the repository.

Next, build a docker image with the project requirements using the following command:

```
docker build -f docker/Dockerfile --tag fitcoach:latest .
```

Next, start a docker container from the built image:

```
docker run --gpus all --name fitcoach -it --rm fitcoach
```

### Download Checkpoints

First, download the LLaMA-2-7B checkpoint available [here](https://huggingface.co/meta-llama/Llama-2-7b). Next, download the weights of the 3D CNN,

```
wget --no-check-certificate -P ./ckpts_efficientnet https://github.com/Qualcomm-AI-research/FitCoach/releases/download/v1.0/efficientnet_3d_cnn_weights.tar.gz
``` 

and then download the weights of the Stream-VLM model which is chuncked into six parts,
```
wget --no-check-certificate -P ./ckpts_streamvlm https://github.com/Qualcomm-AI-research/FitCoach/releases/download/v1.0/streamvlm_weights.tar.gz.aa
wget --no-check-certificate -P ./ckpts_streamvlm https://github.com/Qualcomm-AI-research/FitCoach/releases/download/v1.0/streamvlm_weights.tar.gz.ab
wget --no-check-certificate -P ./ckpts_streamvlm https://github.com/Qualcomm-AI-research/FitCoach/releases/download/v1.0/streamvlm_weights.tar.gz.ac
wget --no-check-certificate -P ./ckpts_streamvlm https://github.com/Qualcomm-AI-research/FitCoach/releases/download/v1.0/streamvlm_weights.tar.gz.ad
wget --no-check-certificate -P ./ckpts_streamvlm https://github.com/Qualcomm-AI-research/FitCoach/releases/download/v1.0/streamvlm_weights.tar.gz.ae
wget --no-check-certificate -P ./ckpts_streamvlm https://github.com/Qualcomm-AI-research/FitCoach/releases/download/v1.0/streamvlm_weights.tar.gz.af
```

The above commands will download the 3D CNN and the Stream-VLM model weights to *./ckpts_efficientnet* and *./ckpts_streamvlm* respectively. These paths can be updated as necessary, please update the *docker run* command to mount the volume for these paths in the docker container.

To extract the weights from the respective tar archives use the *tar xf* command. For the chuncked *streamvlm_weights.tar.gz* archive use the following command,
```
cat streamvlm_weights.tar.gz* | tar xf -
```

Finally, to obtain the LLM-Acc metric download the LLaMA-3-70B-Instruct weights (and tokenizer) available [here](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct).

### Data Extraction

Next, download the [QEVD](https://www.qualcomm.com/developer/software/qevd-dataset) dataset. To evaluate the Stream-VLM model only the QEVD-FIT-COACH benchmark needs to be downloaded. After download, the
EfficientNet 3D CNN features used by the Stream VLM model can be extracted using the following
command:

```
python scripts/extract_efficientnet_features.py --data_dir <Path to the downloaded QEVD dataset.> --model_dir <EfficientNet 3D CNN model weights path> --fit_coach_benchmark
```

Use the `--fit_coach_benchmark` flag to extract the features for the QEVD-FIT-COACH benchmark
and the `--fit_coach_dataset` flag for the QEVD dataset. By default the features will be
extracted to *EFFICIENTNET_FEATURES_DIR* within the QEVD dataset folder.

### Evaluation

To evaluate interactive feedback generated by the Stream-VLM model, use the following command:

```
PYTHONPATH=./ python scripts/evaluate_baseline.py --config configs/base.yaml
```

Set the path to the downloaded LLaMA-2-7B weights (*llama2_7b_path*) and the Stream-VLM weights (
*checkpoint_path*) in `configs/base.yaml`. Set the path to the downloaded QEVD dataset (and
extracted features) in *data_root*. The generated feedbacks will be saved to the path specified by
*feedbacks_save_path* and *feedbacks_save_file_name* json file.

The above script generates the METEOR, ROUGE-L, BERT and Temporal F-scores along with the json
result file (at *feedbacks_save_path/feedbacks_save_file_name.json*). To obtain the LLM-Accuracy use
the following command:

```
python scripts/get_llm_accuracy.py --results_file <Path to the generated json results file> --llm_model_path <Path to the directory containing LLaMA-3-70B-Instruct weights> --llm_tokenizer_path <Path to the directory containing LLaMA-3-70B-Instruct tokenizer (usually same as llm_model_path and can be ignored)>
```

The path to the generated json results file should be
*feedbacks_save_path/feedbacks_save_file_name.json*.

Running the commands above should reproduce the following results reported in the paper:

| METEOR | ROUGE-L | BERT  | LLM-Acc | T-F-Score |
|--------|---------|-------|---------|-----------|
| 0.127  | 0.112   | 0.862 | 2.45    | 0.56      |

### Compute Requirements

We recommend the use of a GPU with at least 48GB VRAM along with at least 32GB of RAM.

## Repository Structure

The repository has the following structure,

```text
fitcoach
|   Software Evaluation License - Fitcoach.doc : License file
|   assets/ : Images in README
|   └── stream_vlm.png
└───configs/ : configuration YAML files for evaluation
|   └── base.yaml : base configuration for the Stream-VLM model
|   docker/ : docker setup files
|   ├── Dockerfile : docker setup file for the Stream-VLM model
|   └── requirements-pip.txt : requirements file for the Stream-VLM model
└───scripts/ : scripts for evaluation
|   ├── extract_efficientnet_features.py : script to extract 3D CNN features
|   ├── evaluate_baseline.py : script to get METEOR, ROUGLE-L, BERT and T-F-Score
|   └── get_llm_accuracy.py : script to the LLM-Acc metric
└───src/ : core library
|   ├── constants.py : defines constants variables used throughout the repo
|   ├── customllama/ : patches the llama model to include cross attention layers
|   │   ├── configuration_llama.py : base config (https://github.com/huggingface/transformers)
|   │   └── modeling_llama.py : updates llama decoder with cross attention layers
|   ├── evaluation_helpers.py : helper functions for evaluation
|   ├── evaluators.py : defines evaluator classes
|   ├── fitness_datasets/ : defines dataset loaders
|   │   ├── fitcoach.py : loads the QEVD-FIT-COACH benchmark and dataset
|   │   └── load_qevd.py : helper functions for data loading
|   ├── model_helpers.py : helper functions for loading the Stream-VLM model
|   ├── model_wrappers.py : wraps the 3D CNN encoder and the language backbone into the Stream-VLM model
|   ├── utils.py : misc functions
|   ├── vision_modules/ : defines the vision model
|   │   ├── adapter.py : defines the adpater between the 3D CNN encoder and the language backbone
|   │   ├── cross_attention.py : defines the cross attention layers
|   │   ├── sense_backbone/ : defines the 3D CNN model
|   │   ├── utils.py : misc helper functions
|   │   └── vision_model.py : defines wrapper around the 3D CNN encoder
```

## Citation

```text
@inproceedings{livefit,
   title = {Live Fitness Coaching as a Testbed for Situated Interaction},
   author = {Sunny Panchal and Apratim Bhattacharyya and Guillaume Berger and Antoine Mercier and Cornelius B{\"{o}}hm and Florian Dietrichkeit and Reza Pourreza and Xuanlin Li and Pulkit Madan and Mingu Lee and Mark Todorovich and Ingo Bax and Roland Memisevic},
   booktitle = {NeurIPS (D&B Track)},
   year = {2024},
}
```

## Full-video finetuning
This branch contains the code for finetuning the language backbone of the Stream-VLM with the goal of targetting improved performance when using a full ~3.5 minute workout sequence for evaluation instead of a 30 second clip of a single exercise. Here is a summary of the changes:

### Implementation of training loop
This was not included in the original repo, and we have implemented this functionality in `scripts/train_language.py`. This supports finetuning using the original 30-second video clips, full 3.5 minute workout videos, and with sliding window segments of a full video. This can be changed by modifiying the `eval_mode` field of a configuration file. Due to GPU memory constraints, we are limited to a batch size of 1 during training (meaning either 1 30-second clip or full workout, since they use the same collator function), and instead have implemented gradient accumulation. The training loop can be initiated with 
```
python scripts/train_language.py --config <path_to_config>
```
and we provide a sample config in `train_segments.yaml`

### More evaluation modes
As a result of introducing the capability of training using the sliding-window mode, in which the model's feedbacks from the previous window become a part of the prompt for the current window, we needed to include support for this in the base evaluators from the original repo. This capability has been added in `src/evaluators.py`, and methods for construcing a dataset that matches this format have been added in `src/fitness_datasets/fitcoach.py`. As before, modifying which mode is used is controlled through the `eval_mode` field in the config.
