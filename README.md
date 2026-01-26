# ClinGen-DSDA
ClinGen-DSDA: Few-Shot Clinical Note Generation via Entity-Guided Dual-Stage Attention

# Requirements

Please install required packages using:


pip install -r requirements.txt


# Datasets

1. ACI-Bench dataset is provided for training, validation and testing
2. Endocrinology dataset is provided at Hugging-Face at https://huggingface.co/collections/starfishdata/endocrinology-synthetic-datasets


# Step 1: Training the Full Pipeline

set number of gpus, like " --num_gpus=1 " depend on number of gpus availability
deepspeed --num_gpus=1 ./Stage4_LammaTraining.py

# Step 2: Evaluation 

python Evaluation.py

# Step 3: Inference to check the output

python Inference.py

Give an dialogue an example as input - model will generate the note

