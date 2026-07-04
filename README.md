# ClinGen-DSDA
ClinGen-DSDA: Few-Shot Clinical Note Generation via Entity-Guided Dual-Stage Attention

# Requirements

Please install required packages using


pip install -r requirements.txt


# Datasets

1. ACI-Bench dataset is provided for training, validation and testing
2. Endocrinology dataset is provided at Hugging-Face at https://huggingface.co/collections/starfishdata/endocrinology-synthetic-datasets


# Step 1: Training the Full Pipeline

set number of gpus with " CUDA_VISIBLE_DEVICES " depend on number of gpus availability, you also can change the model if want to train with another specific backbone

CUDA_VISIBLE_DEVICES=0,1,2,3,4 python Stage4.py

# Step 2: Evaluation 

   CUDA_VISIBLE_DEVICES=0,1,2,3,4 python evaluate_clingen.py \
       --checkpoint outputs/multiseed/stage4/seed_42/best_model_trainable \
       --data_csv   datasets/ACI-TaskB-TestingSet.csv \
       --out_dir    outputs/eval/seed_42 \
       --batch_size 1


# Step 3: Inference to check the output

   CUDA_VISIBLE_DEVICES=0,1,2,3,4 python infer_clingen.py \
       --checkpoint outputs/multiseed/stage4/seed_42/best_model_trainable \
       --input_csv  datasets/new_dialogues.csv \
       --out_jsonl  outputs/inference/notes.jsonl \
       --batch_size 1

Give an dialogue an example as input - model will generate the note

