import os, torch, numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from seqeval.metrics import classification_report, f1_score
import spacy
import torch.nn as nn

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"   
SPACY_MODEL = "en_ner_bc5cdr_md"
ENTITY_TYPES = ["DISEASE", "CHEMICAL"]
LABEL_LIST = ["O"] + [f"{p}-{etype}" for etype in ENTITY_TYPES for p in ["B", "I"]]
label2id = {label: i for i, label in enumerate(LABEL_LIST)}
id2label = {i: label for i, label in enumerate(LABEL_LIST)}

def load_dialogues(csv_path):
    df = pd.read_csv(csv_path)
    return df["dialogue"].astype(str).tolist()

def get_bio_labels(tokens, entities, entity_types):
    labels = ["O"] * len(tokens)
    for ent in entities:
        etype = ent.label_.upper()
        if etype not in entity_types: continue
        start, end = ent.start, ent.end
        labels[start] = f"B-{etype}"
        for i in range(start+1, end):
            labels[i] = f"I-{etype}"
    return labels

def auto_label_with_scispacy(dialogues, nlp, entity_types):
    all_tokens, all_labels = [], []
    for text in tqdm(dialogues, desc="Auto-labeling with scispaCy"):
        doc = nlp(text)
        tokens = [t.text for t in doc]
        bio_labels = get_bio_labels(doc, doc.ents, entity_types)
        all_tokens.append(tokens)
        all_labels.append([label2id.get(l, 0) for l in bio_labels])
    return all_tokens, all_labels

def encode_and_align_labels(tokens, labels, tokenizer, max_length=512):
    tokenized_inputs = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_offsets_mapping=True
    )
    word_ids = tokenized_inputs.word_ids()
    aligned_labels = []
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            aligned_labels.append(-100)
        elif word_idx != previous_word_idx:
            aligned_labels.append(labels[word_idx])
        else:
            aligned_labels.append(-100)
        previous_word_idx = word_idx
    tokenized_inputs['labels'] = aligned_labels
    return {k: tokenized_inputs[k] for k in ['input_ids', 'token_type_ids', 'attention_mask', 'labels']}

def process_examples(all_tokens, all_labels, tokenizer):
    input_ids, token_type_ids, attention_mask, labels_list = [], [], [], []
    for tokens, tags in tqdm(zip(all_tokens, all_labels), desc="Tokenizing", total=len(all_tokens)):
        encoded = encode_and_align_labels(tokens, tags, tokenizer)
        input_ids.append(encoded['input_ids'])
        token_type_ids.append(encoded['token_type_ids'])
        attention_mask.append(encoded['attention_mask'])
        labels_list.append(encoded['labels'])
    return {
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask,
        'labels': labels_list,
    }

class NERDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __len__(self):
        return len(self.encodings['input_ids'])
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'token_type_ids': torch.tensor(self.encodings['token_type_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.encodings['labels'][idx])
        }

def compute_class_weights(labels_list, n_labels):
    counts = np.zeros(n_labels)
    for label_seq in labels_list:
        for l in label_seq:
            if l >= 0: counts[l] += 1
    weights = np.ones(n_labels)
    for i in range(1, n_labels):
        if counts[i] > 0:
            weights[i] = counts[0] / counts[i]
    # Cap the maximum weight to avoid overfitting rare classes
    max_weight = 100 
    weights = np.minimum(weights, max_weight)
    weights[0] = 1.0
    weights = weights / weights.min()
    print("Sanity check class weights:", weights)
    return torch.tensor(weights, dtype=torch.float)

# ========== Main ==========

def main():
    train_path = '/home/raza/EMRV/EMRV2/datasets/ClinGen_TrainingSet.csv'
    val_path   = '/home/raza/EMRV/EMRV2/datasets/ACI-TaskB-ValidationSet.csv'
    test_path  = '/home/raza/EMRV/EMRV2/datasets/ACI-TaskB-TestingSet.csv'
    outdir = '/home/raza/EMRV/EMRV2/outputs/stage1_outputs'
    os.makedirs(outdir, exist_ok=True)

    print(f"Loading scispaCy NER model ({SPACY_MODEL})...")
    nlp = spacy.load(SPACY_MODEL)
    print("Loading dialogues...")
    train_dialogues = load_dialogues(train_path)
    val_dialogues   = load_dialogues(val_path)
    test_dialogues  = load_dialogues(test_path)

    print("\n=== Checking scispaCy NER output on first 3 training dialogues ===")
    for text in train_dialogues[:3]:
        doc = nlp(text)
        print("Detected entities:", [(ent.text, ent.label_) for ent in doc.ents])
    print("="*60)

    print("Auto-labeling with scispaCy (train)...")
    train_tokens, train_labels = auto_label_with_scispacy(train_dialogues, nlp, ENTITY_TYPES)
    print("Auto-labeling with scispaCy (val)...")
    val_tokens, val_labels = auto_label_with_scispacy(val_dialogues, nlp, ENTITY_TYPES)
    print("Auto-labeling with scispaCy (test)...")
    test_tokens, test_labels = auto_label_with_scispacy(test_dialogues, nlp, ENTITY_TYPES)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Tokenizing splits...")
    train_encodings = process_examples(train_tokens, train_labels, tokenizer)
    val_encodings   = process_examples(val_tokens, val_labels, tokenizer)
    test_encodings  = process_examples(test_tokens, test_labels, tokenizer)
    train_dataset = NERDataset(train_encodings)
    val_dataset   = NERDataset(val_encodings)
    test_dataset  = NERDataset(test_encodings)

    print("Computing class weights...")
    class_weights = compute_class_weights(train_encodings["labels"], len(LABEL_LIST))
    print("Class weights:", class_weights)

    print("Loading model...")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=id2label,
        label2id=label2id,
    )

    # ========== Weighted loss, Trainer-style ==========

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch: int | None = None, **kwargs):
            labels = inputs.get("labels")
            # Forward pass
            outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
            logits = outputs.logits
            # Put class_weights on logits.device
            loss_fct = nn.CrossEntropyLoss(weight=class_weights.to(logits.device), ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    args = TrainingArguments(
        output_dir=os.path.join(outdir, 'checkpoints'),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=60,
        save_total_limit=3,
        weight_decay=0.01,
        logging_dir=os.path.join(outdir, 'logs'),
        logging_steps=40,
        report_to='none',
        load_best_model_at_end=True,
        metric_for_best_model='eval_f1',
        greater_is_better=True          
       
    )

    def compute_metrics(p):
        predictions, labels = p.predictions, p.label_ids
        predictions = np.argmax(predictions, axis=2)
        true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [id2label[pred] for (pred, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        return {
            "f1": f1_score(true_labels, true_predictions, zero_division=0),
            "classification_report": classification_report(true_labels, true_predictions, zero_division=0),
        }

    trainer = CustomTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Training...")
    trainer.train()
    print("Evaluating on validation set...")
    print(trainer.evaluate())
    print("Evaluating on test set...")
    print(trainer.evaluate(eval_dataset=test_dataset))

    print("Saving model/tokenizer...")
    model.save_pretrained(os.path.join(outdir, 'final_model'))
    tokenizer.save_pretrained(os.path.join(outdir, 'final_model'))
    print("Stage 1 complete!")

if __name__ == "__main__":
    main()





# Multiple Datasets
# import os, torch, numpy as np
# import pandas as pd
# from tqdm import tqdm
# from torch.utils.data import Dataset
# from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
# from seqeval.metrics import classification_report, f1_score
# import spacy
# import torch.nn as nn

# # ===================== NEW TOGGLES =====================
# # Set this True to use a general-domain NER (for SAMSum and other non-medical data).
# USE_GENERAL_DOMAIN = True      # <- flip to False to restore medical (original) behavior

# # Set this True to read dialogues from the Hugging Face "samsum" dataset instead of CSV files.
# USE_HF_SAMSUM    = True       # <- flip to True for SAMSum; CSV logic stays intact otherwise
# # =======================================================

# # ---------------- ORIGINAL DEFINITIONS (kept) ----------------
# MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"   # (ORIGINAL medical)
# SPACY_MODEL = "en_ner_bc5cdr_md"                 # (ORIGINAL medical sciSpaCy, BC5CDR)
# ENTITY_TYPES = ["DISEASE", "CHEMICAL"]           # (ORIGINAL medical entity set)
# LABEL_LIST = ["O"] + [f"{p}-{etype}" for etype in ENTITY_TYPES for p in ["B", "I"]]
# label2id = {label: i for i, label in enumerate(LABEL_LIST)}
# id2label = {i: label for i, label in enumerate(LABEL_LIST)}
# # ---------------------------------------------------------------

# # ===================== DOMAIN SWITCH (NEW) =====================
# if USE_GENERAL_DOMAIN:
#     # COMMENT OUT originals for clarity (not removed):
#     # MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
#     # SPACY_MODEL = "en_ner_bc5cdr_md"
#     # ENTITY_TYPES = ["DISEASE", "CHEMICAL"]

#     # GENERAL-DOMAIN SETTINGS (spaCy English + BERT)
#     MODEL_NAME = "bert-base-cased"           # safe for TokenClassification (has token_type_ids)
#     SPACY_MODEL = "en_core_web_sm"           # common general NER model (PERSON/ORG/GPE/...)
#     # Broad, but you can trim this list based on your task:
#     ENTITY_TYPES = [
#         "PERSON","ORG","GPE","LOC","NORP","FAC","PRODUCT","EVENT","WORK_OF_ART",
#         "LAW","LANGUAGE","DATE","TIME","PERCENT","MONEY","QUANTITY","ORDINAL","CARDINAL"
#     ]
#     # Recompute label maps
#     LABEL_LIST = ["O"] + [f"{p}-{etype}" for etype in ENTITY_TYPES for p in ["B", "I"]]
#     label2id = {label: i for i, label in enumerate(LABEL_LIST)}
#     id2label = {i: label for i, label in enumerate(LABEL_LIST)}
# # ==============================================================

# def load_dialogues(csv_path):
#     df = pd.read_csv(csv_path)
#     return df["dialogue"].astype(str).tolist()

# def get_bio_labels(tokens, entities, entity_types):
#     labels = ["O"] * len(tokens)
#     for ent in entities:
#         etype = ent.label_.upper()
#         if etype not in entity_types: 
#             continue
#         start, end = ent.start, ent.end
#         labels[start] = f"B-{etype}"
#         for i in range(start+1, end):
#             labels[i] = f"I-{etype}"
#     return labels

# def auto_label_with_scispacy(dialogues, nlp, entity_types):
#     all_tokens, all_labels = [], []
#     for text in tqdm(dialogues, desc="Auto-labeling with spaCy/scispaCy"):
#         doc = nlp(text)
#         tokens = [t.text for t in doc]
#         bio_labels = get_bio_labels(doc, doc.ents, entity_types)  # (passes doc for len(); indices from ent.start/end)
#         all_tokens.append(tokens)
#         all_labels.append([label2id.get(l, 0) for l in bio_labels])
#     return all_tokens, all_labels

# def encode_and_align_labels(tokens, labels, tokenizer, max_length=512):
#     tokenized_inputs = tokenizer(
#         tokens,
#         is_split_into_words=True,
#         truncation=True,
#         padding="max_length",
#         max_length=max_length,
#         return_offsets_mapping=True
#     )
#     word_ids = tokenized_inputs.word_ids()
#     aligned_labels = []
#     previous_word_idx = None
#     for word_idx in word_ids:
#         if word_idx is None:
#             aligned_labels.append(-100)
#         elif word_idx != previous_word_idx:
#             aligned_labels.append(labels[word_idx])
#         else:
#             aligned_labels.append(-100)
#         previous_word_idx = word_idx
#     tokenized_inputs['labels'] = aligned_labels
#     # Some models (e.g., RoBERTa) may not return token_type_ids; keep original key to be safe
#     if 'token_type_ids' not in tokenized_inputs:
#         tokenized_inputs['token_type_ids'] = [0] * len(tokenized_inputs['input_ids'])
#     return {k: tokenized_inputs[k] for k in ['input_ids', 'token_type_ids', 'attention_mask', 'labels']}

# def process_examples(all_tokens, all_labels, tokenizer):
#     input_ids, token_type_ids, attention_mask, labels_list = [], [], [], []
#     for tokens, tags in tqdm(zip(all_tokens, all_labels), desc="Tokenizing", total=len(all_tokens)):
#         encoded = encode_and_align_labels(tokens, tags, tokenizer)
#         input_ids.append(encoded['input_ids'])
#         token_type_ids.append(encoded['token_type_ids'])
#         attention_mask.append(encoded['attention_mask'])
#         labels_list.append(encoded['labels'])
#     return {
#         'input_ids': input_ids,
#         'token_type_ids': token_type_ids,
#         'attention_mask': attention_mask,
#         'labels': labels_list,
#     }

# class NERDataset(Dataset):
#     def __init__(self, encodings):
#         self.encodings = encodings
#     def __len__(self):
#         return len(self.encodings['input_ids'])
#     def __getitem__(self, idx):
#         return {
#             'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
#             'token_type_ids': torch.tensor(self.encodings['token_type_ids'][idx]),
#             'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
#             'labels': torch.tensor(self.encodings['labels'][idx])
#         }

# def compute_class_weights(labels_list, n_labels):
#     counts = np.zeros(n_labels)
#     for label_seq in labels_list:
#         for l in label_seq:
#             if l >= 0: counts[l] += 1
#     weights = np.ones(n_labels)
#     for i in range(1, n_labels):
#         if counts[i] > 0:
#             weights[i] = counts[0] / counts[i]
#     max_weight = 100
#     weights = np.minimum(weights, max_weight)
#     weights[0] = 1.0
#     weights = weights / weights.min()
#     print("Sanity check class weights:", weights)
#     return torch.tensor(weights, dtype=torch.float)

# # ========== Main ==========

# def main():
#     # ---------------- ORIGINAL PATHS (kept) ----------------
#     train_path = '/home/raza/EMRV1/EMRV2/datasets/ClinGen_TrainingSet.csv'
#     val_path   = '/home/raza/EMRV1/EMRV2/datasets/ACI-TaskB-ValidationSet.csv'
#     test_path  = '/home/raza/EMRV1/EMRV2/datasets/ACI-TaskB-TestingSet.csv'
#     outdir = '/home/raza/EMRV1/EMRV2/outputs/stage111_outputs'
#     # -------------------------------------------------------
#     os.makedirs(outdir, exist_ok=True)

#     print(f"Loading spaCy NER model ({SPACY_MODEL})...")
#     try:
#         nlp = spacy.load(SPACY_MODEL)
#     except Exception as e:
#         print(f"Could not load '{SPACY_MODEL}' ({e}). Trying to download it...")
#         try:
#             import spacy.cli
#             spacy.cli.download(SPACY_MODEL)
#             nlp = spacy.load(SPACY_MODEL)
#         except Exception as e2:
#             # Fallback: try a very common small English model if BC5CDR or other model isn't available
#             print(f"Download failed for '{SPACY_MODEL}' ({e2}). Falling back to 'en_core_web_sm'.")
#             try:
#                 nlp = spacy.load("en_core_web_sm")
#             except Exception:
#                 import spacy.cli
#                 spacy.cli.download("en_core_web_sm")
#                 nlp = spacy.load("en_core_web_sm")

#     print("Loading dialogues...")

#     # ---------------- ORIGINAL CSV LOADER (kept but gated) ----------------
#     # train_dialogues = load_dialogues(train_path)
#     # val_dialogues   = load_dialogues(val_path)
#     # test_dialogues  = load_dialogues(test_path)
#     # ---------------------------------------------------------------------

#     if USE_HF_SAMSUM:
#         # ==== NEW: load SAMSum from HF datasets ====
#         try:
#             from datasets import load_dataset
#         except Exception as e:
#             raise RuntimeError("Please install `datasets` to use USE_HF_SAMSUM=True") from e
#         ds = load_dataset("knkarthick/samsum")   #nbroad/mediasum  whu9/mediasum_postprocess
#         train_dialogues = ds["train"]["dialogue"]
#         val_dialogues   = ds["validation"]["dialogue"]
#         test_dialogues  = ds["test"]["dialogue"]

#         # ds = load_dataset("whu9/mediasum_postprocess")   #nbroad/mediasum  whu9/mediasum_postprocess
#         # train_dialogues = ds["train"]["source"]
#         # val_dialogues   = ds["validation"]["source"]
#         # test_dialogues  = ds["test"]["source"]
#     else:
#         # Use the original CSV paths
#         train_dialogues = load_dialogues(train_path)
#         val_dialogues   = load_dialogues(val_path)
#         test_dialogues  = load_dialogues(test_path)

#     print("\n=== Checking spaCy NER output on first 3 training dialogues ===")
#     for text in train_dialogues[:3]:
#         doc = nlp(text)
#         print("Detected entities:", [(ent.text, ent.label_) for ent in doc.ents])
#     print("="*60)

#     print("Auto-labeling with spaCy/scispaCy (train)...")
#     train_tokens, train_labels = auto_label_with_scispacy(train_dialogues, nlp, ENTITY_TYPES)
#     print("Auto-labeling with spaCy/scispaCy (val)...")
#     val_tokens, val_labels = auto_label_with_scispacy(val_dialogues, nlp, ENTITY_TYPES)
#     print("Auto-labeling with spaCy/scispaCy (test)...")
#     test_tokens, test_labels = auto_label_with_scispacy(test_dialogues, nlp, ENTITY_TYPES)

#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#     print("Tokenizing splits...")
#     train_encodings = process_examples(train_tokens, train_labels, tokenizer)
#     val_encodings   = process_examples(val_tokens, val_labels, tokenizer)
#     test_encodings  = process_examples(test_tokens, test_labels, tokenizer)
#     train_dataset = NERDataset(train_encodings)
#     val_dataset   = NERDataset(val_encodings)
#     test_dataset  = NERDataset(test_encodings)

#     print("Computing class weights...")
#     class_weights = compute_class_weights(train_encodings["labels"], len(LABEL_LIST))
#     print("Class weights:", class_weights)

#     print("Loading model...")
#     model = AutoModelForTokenClassification.from_pretrained(
#         MODEL_NAME,
#         num_labels=len(LABEL_LIST),
#         id2label=id2label,
#         label2id=label2id,
#     )

#     # ========== Weighted loss, Trainer-style ==========
#     class CustomTrainer(Trainer):
#         def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch: int | None = None, **kwargs):
#             labels = inputs.get("labels")
#             outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
#             logits = outputs.logits
#             loss_fct = nn.CrossEntropyLoss(weight=class_weights.to(logits.device), ignore_index=-100)
#             loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
#             return (loss, outputs) if return_outputs else loss

#     args = TrainingArguments(
#         output_dir=os.path.join(outdir, 'checkpoints'),
#         eval_strategy="epoch",
#         save_strategy="epoch",
#         learning_rate=1e-5,
#         per_device_train_batch_size=8,
#         per_device_eval_batch_size=8,
#         num_train_epochs=30,
#         save_total_limit=3,
#         weight_decay=0.01,
#         logging_dir=os.path.join(outdir, 'logs'),
#         logging_steps=40,
#         report_to='none',
#         load_best_model_at_end=True,
#         metric_for_best_model='eval_f1',
#         greater_is_better=True
#     )

#     def compute_metrics(p):
#         predictions, labels = p.predictions, p.label_ids
#         predictions = np.argmax(predictions, axis=2)
#         true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
#         true_predictions = [
#             [id2label[pred] for (pred, l) in zip(prediction, label) if l != -100]
#             for prediction, label in zip(predictions, labels)
#         ]
#         return {
#             "f1": f1_score(true_labels, true_predictions, zero_division=0),
#             "classification_report": classification_report(true_labels, true_predictions, zero_division=0),
#         }

#     trainer = CustomTrainer(
#         model=model,
#         args=args,
#         train_dataset=train_dataset,
#         eval_dataset=val_dataset,
#         tokenizer=tokenizer,
#         compute_metrics=compute_metrics,
#     )

#     print("Training...")
#     trainer.train()
#     print("Evaluating on validation set...")
#     print(trainer.evaluate())
#     print("Evaluating on test set...")
#     print(trainer.evaluate(eval_dataset=test_dataset))

#     print("Saving model/tokenizer...")
#     model.save_pretrained(os.path.join(outdir, 'final_model'))
#     tokenizer.save_pretrained(os.path.join(outdir, 'final_model'))
#     print("Stage 1 complete!")

# if __name__ == "__main__":
#     main()


