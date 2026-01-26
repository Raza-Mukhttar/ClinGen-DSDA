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
    train_path = '/datasets/ClinGen_TrainingSet.csv'
    val_path   = '/datasets/ACI-TaskB-ValidationSet.csv'
    test_path  = '/datasets/ACI-TaskB-TestingSet.csv'
    outdir = '/outputs/stage1_outputs'
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


