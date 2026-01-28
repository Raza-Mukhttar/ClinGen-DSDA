import pandas as pd
import time
import openai
import sys
sys.path.append('./UniEval')
from metric.evaluator import get_evaluator
from utils import convert_to_json
import numpy as np

client = openai.OpenAI(api_key="GPT-API-KEY")  # Replace with your actual API key

def construct_prompt(dialogue):
    
    return f"""Given the following patient-doctor dialogue, generate a detailed clinical note, including all relevant sections (such as CHIEF COMPLAINT, HISTORY OF PRESENT ILLNESS, REVIEW OF SYSTEMS, PHYSICAL EXAMINATION, RESULTS, ASSESSMENT, PLAN, ASSESSMENT AND PLAN, INSTRUCTIONS).

Dialogue:
{dialogue}

[NOTE]
"""

def generate_note(dialogue, model="gpt-4"): # or   gpt-3.5-turbo-0125
    prompt = construct_prompt(dialogue)
    for _ in range(3):  
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.2,
                max_tokens=1024,  
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error for model {model}, retrying... {e}")
            time.sleep(2)
    return ""

def canonicalize_note(text):
    
    return text

def eval_unieval(pred, src, ref, evaluator):
    pred = (pred or "").strip()
    src = (src or "").strip()
    ref = (ref or "").strip()
    if not pred or not ref or not src:
        print("Warning: Empty field detected. Skipping UniEval for this sample.")
        return {'coherence': 0, 'consistency': 0, 'fluency': 0, 'relevance': 0}
    ref = canonicalize_note(ref)
    sample_json = convert_to_json(
        output_list=[pred],
        src_list=[src],
        ref_list=[ref]
    )
    try:
        result = evaluator.evaluate(sample_json, print_result=False)
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
            result = result[0]
        return result
    except ZeroDivisionError:
        print("ZeroDivisionError in UniEval. Returning zeros.")
        return {'coherence': 0, 'consistency': 0, 'fluency': 0, 'relevance': 0}

if __name__ == "__main__":
    df = pd.read_csv("/datasets/ACI-TaskB-ValidationSet.csv")
    evaluator = get_evaluator('summarization')

    gpt4_notes, gpt4_unievals = [], []

    for idx, row in df.iterrows():
        dialogue = row['dialogue']
        note = row['note']

        print(f"Processing example {idx+1}/{len(df)}...")

        # GPT-4 or GPT-3.5 generation
        gpt4_note = generate_note(dialogue, model="gpt-4")
        gpt4_notes.append(gpt4_note)
        gpt4_unieval = eval_unieval(gpt4_note, dialogue, note, evaluator)
        gpt4_unievals.append(gpt4_unieval)

    # Add notes and UniEval scores to DataFrame
    df['gpt4_note'] = gpt4_notes

    # Expand UniEval dicts into separate columns
    gpt4_unieval_df = pd.DataFrame(gpt4_unievals)
    gpt4_unieval_df.columns = [f"gpt4_unieval_{c}" for c in gpt4_unieval_df.columns]

    df = pd.concat([df, gpt4_unieval_df], axis=1)

    # Compute averages and print
    gpt4_avg = gpt4_unieval_df.mean()

    print("\n===== Average UniEval Scores (gpt-4) =====")
    for col, val in gpt4_avg.items():
        print(f"{col.replace('gpt4_unieval_', '')}: {val:.4f}")

    df.to_csv("gpt4_notes_and_unieval.csv", index=False)
    print("Done! Results saved to 'gpt4_notes_and_unieval.csv'.")