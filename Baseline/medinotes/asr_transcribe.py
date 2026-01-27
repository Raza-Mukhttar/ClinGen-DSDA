# File: medinotes_repro/scripts/asr_transcribe.py
"""
ASR: transcribe audio files with Whisper and diarize speakers with pyannote.
Outputs same JSONL records with 'dialogue' replaced by transcribed text.
Note: this script requires audio files and may be run offline.
Usage:
  python asr_transcribe.py --audio_dir /path/to/audio --out data/transcribed.jsonl
"""
import argparse, os, json
from glob import glob
import whisper
# diarization requires pretrained pyannote pipelines & models (not shown here)
def transcribe(audio_f):
    model = whisper.load_model("base")
    res = model.transcribe(audio_f, language="en", task="transcribe")
    return res["text"]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--audio_dir", required=True)
    p.add_argument("--out", required=True)
    args=p.parse_args()
    auds=glob(os.path.join(args.audio_dir,"*.wav"))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out,"w",encoding="utf-8") as fh:
        for a in auds:
            text=transcribe(a)
            rec={"id":os.path.basename(a),"dialogue":text,"soap":""}
            fh.write(json.dumps(rec,ensure_ascii=False)+"\n")
    print("Transcribed", len(auds))

if __name__=="__main__":
    main()
