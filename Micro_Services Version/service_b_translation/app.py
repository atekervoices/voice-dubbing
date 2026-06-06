from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import os
import re
import torch
import warnings
import gc  
from transformers import NllbTokenizer, M2M100ForConditionalGeneration
from huggingface_hub import snapshot_download

warnings.filterwarnings("ignore")

# ==========================================
#  PRE-DOWNLOAD DIRECTLY TO DISK ON BOOT
# ==========================================
print(" Checking/Downloading Sunbird SALT NLLB-1.3B to disk (This may take a while)...")
model_name = "Sunbird/translate-nllb-1.3b-salt"

# This ONLY downloads the files to your hard drive. It does NOT use any RAM!
snapshot_download(repo_id=model_name)
print(" Sunbird SALT Model is fully downloaded and securely cached on your hard drive!")

app = FastAPI()

class TranslationRequest(BaseModel):
    transcript_filename: str

@app.post("/translate")
def run_translation(request: TranslationRequest):
    print(f" Service B activated! Translating: {request.transcript_filename}")
    
    # --- SHARED FOLDER PATHS ---
    data_dir = "/app/data"
    input_filepath = f"{data_dir}/{request.transcript_filename}"
    output_filename = f"{data_dir}/translated_transcript_fixed.json"

    if not os.path.exists(input_filepath):
        raise HTTPException(status_code=404, detail=f"File {input_filepath} not found.")

    # ==========================================
    # 🚀 LOAD MODEL FROM HARD DRIVE -> GPU VRAM
    # ==========================================
    print("Loading Sunbird SALT model into GPU memory...")
    
    tokenizer = NllbTokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name).to("cuda")
    print(" Sunbird SALT Model loaded into VRAM!")

    # ==========================================
    # PHASE 1: LOAD DATA
    # ==========================================
    with open(input_filepath, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)

    for entry in transcript_data:
        if "speaker" not in entry:
            entry["speaker"] = "UNKNOWN_SPEAKER"

    # ==========================================
    # PHASE 2: DEEP TRANSLATION ON SEGMENTS
    # ==========================================
    print("Processing Deep Translation via Beam Search on segments...")
    
    # Language token IDs for the SALT model
    LANGUAGE_TOKENS = {
        'eng': 256047,
        'ach': 256111,
        'lgg': 256008,
        'lug': 256110,
        'nyn': 256002,
        'teo': 256006,
    }
    
    translated_data = []

    for segment in transcript_data:
        original_text = segment.get("text", "").strip()
        clean_text = original_text.replace("♪", "").strip()
        if not clean_text: continue
        
        inputs = tokenizer(clean_text, return_tensors="pt").to("cuda")
        # Set source language token
        inputs['input_ids'][0][0] = LANGUAGE_TOKENS['eng']
        
        translated_tokens = model.generate(
            **inputs, 
            forced_bos_token_id=LANGUAGE_TOKENS['lug'],  # Target language
            max_length=100, 
            num_beams=5,             
            repetition_penalty=1.2   
        )
        
        translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        
        print(f"[{segment['start']}s] {segment['speaker']}: {translated_text}")
        
        translated_data.append({
            "speaker": segment["speaker"],
            "start": segment["start"],
            "end": segment["end"],
            "english_original": original_text,
            "french_translation": translated_text
        })

    # ==========================================
    # PHASE 4: FINAL EXPORT
    # ==========================================
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(translated_data, f, indent=4, ensure_ascii=False)

    print(f" Translation Complete. Saved to {output_filename}")

    # ==========================================
    #  THE VRAM ASSASSINATION
    # ==========================================
    print(" Wiping NLLB from GPU memory...")
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return {"status": "success", "file_ready": "translated_transcript_fixed.json"}