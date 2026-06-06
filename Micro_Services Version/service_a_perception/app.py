from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import glob
import subprocess
import json
import warnings
import torch
import shutil
import gc 
import requests
from pydub import AudioSegment

warnings.filterwarnings("ignore")

app = FastAPI()

# Deployed diarization endpoint (includes transcription)
DIARIZATION_API_URL = "https://shad-top-ideally.ngrok-free.app/diarize"

class VideoRequest(BaseModel):
    video_filename: str

@app.post("/extract")
def run_extraction(request: VideoRequest):
    print(f" Service A activated! Processing: {request.video_filename}")
    
    # --- SHARED FOLDER PATHS ---
    data_dir = "/app/data"
    video_path = f"{data_dir}/{request.video_filename}"
    raw_audio_path = f"{data_dir}/raw_audio.wav"
    demucs_out_dir = f"{data_dir}/separated"
    final_json_path = f"{data_dir}/final_master_transcript.json"

    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail=f"Video {video_path} not found in shared volume.")

    # ==========================================
    # 1. EXTRACT RAW AUDIO
    # ==========================================
    print("1️⃣ Extracting raw audio...")
    os.system(f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {raw_audio_path} -y -loglevel error")

    # ==========================================
    # 2. PURIFY THE AUDIO (Demucs)
    # ==========================================
    print("2️⃣ Isolating vocals to prevent AI hallucinations...")
    subprocess.run(
        ["python", "-m", "demucs.separate", "-n", "htdemucs", "--two-stems", "vocals", "-d", "cuda", "-o", demucs_out_dir, raw_audio_path], 
        check=True
    )

    clean_vocal_path = glob.glob(f"{demucs_out_dir}/htdemucs/raw_audio/vocals.*")[0]
    shutil.copy(clean_vocal_path, f"{data_dir}/clean_vocals.wav")

    # ==========================================
    # 3. DIARIZATION & TRANSCRIPTION (Deployed API)
    # ==========================================
    print("3️⃣ Calling deployed diarization & transcription API...")
    
    with open(clean_vocal_path, 'rb') as audio_file:
        files = {'file': (os.path.basename(clean_vocal_path), audio_file, 'audio/wav')}
        data = {
            'num_speakers': '5',
            'language': 'eng'
        }
        
        try:
            response = requests.post(DIARIZATION_API_URL, files=files, data=data)
            response.raise_for_status()
            api_result = response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error calling diarization API: {e}")
            raise HTTPException(status_code=500, detail=f"Diarization API error: {e}")

    # Extract diarization segments (includes speaker, start, end, text)
    diarization_segments = api_result.get("diarization", [])
    
    # ==========================================
    # 4. REFERENCE EXTRACTION (Pydub)
    # ==========================================
    print("4️⃣ Building dynamic voice references...")
    clean_vocals = AudioSegment.from_file(clean_vocal_path)
    
    # Get unique speakers from diarization result
    unique_speakers = list(set([seg["speaker"] for seg in diarization_segments]))

    for target_speaker in unique_speakers:
        longest_duration = 0
        best_start = 0
        best_end = 0
        
        for seg in diarization_segments:
            if seg["speaker"] == target_speaker:
                duration = seg["end"] - seg["start"]
                if duration > longest_duration:
                    longest_duration = duration
                    best_start = seg["start"]
                    best_end = seg["end"]
        
        start_ms = int(best_start * 1000)
        end_ms = int(best_end * 1000)
        
        if (end_ms - start_ms) > 6000:
            end_ms = start_ms + 6000
            
        ref_filename = f"{data_dir}/ref_{target_speaker}.wav"
        clean_vocals[start_ms:end_ms].export(ref_filename, format="wav")

    # ==========================================
    # 5. BUILD MASTER TRANSCRIPT FROM API RESPONSE
    # ==========================================
    print("5️⃣ Building master transcript from API segments...")
    
    final_pipeline_data = []
    
    # Use API segments directly (each contains speaker, start, end, text at segment level)
    for seg in diarization_segments:
        segment_data = {
            "speaker": seg["speaker"],
            "start": seg["start"],
            "end": seg["end"],
            "text": seg.get("text", "").strip()
        }
        final_pipeline_data.append(segment_data)
        print(f"[{seg['start']}s -> {seg['end']}s] {seg['speaker']}: {seg.get('text', '')}")

    print(f"Total segments: {len(final_pipeline_data)}")

    # ==========================================
    # 6. SAVE MASTER HANDOFF FILE
    # ==========================================
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(final_pipeline_data, f, indent=4, ensure_ascii=False)

    print(f"\n Service A Complete! Saved {final_json_path}")
    return {"status": "success", "file_ready": "final_master_transcript.json"}
