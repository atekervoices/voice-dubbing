import gradio as gr
import os
os.environ["COQUI_TOS_AGREED"] = "1"
import json
import glob
import subprocess
import warnings
import torch
import shutil
import gc 
import re
import sys
import requests

# --- Python 3.12 `imp` Module Shim for Legacy Libraries ---
if "imp" not in sys.modules:
    import importlib.util
    class MockImp:
        def reload(self, module):
            import importlib
            return importlib.reload(module)
        def load_source(self, name, pathname):
            spec = importlib.util.spec_from_file_location(name, pathname)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            return mod
    sys.modules["imp"] = MockImp()

# --- Numpy 2.x Backward Compatibility Shim ---
try:
    import numpy as np
    if not hasattr(np, 'float'): np.float = float
    if not hasattr(np, 'int'): np.int = int
    if not hasattr(np, 'bool'): np.bool = bool
    if not hasattr(np, 'complex'): np.complex = complex
    if not hasattr(np, 'object'): np.object = object
except ImportError:
    pass

from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from huggingface_hub import snapshot_download

# Deployed diarization endpoint (includes transcription)
DIARIZATION_API_URL = "https://shad-top-ideally.ngrok-free.app/diarize"

# Deployed TTS endpoint (voice cloning)
TTS_API_URL = "https://laurine-unappropriable-unvolcanically.ngrok-free.app/v1/audio/speech/clone/upload"

# ==========================================
# OpenAI-compatible translation config
# ==========================================
# Set OPENAI_API_KEY in your environment (or docker-compose env) to enable translation.
# The base URL and model are also configurable so you can point at OpenRouter, Groq,
# Together, or any local llama.cpp server that speaks the /v1/chat/completions schema.
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com").rstrip("/")
OPENAI_MODEL    = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

warnings.filterwarnings("ignore")
os.makedirs("./data", exist_ok=True)

# --- The Notebook-Safe PyTorch Fix ---
if not hasattr(torch, "_is_patched"):
    _original_load = torch.load
    def _unlocked_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_load(*args, **kwargs)
    torch.load = _unlocked_load
    torch._is_patched = True  

try:
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
except AttributeError:
    pass

# --- Python 3.11 / 3.12 coqpit Generic Type Fix ---
try:
    import coqpit.coqpit as _coqpit_module
    import builtins as _builtins

    _orig_issubclass = _builtins.issubclass
    def _safe_issubclass(cls, classinfo):
        try:
            return _orig_issubclass(cls, classinfo)
        except TypeError:
            return False
    _coqpit_module.issubclass = _safe_issubclass

    _orig_deserialize = _coqpit_module._deserialize
    def _safe_deserialize(x, field_type):
        try:
            return _orig_deserialize(x, field_type)
        except TypeError:
            return x
        except ValueError as e:
            if "does not match" in str(e):
                return x
            raise e
    _coqpit_module._deserialize = _safe_deserialize
except Exception:
    pass

# ==========================================
# PRE-DOWNLOAD DIRECTLY TO DISK ON BOOT
# ==========================================
print("Caching Demucs Audio Separation Models...")
try:
    import demucs.pretrained
    demucs.pretrained.get_model('htdemucs')
except Exception as e:
    print(f"Demucs cache failed: {e}")

print("Models securely cached on your hard drive.")

LANGUAGE_MAP = [
    "English",
    "Luganda",
    "Acholi",
    "Lugbara",
    "Runyankole",
    "Ateso",
]

# Friendly UI name -> ISO code that GPT models recognize
TARGET_LANG_CODE = {
    "English":    "English",
    "Luganda":    "Luganda",
    "Acholi":     "Acholi",
    "Lugbara":    "Lugbara",
    "Runyankole": "Runyankole (Nkore)",
    "Ateso":      "Ateso (Atesot)",
}

def run_extraction(video_filename: str, num_speakers: int = 3):
    print(f"\nPhase 1: Extraction & Diarization for {video_filename} (num_speakers={num_speakers})")
    data_dir = "./data"
    video_path = f"{data_dir}/{video_filename}"
    raw_audio_path = f"{data_dir}/raw_audio.wav"
    demucs_out_dir = f"{data_dir}/separated"
    final_json_path = f"{data_dir}/final_master_transcript.json"

    print("Extracting raw audio...")
    os.system(f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {raw_audio_path} -y -loglevel error")

    print("Isolating vocals...")
    subprocess.run(["python", "-m", "demucs.separate", "-n", "htdemucs", "--two-stems", "vocals", "-d", "cuda", "-o", demucs_out_dir, raw_audio_path], check=True)
    clean_vocal_path = glob.glob(f"{demucs_out_dir}/htdemucs/raw_audio/vocals.*")[0]
    shutil.copy(clean_vocal_path, f"{data_dir}/clean_vocals.wav")

    print("Calling deployed diarization & transcription API...")
    
    with open(clean_vocal_path, 'rb') as audio_file:
        files = {'file': (os.path.basename(clean_vocal_path), audio_file, 'audio/wav')}
        data = {
            'num_speakers': str(int(num_speakers)),
            'language': 'eng'
        }
        
        try:
            response = requests.post(DIARIZATION_API_URL, files=files, data=data)
            response.raise_for_status()
            api_result = response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error calling diarization API: {e}")
            raise Exception(f"Diarization API error: {e}")

    diarization_segments = api_result.get("diarization", [])
    
    print("Building dynamic voice references...")
    clean_vocals = AudioSegment.from_file(clean_vocal_path)
    unique_speakers = list(set([seg["speaker"] for seg in diarization_segments]))
    
    for target_speaker in unique_speakers:
        longest_duration = 0
        best_start, best_end = 0, 0
        
        for seg in diarization_segments:
            if seg["speaker"] == target_speaker:
                duration = seg["end"] - seg["start"]
                if duration > longest_duration:
                    longest_duration = duration
                    best_start, best_end = seg["start"], seg["end"]
        
        start_ms, end_ms = int(best_start * 1000), int(best_end * 1000)
        if (end_ms - start_ms) > 6000: end_ms = start_ms + 6000
        ref_filename = f"{data_dir}/ref_{target_speaker}.wav"
        clean_vocals[start_ms:end_ms].export(ref_filename, format="wav")

    print("Building master transcript from API segments...")
    final_pipeline_data = []
    
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
    
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(final_pipeline_data, f, indent=4, ensure_ascii=False)
    
    return final_json_path

def _openai_translate_segment(text: str, target_language: str) -> str:
    """Translate a single segment via OpenAI-compatible /v1/chat/completions."""
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your docker-compose.yml environment "
            "section (e.g. - OPENAI_API_KEY=sk-...) and rebuild."
        )
    url = f"{OPENAI_BASE_URL}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type":  "application/json",
    }
    system_prompt = (
        "You are a professional dialogue translator for movie dubbing. "
        f"Translate the user's text from English into {target_language}. "
        "Preserve speaker intent, emotion, and natural conversational tone. "
        "Output ONLY the translated text with no quotes, no explanation, no prefix."
    )
    payload = {
        "model": OPENAI_MODEL,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": text},
        ],
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()

def run_translation(transcript_filepath: str, target_lang_name: str):
    print(f"\nPhase 2: Translation via OpenAI ({OPENAI_MODEL}) -> {target_lang_name}...")
    data_dir = "./data"
    output_filename = f"{data_dir}/translated_transcript_fixed.json"

    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Set it in docker-compose.yml (e.g. "
            "`environment: - OPENAI_API_KEY=sk-...`) and rebuild the container."
        )

    target_language = TARGET_LANG_CODE.get(target_lang_name, target_lang_name)

    with open(transcript_filepath, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)

    for entry in transcript_data:
        if "speaker" not in entry:
            entry["speaker"] = "UNKNOWN_SPEAKER"

    print(f"Translating {len(transcript_data)} segments...")
    translated_data = []
    for idx, segment in enumerate(transcript_data, 1):
        original_text = segment.get("text", "").strip()
        clean_text = original_text.replace("♪", "").strip()
        if not clean_text:
            translated_data.append({
                "speaker": segment["speaker"],
                "start": segment["start"],
                "end":   segment["end"],
                "english_original": original_text,
                "translated_text":  "",
            })
            continue

        try:
            translated_text = _openai_translate_segment(clean_text, target_language)
        except Exception as e:
            print(f"[!] Translation failed on segment {idx}: {e}")
            translated_text = clean_text  # graceful fallback so pipeline keeps moving

        print(f"[{idx}/{len(transcript_data)}] [{segment['start']}s] {segment['speaker']}: {translated_text}")
        translated_data.append({
            "speaker":           segment["speaker"],
            "start":             segment["start"],
            "end":               segment["end"],
            "english_original":  original_text,
            "translated_text":   translated_text,
        })

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(translated_data, f, indent=4, ensure_ascii=False)

    print(f" Translation Complete. Saved to {output_filename}")
    return output_filename

def format_timestamp(seconds: float) -> str:
    ms = int((seconds % 1) * 1000)
    s = int(seconds)
    m = s // 60
    h = m // 60
    return f"{h:02d}:{m%60:02d}:{s%60:02d},{ms:03d}"

def process_pipeline(youtube_url, uploaded_video_path, target_lang_name, burn_subtitles, num_speakers,
                     whisper_model_choice="large-v3", xtts_temperature=0.65, max_atempo=1.5,
                     silence_floor=-38, min_slot_ms=50,
                     normalize_audio=True, target_volume=-16.0, noise_reduction=15, high_pass=80, low_pass=10000):
    yield "Initializing Pipeline...", None

    data_dir = "./data"
    
    yield "Purging old video data (Safeguarding models)...", None
    for item in os.listdir(data_dir):
        if item == "models_cache":
            continue
        item_path = os.path.join(data_dir, item)
        try:
            if os.path.isdir(item_path):
                shutil.rmtree(item_path, ignore_errors=True)
            else:
                os.remove(item_path)
        except Exception:
            pass
            
    shared_video_path = f"{data_dir}/source_video.mp4"
    
    is_yt = isinstance(youtube_url, str) and youtube_url.strip() != ""
    if is_yt:
        yield "Fetching video straight from YouTube...", None
        if os.path.exists(shared_video_path): os.remove(shared_video_path)
        safe_url = youtube_url.strip().replace('"', '\\"')
        os.system(f"yt-dlp -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4' \"{safe_url}\" -o {shared_video_path} -N 4")
        if not os.path.exists(shared_video_path):
            yield "FAILED: Failed to download from YouTube. Check URL.", None
            return
    elif uploaded_video_path is not None and str(uploaded_video_path).strip() != "":
        yield "Saving uploaded video to shared volume...", None
        try:
            shutil.copy(str(uploaded_video_path), shared_video_path)
        except Exception as e:
            yield f"FAILED: Copy uploaded video error: {e}", None
            return
    else:
        yield "FAILED: No video provided! Ignoring request.", None
        return
        
    yield "Phase 1: High-Fidelity Audio Extraction...", None
    try:
        transcript_path = run_extraction("source_video.mp4", num_speakers=num_speakers)
    except Exception as e:
        yield f"FAILED: Processing error during extraction: {e}", None
        return
    
    yield f"Phase 2: Neural Translation to {target_lang_name}...", None
    try:
        translated_json_path = run_translation(transcript_path, target_lang_name)
    except Exception as e:
        yield f"FAILED: Translation error: {e}", None
        return
    
    yield "Phase 3: Auto-Syncing & Cloning Voices...", None
    
    with open(translated_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    subs_path = f"{data_dir}/subs.srt"
    if burn_subtitles:
        with open(subs_path, "w", encoding="utf-8") as srt_file:
            for idx, line in enumerate(data):
                start_str = format_timestamp(line["start"])
                end_str = format_timestamp(line["end"])
                srt_file.write(f"{idx+1}\n{start_str} --> {end_str}\n{line['translated_text']}\n\n")

    base_audio_path = f"{data_dir}/base_audio.wav"
    os.system(f"/usr/bin/ffmpeg -i {shared_video_path} -vn -acodec pcm_s16le -ar 22050 -ac 1 {base_audio_path} -y -loglevel error")
    base_audio = AudioSegment.from_wav(base_audio_path)

    subprocess.run(["demucs", "-d", "cuda", "--two-stems", "vocals", "-o", data_dir, base_audio_path], check=True)
    vocal_stem_path = f"{data_dir}/htdemucs/base_audio/vocals.wav"
    if not os.path.exists(vocal_stem_path):
        vocal_stem_path = glob.glob(f"{data_dir}/*/base_audio/vocals.wav")[0]
    clean_vocals = AudioSegment.from_wav(vocal_stem_path)

    refs = {}
    for line in data:
        spk = line["speaker"]
        duration = line["end"] - line["start"]
        if spk not in refs: refs[spk] = {"start": line["start"], "end": line["end"], "duration": duration}
        else:
            if abs(duration - 6.0) < abs(refs[spk]["duration"] - 6.0):
                refs[spk] = {"start": line["start"], "end": line["end"], "duration": duration}

    for spk, info in refs.items():
        start_ms, end_ms = int(info["start"] * 1000), int(info["end"] * 1000)
        ref_path = f"{data_dir}/ref_{spk}.wav"
        clean_vocals[start_ms:end_ms].export(ref_path, format="wav")

    total_duration_ms = int(data[-1]["end"] * 1000) + 15000 
    master_dub_audio = AudioSegment.silent(duration=total_duration_ms)

    xtts_lang_code = "en"

    def generate_tts_via_api(text: str, reference_audio_path: str, language: str, temperature: float, output_path: str):
        """Generate TTS using the deployed API service."""
        files = {'reference_audio': open(reference_audio_path, 'rb')}
        data = {'text': text, 'temperature': temperature}
        try:
            response = requests.post(TTS_API_URL, files=files, data=data, stream=True)
            response.raise_for_status()
            raw_audio_data = bytearray()
            for chunk in response.iter_content(chunk_size=8192):
                raw_audio_data.extend(chunk)
            if not raw_audio_data:
                return False
            with wave.open(output_path, 'wb') as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
                wf.writeframes(raw_audio_data)
            return True
        except Exception as e:
            print(f"TTS API error: {e}")
            return False
        finally:
            files['reference_audio'].close()

    import wave

    for i, line in enumerate(data):
        yield f"Generating dialogue block {i+1}/{len(data)}...", None
        spk = line["speaker"]
        translated_text = line["translated_text"]
        actual_start_ms = int(line["start"] * 1000)
        current_ref = f"{data_dir}/ref_{spk}.wav"
        raw_path = f"{data_dir}/raw_line_{i}.wav"
        if not generate_tts_via_api(text=translated_text, reference_audio_path=current_ref, language=xtts_lang_code, temperature=xtts_temperature, output_path=raw_path):
            yield f"FAILED: TTS generation failed for block {i+1}", None
            return
        filter_chain = []
        if high_pass > 0: filter_chain.append(f"highpass=f={high_pass}")
        if low_pass < 15000: filter_chain.append(f"lowpass=f={low_pass}")
        if noise_reduction > 0: filter_chain.append(f"afftdn=nr={noise_reduction}")
        if filter_chain:
            filtered_path = f"{data_dir}/filtered_line_{i}.wav"
            af_val = ",".join(filter_chain)
            os.system(f"/usr/bin/ffmpeg -i {raw_path} -af \"{af_val}\" {filtered_path} -y -loglevel error")
            temp_audio = AudioSegment.from_wav(filtered_path)
        else:
            temp_audio = AudioSegment.from_wav(raw_path)
        nonsilent_ranges = detect_nonsilent(temp_audio, min_silence_len=50, silence_thresh=silence_floor)
        trimmed_audio = temp_audio[nonsilent_ranges[0][0]:nonsilent_ranges[-1][1]] if nonsilent_ranges else temp_audio
        if normalize_audio and trimmed_audio.dBFS != float('-inf'):
            change_in_dBFS = target_volume - trimmed_audio.dBFS
            if change_in_dBFS < 20.0:
                trimmed_audio = trimmed_audio.apply_gain(change_in_dBFS)
        if i + 1 < len(data):
            next_start_ms = int(data[i+1]["start"] * 1000)
            gap_duration_ms = (next_start_ms - actual_start_ms)
            max_safe_duration_ms = max(int(min_slot_ms), gap_duration_ms)
        else:
            max_safe_duration_ms = len(trimmed_audio) + 3000
        if len(trimmed_audio) > max_safe_duration_ms:
            speed_ratio = len(trimmed_audio) / max_safe_duration_ms
            safe_ratio = min(speed_ratio, max_atempo)
            temp_in = f"{data_dir}/temp_squish_in.wav"; temp_out = f"{data_dir}/temp_squish_out.wav"
            trimmed_audio.export(temp_in, format="wav")
            os.system(f"/usr/bin/ffmpeg -i {temp_in} -filter:a atempo={safe_ratio} {temp_out} -y -loglevel error")
            trimmed_audio = AudioSegment.from_wav(temp_out)
            if len(trimmed_audio) > max_safe_duration_ms:
                trimmed_audio = trimmed_audio[:max_safe_duration_ms].fade_out(100)
        trimmed_audio = trimmed_audio.fade_in(30).fade_out(30)
        master_dub_audio = master_dub_audio.overlay(trimmed_audio, position=actual_start_ms)

    master_dub_audio = master_dub_audio[:len(base_audio)]
    dub_only_path = f"{data_dir}/final_dub_SYNCED.wav"
    master_dub_audio.export(dub_only_path, format="wav")

    yield "Re-integrating Original Background Audio & FX...", None
    original_full_audio = f"{data_dir}/original_full_audio.wav"
    os.system(f"/usr/bin/ffmpeg -i {shared_video_path} -vn -acodec pcm_s16le -ar 44100 -ac 2 {original_full_audio} -y -loglevel error")
    subprocess.run(["demucs", "-d", "cuda", "--two-stems", "vocals", "-o", data_dir, original_full_audio], check=True)
    background_path = glob.glob(f"{data_dir}/htdemucs/original_full_audio/no_vocals.*")[0]

    yield "Performing Final Video Mix...", None
    final_video_output = f"{data_dir}/ULTIMATE_DUBBED_VIDEO.mp4"
    if burn_subtitles:
        subtitle_filter = f"-vf \"subtitles={subs_path}\""; video_codec = "-c:v libx264 -preset ultrafast"
    else:
        subtitle_filter = ""; video_codec = "-c:v copy"
    os.system(f"""/usr/bin/ffmpeg -i {shared_video_path} -i {dub_only_path} -i {background_path} -filter_complex "[1:a][2:a]amix=inputs=2:duration=first[aout]" -map 0:v -map "[aout]" {subtitle_filter} {video_codec} -c:a aac -b:a 192k {final_video_output} -y -loglevel error""")
    gc.collect(); torch.cuda.empty_cache()
    yield "PROJECT COMPLETE! Video is Ready.", final_video_output

# ==========================================
# MODERN PORTFOLIO VIBE UI
# ==========================================

custom_theme = gr.themes.Base(font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "ui-sans-serif", "system-ui", "sans-serif"], primary_hue="purple", secondary_hue="blue", neutral_hue="slate").set(
    body_background_fill="#0A0910", body_background_fill_dark="#0A0910",
    body_text_color="#FFFFFF", body_text_color_dark="#FFFFFF",
    background_fill_primary="#12111A", background_fill_primary_dark="#12111A",
    background_fill_secondary="#0A0910", background_fill_secondary_dark="#0A0910",
    border_color_primary="#1E1D2B", border_color_primary_dark="#1E1D2B",
    block_background_fill="#12111A", block_background_fill_dark="#12111A",
    block_border_width="1px",
    block_label_background_fill="#1A1826", block_label_background_fill_dark="#1A1826",
    block_label_text_color="#94A3B8", block_label_text_color_dark="#94A3B8",
    block_title_text_color="#FFFFFF", block_title_text_color_dark="#FFFFFF",
    input_background_fill="#1A1826", input_background_fill_dark="#1A1826",
    button_primary_background_fill="linear-gradient(90deg, #A855F7, #D946EF)",
    button_primary_background_fill_dark="linear-gradient(90deg, #A855F7, #D946EF)",
    button_primary_background_fill_hover="linear-gradient(90deg, #9333EA, #C026D3)",
    button_primary_background_fill_hover_dark="linear-gradient(90deg, #9333EA, #C026D3)",
    button_primary_text_color="#FFFFFF", button_primary_text_color_dark="#FFFFFF",
    button_secondary_background_fill="#1A1826", button_secondary_background_fill_dark="#1A1826",
    button_secondary_text_color="#FFFFFF", button_secondary_text_color_dark="#FFFFFF",
    slider_color="#A855F7", slider_color_dark="#A855F7")

css = open("./_inline_css.txt").read() if os.path.exists("./_inline_css.txt") else ""

with gr.Blocks(theme=custom_theme, css=css) as ui:
    gr.HTML('<div class="hero-section"><div class="hero-chip">AI Video Dubbing Studio</div><h1 class="hero-h1">Dub Any Video.<br><span>In Any Language.</span></h1><p class="hero-desc">Voice cloning, neural translation & automated lip-sync.</p></div>')
    with gr.Row(elem_classes="app-grid"):
        with gr.Column(scale=4, min_width=400):
            with gr.Column(elem_classes="card delay-1"):
                yt_url = gr.Textbox(label="YouTube URL", placeholder="https://youtube.com/watch?v=...", lines=1)
                gr.HTML("<div class='or-line'>or</div>")
                upload_btn = gr.UploadButton("Upload Local Video", file_types=["video"], elem_classes="upload-btn")
                yt_preview = gr.HTML(visible=False)
                vid_in = gr.Video(label="Preview Container", visible=False, interactive=False)
            with gr.Column(elem_classes="card delay-2"):
                with gr.Row():
                    target_lang = gr.Dropdown(choices=LANGUAGE_MAP, value="English", label="Target Language", interactive=True, filterable=False)
                    burn_subs = gr.Checkbox(label="Burn Subtitles", value=True)
                num_speakers = gr.Slider(minimum=1, maximum=10, value=2, step=1, label="Number of Speakers")
            with gr.Accordion("Advanced Engine Settings", open=False):
                whisper_choice = gr.Radio(["tiny", "base", "medium", "large-v2", "large-v3"], value="large-v3", label="Whisper Model")
                with gr.Row():
                    xtts_temp  = gr.Slider(0.1, 1.0, value=0.65, step=0.05, label="Voice Temperature")
                    atempo_max = gr.Slider(1.0, 3.0, value=1.5,  step=0.05, label="Max Speed-Up")
                with gr.Row():
                    sil_floor = gr.Slider(-60, -10, value=-38, step=1, label="Silence Floor (dBFS)")
                    min_slot  = gr.Slider(0, 1000, value=50, step=50, label="Min Gap (ms)")
                with gr.Row():
                    normalize_audio = gr.Checkbox(label="Normalize Audio (RMS)", value=True)
                    target_volume = gr.Slider(-35.0, -5.0, value=-16.0, step=1.0, label="Target Vol (dBFS)")
                    noise_reduction = gr.Slider(0, 50, value=15, step=1, label="Noise Reduction (dB)")
                with gr.Row():
                    high_pass = gr.Slider(0, 500, value=80, step=10, label="High-pass Filter (Hz)")
                    low_pass = gr.Slider(2000, 15000, value=10000, step=500, label="Low-pass Filter (Hz)")
            btn = gr.Button("Run Dubbing Pipeline", variant="primary", size="lg")
        with gr.Column(scale=6):
            with gr.Column(elem_classes="card delay-4"):
                status_out = gr.Textbox(lines=14, interactive=False, elem_id="terminal-log", show_label=False, placeholder="Waiting for pipeline to start...")
            with gr.Column(elem_classes="card delay-5"):
                vid_out = gr.Video(interactive=False, show_label=False, elem_classes="render-box")

    btn.click(fn=process_pipeline, inputs=[yt_url, vid_in, target_lang, burn_subs, num_speakers, whisper_choice, xtts_temp, atempo_max, sil_floor, min_slot, normalize_audio, target_volume, noise_reduction, high_pass, low_pass], outputs=[status_out, vid_out])

    def update_yt_preview(url):
        if not url: return gr.update(visible=False, value=""), gr.update(visible=False, value=None)
        match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
        if match:
            vid = match.group(1)
            iframe = f'<div style="margin-top:10px;"><iframe width="100%" height="200" src="https://www.youtube.com/embed/{vid}" frameborder="0" allowfullscreen style="border-radius:10px;border:1px solid #1E1B2E;"></iframe></div>'
            return gr.update(visible=True, value=iframe), gr.update(visible=False, value=None)
        return gr.update(visible=False, value=""), gr.update(visible=False, value=None)
    yt_url.change(fn=update_yt_preview, inputs=[yt_url], outputs=[yt_preview, vid_in])

    def handle_upload(filepath):
        if filepath is None: return gr.update(visible=False, value=None), gr.update(value="")
        val = getattr(filepath, "name", filepath)
        return gr.update(visible=True, value=val), gr.update(value="")
    upload_btn.upload(fn=handle_upload, inputs=[upload_btn], outputs=[vid_in, yt_url])

if __name__ == "__main__":
    ui.queue().launch(server_name="0.0.0.0", server_port=8060, share=True)
