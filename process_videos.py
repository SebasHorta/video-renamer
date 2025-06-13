import os
import subprocess
import whisper
from datetime import datetime
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import moviepy as mp
from tqdm import tqdm
from dateutil import parser
import shutil
import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"

VIDEO_DIR = "videos"
OUTPUT_DIR = "output"
LOG_FILE = "rename_log.csv"

# Load models
whisper_model = whisper.load_model("tiny")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model.to(device)

transcript_cache = {}

prompts = [
    "a person getting a haircut", "a wedding", "a birthday party", "a beach",
    "a baby", "a dog", "a person working out", "a drone shot", "nothing interesting"
]

keywords = {
    "wedding": "a wedding",
    "birthday": "a birthday party",
    "beach": "a beach",
    "dog": "a dog",
    "workout": "a person working out",
    "drone": "a drone shot",
    "baby": "a baby",
    "haircut": "a person getting a haircut"
}

def check_ffmpeg_installed():
    for tool in ['ffmpeg', 'ffprobe']:
        if not shutil.which(tool):
            raise EnvironmentError(f"'{tool}' not found. Install FFmpeg: https://ffmpeg.org/download.html")

def get_video_date(filepath):
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "format_tags=creation_time", "-of",
        "default=noprint_wrappers=1:nokey=1", filepath
    ]
    try:
        output = subprocess.check_output(cmd).decode().strip()
        if output:
            dt = parser.parse(output)
            return dt.strftime("%m-%d-%y")
    except Exception:
        pass
    mtime = os.path.getmtime(filepath)
    return datetime.fromtimestamp(mtime).strftime("%m-%d-%y")

def get_video_location(filepath):
    try:
        output = subprocess.check_output(["exiftool", "-GPSLatitude", "-GPSLongitude", filepath]).decode()
        lat, lon = None, None
        for line in output.splitlines():
            if "GPS Latitude" in line and not lat:
                lat = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])
            elif "GPS Longitude" in line and not lon:
                lon = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])

        if lat is not None and lon is not None:
            from geopy.geocoders import Nominatim
            geolocator = Nominatim(user_agent="video-renamer")
            location = geolocator.reverse((lat, lon), timeout=10)
            if location:
                return location.address.split(",")[0]  # First part of address (usually place name)
    except Exception as e:
        print(f"Could not get location for {filepath}: {e}")
    return None

def extract_frame(video_path, timestamp=2.0):
    with mp.VideoFileClip(video_path) as clip:
        frame = clip.get_frame(timestamp)
    return Image.fromarray(frame)

def describe_frame(img):
    inputs = clip_processor(text=prompts, images=img, return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    best = torch.argmax(probs, dim=1)
    return prompts[best.item()]

def transcribe_audio(video_path):
    if video_path in transcript_cache:
        return transcript_cache[video_path]
    result = whisper_model.transcribe(video_path, language="en", fp16=False, task="transcribe", word_timestamps=False)
    text = result.get("text", "").strip()
    transcript_cache[video_path] = text
    return text

def extract_context_around_keyword(transcript, keyword):
    sentences = re.split(r'[.?!]', transcript)
    for sentence in sentences:
        if keyword in sentence.lower():
            cleaned = sentence.strip().capitalize()
            if len(cleaned) > 80:
                cleaned = cleaned[:77] + "..."
            return cleaned
    return keyword.capitalize()

def describe_frame_or_audio(img, video_path):
    visual_label = describe_frame(img).strip().capitalize()

    if visual_label == "Nothing interesting":
        transcript = transcribe_audio(video_path)
        for kw in keywords.keys():
            if kw in transcript.lower():
                context = extract_context_around_keyword(transcript, kw)
                return "Unidentified scene", context.strip().rstrip("."), True
        return "Unidentified scene", "", False

    # Try to enrich with audio even if visual label exists
    transcript = transcribe_audio(video_path)
    for kw in keywords.keys():
        if kw in transcript.lower():
            context = extract_context_around_keyword(transcript, kw)
            return visual_label, context.strip().rstrip("."), True

    return visual_label, "", False


def rename_file(old_path, label, date, log_file, extra_desc="", location=None):
    base = os.path.basename(old_path)

    # Sanitize and format
    label_clean = re.sub(r'[\\/*?:"<>|]', '', label).strip().capitalize()
    extra_clean = re.sub(r'[\\/*?:"<>|]', '', extra_desc).strip().capitalize()
    location_clean = re.sub(r'[\\/*?:"<>|]', '', location).strip().title() if location else None

    if extra_clean and extra_clean.lower() != label_clean.lower():
        label_clean += f" ({extra_clean})"

    parts = [label_clean]
    if location_clean:
        parts.append(location_clean)

    new_name = " - ".join(parts + [date]) + ".mp4"
    new_path = os.path.join(OUTPUT_DIR, new_name)

    os.rename(old_path, new_path)
    with open(log_file, "a") as f:
        f.write(f"{new_path},{base}\n")

    print(f"Renamed: {base} → {new_name}")


def revert_renames(log_file):
    with open(log_file, "r") as f:
        for line in f:
            new_path, old_name = line.strip().split(",")
            old_path = os.path.join(VIDEO_DIR, old_name)
            if os.path.exists(new_path):
                os.rename(new_path, old_path)
                print(f"Reverted: {new_path} → {old_name}")
            else:
                print(f"File not found, can't revert: {new_path}")

def main(rename=True, batch_size=5):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if rename:
        with open(LOG_FILE, "w") as f:
            pass
        video_files = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(".mp4")]

        for i in range(0, len(video_files), batch_size):
            batch = video_files[i:i + batch_size]
            for filename in tqdm(batch, desc=f"Processing batch {i // batch_size + 1}"):
                path = os.path.join(VIDEO_DIR, filename)
                try:
                    date = get_video_date(path)
                    location = get_video_location(path)
                    img = extract_frame(path)
                    label, context, from_audio = describe_frame_or_audio(img, path)
                    extra_desc = context if from_audio and context else ""
                    rename_file(path, label, date, LOG_FILE, extra_desc, location)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
    else:
        revert_renames(LOG_FILE)

def can_rename():
    return any(f.lower().endswith(".mp4") for f in os.listdir(VIDEO_DIR))

def can_revert():
    return len(os.listdir(OUTPUT_DIR)) > 0

def cli():
    check_ffmpeg_installed()
    print("What do you want to do?")
    print("1) Rename videos")
    print("2) Revert renames")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        if not can_rename():
            print(f"No .MP4 files found in '{VIDEO_DIR}' folder to rename. Aborting.")
            return
        main(rename=True)
    elif choice == "2":
        if not can_revert():
            print(f"No files found in '{OUTPUT_DIR}' folder to revert. Aborting.")
            return
        main(rename=False)
    else:
        print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    cli()
