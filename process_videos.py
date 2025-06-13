import os
import subprocess
import whisper
from datetime import datetime
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import moviepy as mp
from tqdm import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "false"

VIDEO_DIR = "videos"
OUTPUT_DIR = "output"
LOG_FILE = "rename_log.csv"  # Log file path

# Load models
whisper_model = whisper.load_model("base")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model.to(device)

def get_video_date(filepath):
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "format_tags=creation_time", "-of", "default=noprint_wrappers=1:nokey=1", filepath
    ]
    try:
        output = subprocess.check_output(cmd).decode().strip()
        print(f"Metadata creation_time for {filepath}: '{output}'")  # Debug line
        if output:
            date = datetime.fromisoformat(output.replace("Z", "")).strftime("%m-%d-%y")
            return date
        else:
            return "unknown-date"
    except Exception as e:
        print(f"Error reading date from {filepath}: {e}")
        return "unknown-date"

def extract_frame(video_path, timestamp=2.0):
    clip = mp.VideoFileClip(video_path)
    frame = clip.get_frame(timestamp)
    clip.close()
    return Image.fromarray(frame)

def describe_frame(img):
    prompts = [
        "a person getting a haircut", "a wedding", "a birthday party", "a beach",
        "a baby", "a dog", "a person working out", "a drone shot", "nothing interesting"
    ]
    inputs = clip_processor(text=prompts, images=img, return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    best = torch.argmax(probs, dim=1)
    return prompts[best.item()]

def rename_file(old_path, label, date, log_file):
    base = os.path.basename(old_path)
    new_name = f"{label.capitalize()} - {date}.mp4"
    new_path = os.path.join(OUTPUT_DIR, new_name)
    os.rename(old_path, new_path)
    # Save the mapping for reverting
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

def main(rename=True):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if rename:
        # Clear log file before starting new renames
        with open(LOG_FILE, "w") as f:
            pass
        video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".MP4")]

        for filename in tqdm(video_files, desc="Processing"):
            path = os.path.join(VIDEO_DIR, filename)
            date = get_video_date(path)
            img = extract_frame(path)
            label = describe_frame(img)
            rename_file(path, label, date, LOG_FILE)

    else:
        # Run revert process
        revert_renames(LOG_FILE)

def can_rename():
    return any(f.endswith(".MP4") for f in os.listdir(VIDEO_DIR))

def can_revert():
    return len(os.listdir(OUTPUT_DIR)) > 0

def cli():
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
