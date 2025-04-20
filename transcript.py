import os
import csv
import tempfile
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import whisper
import re

# Load Whisper model
whisper_model = whisper.load_model("base")

# Function to load profane words from a CSV file into a set for faster matching
def load_profane_words_from_csv(csv_path):
    profane_words = set()
    if os.path.exists(csv_path):
        with open(csv_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if 'Profanity' in row:
                    profane_words.add(row['Profanity'].strip().lower())  # Add words to set
        print(f"🔢 Profane words loaded from {csv_path}.")
    else:
        print(f"❌ CSV file not found: {csv_path}")
    return profane_words

# Function to check if text contains profane words based on the dataset
def contains_profane_words(text, profane_words):
    # Convert the text to lowercase for case-insensitive comparison
    text = text.lower()

    # Check for any profane words in the text
    for word in profane_words:
        if re.search(r'\b' + re.escape(word) + r'\b', text):  # Word boundary to match whole words
            return True
    return False

# Extract audio from video
def extract_audio_from_video(video_path, output_audio_path):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(output_audio_path, logger=None)

# Split audio into 5-second chunks
def split_audio(audio_path, chunk_length_ms=5000):
    audio = AudioSegment.from_file(audio_path)
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunks.append((i, chunk))
    return chunks

# Transcribe audio and label segments based on profanity words
def transcribe_and_label_chunks(chunks, profane_words, language):
    transcriptions = []

    for start_ms, chunk in chunks:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio_file:
            temp_audio_path = temp_audio_file.name

        # Export chunk and run transcription
        chunk.export(temp_audio_path, format="mp3")
        result = whisper_model.transcribe(temp_audio_path, language=language)

        # Get text and label based on profane words
        text = result['text'].strip()
        label = 1 if contains_profane_words(text, profane_words) else 0

        transcriptions.append({
            "segment": f"{start_ms / 1000:.1f}s - {(start_ms + len(chunk)) / 1000:.1f}s",
            "text": text,
            "label": label
        })

        # Remove temp file
        os.remove(temp_audio_path)

    return transcriptions

# Save results to CSV with proper handling of file write permissions
def save_to_csv(transcriptions, output_file):
    try:
        # Ensure the directory exists and has the right permissions
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Try writing the output to the specified CSV file
        with open(output_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=['segment', 'text', 'label'])
            writer.writeheader()
            for row in transcriptions:
                writer.writerow(row)
        print(f"💾 Results saved to {output_file}")
    
    except PermissionError:
        print(f"❌ Permission denied! Could not write to {output_file}. Try a different location.")
    except Exception as e:
        print(f"❌ Error: {e}")

# Main function
def main(video_path, output_csv, profanity_csv, language):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"❌ Video file not found: {video_path}")

    # Load the CSV of profane words for the selected language
    profane_words = load_profane_words_from_csv(profanity_csv)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio_file:
        print("🔊 Extracting audio from video...")
        extract_audio_from_video(video_path, temp_audio_file.name)

        print("✂️ Splitting audio into 5-second chunks...")
        chunks = split_audio(temp_audio_file.name)

        print("🧠 Transcribing & detecting profanity...")
        results = transcribe_and_label_chunks(chunks, profane_words, language)

        print("💾 Saving results to CSV...")
        save_to_csv(results, output_csv)

        print("✅ Done! Check your CSV file.")

if __name__ == "__main__":
    print("📂 Please enter the full path to your video file (e.g. D:/FamWatch/test_video.mp4):")
    video_file = input("▶️ Video file path: ").strip()
    
    # Specify the directory where the CSV will be saved
    output_csv = "output/transcription_toxic_labels.csv"
    
    # Create the output folder if it doesn't exist
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Ask user for the path to the profanity CSV file
    print("📄 Please provide the path to the CSV file with profane words (containing a column named 'Profanity'):")
    profanity_csv_path = input("▶️ Profanity CSV file path: ").strip()

    # Ask user for the language selection
    print("🌍 Select the language of the transcription:")
    print("1: English")
    print("2: Spanish")
    print("3: German")
    language_choice = input("▶️ Choose language (1/2/3): ").strip()

    # Map the language choice to the language code for Whisper
    if language_choice == "1":
        language = "en"
    elif language_choice == "2":
        language = "es"
    elif language_choice == "3":
        language = "de"
    else:
        print("❌ Invalid selection. Defaulting to English.")
        language = "en"

    # Run the main function
    main(video_file, output_csv, profanity_csv_path, language)
