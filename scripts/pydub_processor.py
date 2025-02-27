"""
Audio processing script using PyDub for high-level audio operations.
"""
import os
from pydub import AudioSegment
from pydub.effects import normalize, speedup
from dotenv import load_dotenv

load_dotenv()

def enhance_audio(input_path, output_dir="output"):
    """
    Enhance and modify audio file using PyDub with various effects.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the audio file
    print(f"Loading audio file: {input_path}")
    audio = AudioSegment.from_wav(input_path)
    print(f"Duration: {len(audio)/1000:.2f} seconds")
    print(f"Channels: {audio.channels}")
    print(f"Sample Width: {audio.sample_width} bytes")
    print(f"Frame Rate: {audio.frame_rate} Hz")
    
    # 1. Normalize audio
    normalized = normalize(audio)
    normalized_path = os.path.join(output_dir, "normalized.wav")
    normalized.export(normalized_path, format="wav")
    print(f"Saved normalized audio to: {normalized_path}")
    
    # 2. Add fade in/out
    faded = audio.fade_in(2000).fade_out(2000)  # 2 seconds each
    faded_path = os.path.join(output_dir, "faded.wav")
    faded.export(faded_path, format="wav")
    print(f"Saved faded audio to: {faded_path}")
    
    # 3. Speed up
    fast = speedup(audio, playback_speed=1.5)
    fast_path = os.path.join(output_dir, "fast.wav")
    fast.export(fast_path, format="wav")
    print(f"Saved speed-up audio to: {fast_path}")
    
    # 4. Lower volume
    quiet = audio - 10  # Reduce by 10 dB
    quiet_path = os.path.join(output_dir, "quiet.wav")
    quiet.export(quiet_path, format="wav")
    print(f"Saved quieter audio to: {quiet_path}")
    
    # 5. Create a mix of effects
    mixed = (
        normalize(audio)  # First normalize
        .fade_in(1000)   # Add 1s fade in
        .fade_out(1000)  # Add 1s fade out
        + 5              # Increase volume by 5 dB
    )
    mixed_path = os.path.join(output_dir, "mixed_effects.wav")
    mixed.export(mixed_path, format="wav")
    print(f"Saved mixed effects audio to: {mixed_path}")
    
    # 6. Split and combine segments
    if len(audio) > 10000:  # If longer than 10 seconds
        # Take first 5 seconds and last 5 seconds
        first_5 = audio[:5000]
        last_5 = audio[-5000:]
        # Combine them with a crossfade
        combined = first_5.append(last_5, crossfade=1000)
        combined_path = os.path.join(output_dir, "combined.wav")
        combined.export(combined_path, format="wav")
        print(f"Saved combined segments to: {combined_path}")
    
    return {
        "duration": len(audio)/1000,
        "channels": audio.channels,
        "sample_width": audio.sample_width,
        "frame_rate": audio.frame_rate,
        "output_files": [
            normalized_path,
            faded_path,
            fast_path,
            quiet_path,
            mixed_path
        ]
    }

if __name__ == "__main__":
    audio_path = "sample_audio.wav"  # Replace with your audio file
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        exit(1)
        
    results = enhance_audio(audio_path)
    print("\nEnhancement Results:")
    for key, value in results.items():
        print(f"{key}: {value}") 