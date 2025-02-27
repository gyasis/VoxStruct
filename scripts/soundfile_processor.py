"""
Audio processing script using SoundFile for basic audio operations.
"""
import os
import soundfile as sf
import numpy as np
from dotenv import load_dotenv

load_dotenv()

def process_audio(input_path, output_dir="output"):
    """
    Process audio file using SoundFile with various operations.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the audio file
    print(f"Loading audio file: {input_path}")
    data, samplerate = sf.read(input_path)
    duration = len(data) / samplerate
    print(f"Sample rate: {samplerate} Hz")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Number of channels: {data.shape[1] if len(data.shape) > 1 else 1}")
    
    # 1. Save a reversed version
    reversed_path = os.path.join(output_dir, "reversed.wav")
    sf.write(reversed_path, np.flip(data, axis=0), samplerate)
    print(f"Saved reversed audio to: {reversed_path}")
    
    # 2. Change speed (by resampling)
    speed_up_path = os.path.join(output_dir, "speed_up.wav")
    sf.write(speed_up_path, data[::2], samplerate)  # Take every second sample
    print(f"Saved speed-up audio to: {speed_up_path}")
    
    # 3. Adjust volume
    volume_up_path = os.path.join(output_dir, "volume_up.wav")
    sf.write(volume_up_path, data * 1.5, samplerate)  # Increase volume by 50%
    print(f"Saved volume-increased audio to: {volume_up_path}")
    
    # 4. Create mono version if stereo
    if len(data.shape) > 1 and data.shape[1] > 1:
        mono_path = os.path.join(output_dir, "mono.wav")
        mono_data = np.mean(data, axis=1)
        sf.write(mono_path, mono_data, samplerate)
        print(f"Saved mono version to: {mono_path}")
    
    # 5. Create a fade in/out effect
    fade_samples = int(samplerate * 0.5)  # 0.5 second fade
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    
    faded_data = data.copy()
    if len(data.shape) > 1:  # Stereo
        faded_data[:fade_samples] *= fade_in[:, np.newaxis]
        faded_data[-fade_samples:] *= fade_out[:, np.newaxis]
    else:  # Mono
        faded_data[:fade_samples] *= fade_in
        faded_data[-fade_samples:] *= fade_out
    
    fade_path = os.path.join(output_dir, "fade_effect.wav")
    sf.write(fade_path, faded_data, samplerate)
    print(f"Saved version with fade effects to: {fade_path}")
    
    return {
        "samplerate": samplerate,
        "duration": duration,
        "channels": data.shape[1] if len(data.shape) > 1 else 1,
        "output_files": [
            reversed_path,
            speed_up_path,
            volume_up_path,
            fade_path
        ]
    }

if __name__ == "__main__":
    audio_path = "sample_audio.wav"  # Replace with your audio file
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        exit(1)
        
    results = process_audio(audio_path)
    print("\nProcessing Results:")
    for key, value in results.items():
        print(f"{key}: {value}") 