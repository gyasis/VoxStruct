"""
Audio processing script using Librosa for advanced audio analysis features.
"""
import os
import librosa
import numpy as np
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()

def analyze_audio(audio_path):
    """
    Analyze audio file using Librosa for various audio features.
    """
    # Load the audio file
    print(f"Loading audio file: {audio_path}")
    y, sr = librosa.load(audio_path)
    
    # Get duration
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"Duration: {duration:.2f} seconds")
    
    # Extract features
    # 1. Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    print(f"Estimated tempo: {tempo:.2f} BPM")
    
    # 2. Spectral Centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    print(f"Average spectral centroid: {np.mean(spectral_centroids):.2f} Hz")
    
    # 3. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    print(f"Average zero crossing rate: {np.mean(zcr):.4f}")
    
    # 4. MFCC (Mel-frequency cepstral coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    print(f"MFCC shape: {mfccs.shape}")
    
    # Generate and save visualizations
    plt.figure(figsize=(12, 8))
    
    # Waveform
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    
    # Mel Spectrogram
    plt.subplot(3, 1, 2)
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
    librosa.display.specshow(mel_spect_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    
    # MFCC
    plt.subplot(3, 1, 3)
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    
    # Save plot
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "audio_analysis.png"))
    plt.close()
    
    return {
        "duration": duration,
        "tempo": tempo,
        "spectral_centroid_mean": np.mean(spectral_centroids),
        "zero_crossing_rate_mean": np.mean(zcr),
        "mfcc_shape": mfccs.shape
    }

if __name__ == "__main__":
    audio_path = "sample_audio.wav"  # Replace with your audio file
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        exit(1)
        
    results = analyze_audio(audio_path)
    print("\nAnalysis Results:")
    for key, value in results.items():
        print(f"{key}: {value}") 