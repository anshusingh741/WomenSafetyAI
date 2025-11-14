from pydub import AudioSegment

# Load the existing alert.wav (float32)
sound = AudioSegment.from_file("alert.wav", format="wav")

# Export as PCM signed 16-bit WAV
sound.export("alert_fixed.wav", format="wav", parameters=["-acodec", "pcm_s16le"])

print("✅ Fixed alert sound saved as alert_fixed.wav")
