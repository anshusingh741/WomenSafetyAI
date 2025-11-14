import numpy as np
from scipy.io.wavfile import write

# Parameters
samplerate = 44100  # CD quality
duration = 1.0      # seconds
frequency = 1000.0  # Hz (pitch of beep)

t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
audio = 0.5 * np.sin(2 * np.pi * frequency * t)

# Save as WAV
write("alert.wav", samplerate, audio.astype(np.float32))

print("✅ alert.wav generated successfully!")
