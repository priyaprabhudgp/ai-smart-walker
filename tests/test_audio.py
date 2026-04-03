import os
import pyaudio
import sounddevice as sd
from scipy.io.wavfile import write


# Test for listing audio devices
def test_list_audio_inputs():
    p = pyaudio.PyAudio()
    device_count = p.get_device_count()
    print(f"Found {device_count} audio devices.")

    if device_count == 0:
        print("No audio devices found.")
    else:
        for i in range(device_count):
            device_info = p.get_device_info_by_index(i)
            print(f"Device {i}: {device_info['name']} - {device_info['maxInputChannels']} input channels")
    p.terminate()


# Test for recording audio
def test_record_audio():
    fs = 44100  # Sample rate
    seconds = 5  # Duration
    output_file = "test_output.wav"

    # Record audio
    print("Recording...")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished

    # Check if the recording has data
    assert myrecording is not None, "Recording failed. No audio data captured."
    print(f"Recording captured {myrecording.shape[0]} samples.")

    # Save to a file
    print("Saving...")
    write(output_file, fs, myrecording)
    print(f"Audio saved as {output_file}.")

    # Verify if the file exists and is a WAV file
    assert os.path.exists(output_file), f"File {output_file} was not saved successfully."
    assert output_file.endswith(".wav"), f"The saved file is not a WAV file: {output_file}"

    # Optionally check file size (it shouldn't be empty)
    file_size = os.path.getsize(output_file)
    assert file_size > 0, f"The file {output_file} is empty."

    print("Recording test passed.")


# Run the tests
if __name__ == "__main__":
    print("Testing audio device listing:")
    test_list_audio_inputs()

    print("\nTesting audio recording:")
    test_record_audio()
