import torch
import time
import whisper
import pyaudio
import warnings
import numpy as np
from queue import Queue
from threading import Event
import sys
import threading
import select
import termios
import tty
import os

# Global warning filter
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
torch.set_warn_always(False)
NEW_TEXT_EVENT = Event()

class Config:
    # Use a larger model to improve multilingual recognition accuracy
    MODEL_SIZE = "medium"  # Best multilingual model (tiny/base/small/medium/large/large-v3)
    SAMPLE_RATE = 16000
    CHUNK = 1024
    LANGUAGE = None          # Auto-detect language
    TEMPERATURE = 0.0        # Lower temperature for more deterministic output (0.0–1.0)
    USE_CUDA = True          # Enable CUDA acceleration
    FP16 = True              # Use FP16 precision to accelerate inference
    MAX_RECORD_SECONDS = 10  # Maximum recording duration (seconds)
    
    # Multilingual recognition optimization parameters
    LANGUAGE_PRIORITY = ["de", "en", "zh"]  # Language priority: German, English, Chinese
    VOCABULARY = []  # Optional: Add specific vocabulary to improve recognition accuracy
    
    # Language-specific optimization parameters
    LANGUAGE_SPECIFIC_PARAMS = {
        "de": {"temperature": 0.1},  # Optimization for German
        "en": {"temperature": 0.1},  # Optimization for English
        "zh": {"temperature": 0.0}   # Optimization for Chinese
    }


class LinuxKeyListener:
    """Linux下替代keyboard库的按键监听器"""
    def __init__(self):
        self.running = False
        self.s_pressed = False
        self.old_settings = None
        
    def setup_terminal(self):
        if os.name == 'posix':  # Unix/Linux系统
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())
    
    def restore_terminal(self):
        if self.old_settings and os.name == 'posix':
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
    
    def start_listening(self, on_press_s, on_release_s):
        self.running = True
        self.setup_terminal()
        
        def listen_loop():
            try:
                while self.running:
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        key = sys.stdin.read(1)
                        if key.lower() == 's' and not self.s_pressed:
                            self.s_pressed = True
                            on_press_s()
                        elif key.lower() != 's' and self.s_pressed:
                            self.s_pressed = False
                            on_release_s()
                        elif key == '\x1b':  # ESC
                            self.running = False
                            break
                    elif self.s_pressed:
                        # 检查S键是否还在按下状态（简化处理）
                        pass
            except Exception as e:
                print(f"Key listener error: {e}")
            finally:
                self.restore_terminal()
        
        self.listen_thread = threading.Thread(target=listen_loop, daemon=True)
        self.listen_thread.start()
    
    def stop_listening(self):
        self.running = False
        self.restore_terminal()


class VoiceTranscriber:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.audio_queue = Queue()
        self.is_recording = Event()
        self.stream = None
        
        # Detect available device
        self.device = "cuda" if Config.USE_CUDA and torch.cuda.is_available() else "cpu"
        print(f"🔄 Loading multilingual model to {self.device.upper()}...")
        
        # Load model with device specification
        self.model = whisper.load_model(Config.MODEL_SIZE, device=self.device)
        
        # Enable FP16 if supported
        if self.device == "cuda" and Config.FP16:
            print("⚡ Using FP16 precision")
        
        print(f"✅ Model loaded (supports Chinese-English recognition)")

    def start_recording(self):
        if not self.is_recording.is_set():
            try:
                self.stream = self.p.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=Config.SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=Config.CHUNK,
                    stream_callback=self.audio_callback
                )
                self.stream.start_stream()
                self.is_recording.set()
                print("\n🔴 Recording...")
            except Exception as e:
                print(f"❌ Microphone error: {str(e)}")
                self.cleanup()
                sys.exit(1)

    def stop_recording(self):
        if self.is_recording.is_set():
            self.is_recording.clear()
            time.sleep(0.1)  # Buffer wait
            
            audio_data = self.process_audio()
            if audio_data is not None:
                self.transcribe(audio_data)
            
            self.stream.stop_stream()
            self.stream.close()
            print("⏹️ Stopped")

    def audio_callback(self, in_data, frame_count, time_info, status):
        if self.is_recording.is_set():
            self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)

    def process_audio(self):
        frames = []
        while not self.audio_queue.empty():
            frames.append(self.audio_queue.get())
        return np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0 if frames else None

    def transcribe(self, audio_data):
        try:
            # Use FP16 only when CUDA is available
            fp16 = Config.FP16 and self.device == "cuda"
            
            result = self.model.transcribe(
                audio_data,
                language=Config.LANGUAGE,
                temperature=Config.TEMPERATURE,
                task="transcribe",
                fp16=fp16  # Enable FP16 only on CUDA devices
            )
            text = result["text"].strip()
            print(f"\n📝 Result: {text}")
        
            OUTPUT_PATH = "src/transcribe/transcription.txt"
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                f.write(text)

            NEW_TEXT_EVENT.set()

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("❌ Not enough CUDA memory! Please try:\n"
                    "1. Using a smaller model (tiny/base/small)\n"
                    "2. Reducing the recording duration\n"
                    "3. Closing other programs that are using GPU memory")
            else:
                print(f"❌ Recognition failed: {str(e)}")
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")

    def cleanup(self):
        if self.stream and self.stream.is_active():
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
        # Cleanup GPU memory
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def auto_record_and_transcribe(self, duration: int):
        """
        Automatically record for a fixed duration (in seconds) and return the transcribed text.
        """
        try:
            stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=Config.SAMPLE_RATE,
                input=True,
                frames_per_buffer=Config.CHUNK
            )
            print(f"\n🔴 Auto-recording for {duration} seconds...")
            frames = []

            for _ in range(0, int(Config.SAMPLE_RATE / Config.CHUNK * duration)):
                data = stream.read(Config.CHUNK, exception_on_overflow=False)
                frames.append(data)

            stream.stop_stream()
            stream.close()
            print("⏹️ Auto-recording stopped.")

            audio_data = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0
            fp16 = Config.FP16 and self.device == "cuda"
            result = self.model.transcribe(
                audio_data,
                language=Config.LANGUAGE,
                temperature=Config.TEMPERATURE,
                task="transcribe",
                fp16=fp16
            )
            text = result["text"].strip()
            print(f"\n📝 Auto Transcription Result: {text}")
            return text

        except Exception as e:
            print(f"❌ Auto recording/transcription error: {str(e)}")
            return ""

def run_stt(blocking: bool = True):
    ...
    transcriber = VoiceTranscriber()

    # ---- 终端按键监听替代 keyboard ----
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    tty.setraw(fd)             # ← 关键！改用 raw 而不是 cbreak
    HOLD_TIMEOUT = 0.8
    last_s_time = None
    recording = False

    print("\n🎧 Hold S to start Chinese-English recording (ESC to exit)")
    try:
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.05)
            now = time.time()

            if ready:
                ch = sys.stdin.read(1)
                if ch == '\x1b':          # ESC
                    break
                elif ch.lower() == 's':   # Press/keep holding S
                    last_s_time = now
                    if not recording:
                        transcriber.start_recording()
                        recording = True

            if recording and last_s_time and (now - last_s_time) > HOLD_TIMEOUT:
                transcriber.stop_recording()
                recording = False

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        transcriber.cleanup()
    print("\n👋 Program exited")


if __name__ == "__main__":
    # run_stt()
    transcriber = VoiceTranscriber()
    text = transcriber.auto_record_and_transcribe(5)  # 自动录音5秒
    print("Final result:", text)
    transcriber.cleanup()