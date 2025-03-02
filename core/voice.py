import tempfile
import threading
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import tkinter as tk
import time
import win32gui
from gtts import gTTS
from window_focus import activate_window_title

# Initialize Pygame's mixer
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
volume = 0.25
subtitles = True

class TransparentSubtitlesWindow:
    _current_instance = None  # Class variable to track current instance
    
    def __init__(self, text):
        # Close any existing subtitle window
        if TransparentSubtitlesWindow._current_instance is not None:
            try:
                TransparentSubtitlesWindow._current_instance.close()
            except:
                pass
        
        TransparentSubtitlesWindow._current_instance = self
        self.root = tk.Tk()
        self.text = text

        # Calculate wraplength to be slightly less than the full screen width
        screen_width = self.root.winfo_screenwidth()
        self.padding = 10
        wraplength = screen_width - self.padding * 2

        self.label = tk.Label(self.root, text=self.text, font=('Helvetica', 20), 
                            fg='light sky blue', bg='DodgerBlue4', 
                            wraplength=wraplength, justify='center')
        self.label.pack()

        # Set the window to be always on top, transparent, and without decorations
        self.root.overrideredirect(True)
        self.root.attributes('-topmost', True)
        self.root.attributes('-transparentcolor', 'DodgerBlue4')
        
        # Prevent focus stealing
        self.root.withdraw()  # Hide window initially
        # self.root.attributes('-alpha', 0.8)  # Make slightly transparent
        
        # Set window position
        label_width = self.label.winfo_reqwidth()
        x_position = (screen_width - label_width) // 2
        self.root.geometry('+%d+%d' % (x_position, self.root.winfo_screenheight() - 150))
        
        # Show window without stealing focus
        self.root.deiconify()
        self.update()

    def update(self):
        self.label.configure(text=self.text)
        self.root.update_idletasks()
        try:
            self.root.update()
        except:
            pass

    def change_text(self, new_text, duration):
        self.text = new_text
        self.update()

        # Schedule removing the text after the duration
        self.root.after(duration, self.close)

    def close(self):
        try:
            self.root.quit()
            self.root.destroy()
        except:
            pass
        finally:
            if TransparentSubtitlesWindow._current_instance == self:
                TransparentSubtitlesWindow._current_instance = None

def calculate_duration_of_speech(text, lang='en', wpm=150):
    """Estimate the duration the subtitles should be displayed based on words per minute (WPM)"""
    words = text.split()
    word_count = len(words)
    # Adjust WPM for Vietnamese (generally slower than English)
    if lang == 'vi':
        wpm = 120
    duration_in_seconds = (word_count / wpm) * 60
    return int(duration_in_seconds * 1000)  # Convert to milliseconds for tkinter's after method

def play_audio(file_path):
    # Load and play audio file
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.set_volume(volume)
    pygame.mixer.music.play()

    # When the audio finishes, stop the mixer and remove the temporary file
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.music.unload()
    os.remove(file_path)

def set_volume(volume_level):
    global volume
    volume = volume_level / 100.0  # Normalize volume to 0.0 - 1.0
    pygame.mixer.music.set_volume(volume) # Set volume immediately

def set_subtitles(subtitles_bool):
    global subtitles
    subtitles = subtitles_bool

def speaker(text, additional_text=None, lang='en', skip_focus=False):
    # Store current focused window only if we're not skipping focus
    if os.name == 'nt' and not skip_focus:
        current_window = win32gui.GetForegroundWindow()
        current_title = win32gui.GetWindowText(current_window)
    else:
        current_title = None

    # Initialize all of pygame's modules
    pygame.init()

    from utils import app_instance, translate
    if app_instance is not None:
        lang = getattr(app_instance, 'current_language', 'en')
        text = f"{translate(text, lang)}{f" {translate(additional_text, lang)}" if additional_text else ''}"

    # Temporary mp3 file creation
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(fp.name)
        temp_file_path = fp.name

    # Start the subtitles thread
    subtitles_thread = None
    if subtitles is True:
        def setup_subtitles():
            window = TransparentSubtitlesWindow(text)
            window.change_text(text, calculate_duration_of_speech(text, lang))
            window.root.mainloop()

        subtitles_thread = threading.Thread(target=setup_subtitles)
        subtitles_thread.daemon = True
        subtitles_thread.start()

    # Start the audio thread
    audio_thread = threading.Thread(target=play_audio, args=(temp_file_path,))
    audio_thread.daemon = True
    audio_thread.start()

    # Restore focus to previous window after a short delay
    if os.name == 'nt' and current_title and not skip_focus:
        def restore_focus():
            time.sleep(0.5)  # Small delay to let the audio start
            try:
                new_window = win32gui.GetForegroundWindow()
                if new_window != current_window:
                    activate_window_title(current_title)
            except:
                pass

        focus_thread = threading.Thread(target=restore_focus)
        focus_thread.daemon = True
        focus_thread.start()

    # Return the threads in case the caller wants to track them
    return audio_thread, subtitles_thread

if __name__ == '__main__':
    text_to_speak = "Hello, this is a test."
    speaker(text_to_speak)
    # Main script can do other tasks here, threads will not prevent script from exiting
