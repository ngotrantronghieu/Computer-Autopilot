import customtkinter as Ctk
import speech_recognition as sr
import threading
import keyboard
import json
from voice import speaker, set_volume, set_subtitles
from driver import assistant, act, fast_act, auto_role, perform_simulated_keypress, write_action
from window_focus import activate_window_title
from utils import set_app_instance, print_to_chat
from core_api import set_llm_model, set_api_key, set_vision_llm_model, set_vision_api_key

# Initialize components
Ctk.set_appearance_mode("system")
Ctk.set_default_color_theme("blue")

SETTINGS_FILE = "settings.json"

def load_settings():
    try:
        with open(SETTINGS_FILE, "r") as f:
            settings = json.load(f)
            set_llm_model(settings.get("llm_model", ""))
            set_api_key("", settings.get("api_key_env_name", ""))
            set_vision_llm_model(settings.get("vision_llm_model", ""))
            set_vision_api_key("", settings.get("vision_api_key_env_name", ""))
    except FileNotFoundError:
        pass


class LLMSettingsWindow:
    def __init__(self, parent, llm_model, api_key_env_name, vision_llm_model, vision_api_key_env_name):
        self.parent = parent
        self.settings_window = Ctk.CTkToplevel(parent.root)
        self.settings_window.title("LLM Settings")
        self.settings_window.geometry("400x600")
        self.settings_window.transient(parent.root)

        # LLM Model Setting
        self.llm_model_label = Ctk.CTkLabel(self.settings_window, text="Main Model Name:")
        self.llm_model_label.pack(pady=(20, 5), padx=20)
        self.llm_model_entry = Ctk.CTkEntry(self.settings_window, width=200)
        self.llm_model_entry.insert(0, llm_model)
        self.llm_model_entry.pack(pady=5, padx=20)
        self.llm_model_entry.bind("<KeyRelease>", self.check_fields)

        # API Key Environment Variable Name
        self.api_key_env_name_label = Ctk.CTkLabel(self.settings_window, text="API Key Env Name:")
        self.api_key_env_name_label.pack(pady=(10, 5), padx=20)
        self.api_key_env_name_entry = Ctk.CTkEntry(self.settings_window, width=200)
        self.api_key_env_name_entry.insert(0, api_key_env_name)
        self.api_key_env_name_entry.pack(pady=5, padx=20)
        self.api_key_env_name_entry.bind("<KeyRelease>", self.check_fields)

        # API Key Setting
        self.api_key_label = Ctk.CTkLabel(self.settings_window, text="API Key (Optional):")
        self.api_key_label.pack(pady=(10, 5), padx=20)
        self.api_key_entry = Ctk.CTkEntry(self.settings_window, width=200)
        self.api_key_entry.pack(pady=5, padx=20)
        self.api_key_entry.bind("<KeyRelease>", self.check_fields)

        # Vision LLM Model Setting
        self.vision_llm_model_label = Ctk.CTkLabel(self.settings_window, text="Vision Model Name:")
        self.vision_llm_model_label.pack(pady=(20, 5), padx=20)
        self.vision_llm_model_entry = Ctk.CTkEntry(self.settings_window, width=200)
        self.vision_llm_model_entry.insert(0, vision_llm_model)
        self.vision_llm_model_entry.pack(pady=5, padx=20)
        self.vision_llm_model_entry.bind("<KeyRelease>", self.check_fields)

        # Vision API Key Environment Variable Name
        self.vision_api_key_env_name_label = Ctk.CTkLabel(self.settings_window, text="Vision API Key Env Name:")
        self.vision_api_key_env_name_label.pack(pady=(10, 5), padx=20)
        self.vision_api_key_env_name_entry = Ctk.CTkEntry(self.settings_window, width=200)
        self.vision_api_key_env_name_entry.insert(0, vision_api_key_env_name)
        self.vision_api_key_env_name_entry.pack(pady=5, padx=20)
        self.vision_api_key_env_name_entry.bind("<KeyRelease>", self.check_fields)

        # Vision API Key Setting
        self.vision_api_key_label = Ctk.CTkLabel(self.settings_window, text="Vision API Key (Optional):")
        self.vision_api_key_label.pack(pady=(10, 5), padx=20)
        self.vision_api_key_entry = Ctk.CTkEntry(self.settings_window, width=200)
        self.vision_api_key_entry.pack(pady=5, padx=20)
        self.vision_api_key_entry.bind("<KeyRelease>", self.check_fields)

        self.update_button = Ctk.CTkButton(self.settings_window, text="Update", command=self.update_settings)
        self.update_button.pack(pady=20, padx=20)
        self.check_fields()

    def update_settings(self):
        new_model = self.llm_model_entry.get()
        new_api_key = self.api_key_entry.get()
        new_api_key_env_name = self.api_key_env_name_entry.get()
        new_vision_model = self.vision_llm_model_entry.get()
        new_vision_api_key = self.vision_api_key_entry.get()
        new_vision_api_key_env_name = self.vision_api_key_env_name_entry.get()

        set_llm_model(new_model)
        set_api_key(new_api_key, new_api_key_env_name)
        set_vision_llm_model(new_vision_model)
        set_vision_api_key(new_vision_api_key, new_vision_api_key_env_name)

        settings = {
            "llm_model": new_model,
            "api_key_env_name": new_api_key_env_name,
            "vision_llm_model": new_vision_model,
            "vision_api_key_env_name": new_vision_api_key_env_name
        }
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f)

        print_to_chat(f"LLM settings updated.")
        self.settings_window.withdraw()

    def check_fields(self, event=None):
        if self.llm_model_entry.get() and self.api_key_env_name_entry.get() and self.vision_llm_model_entry.get() and self.vision_api_key_env_name_entry.get():
            self.update_button.configure(state="normal")
        else:
            self.update_button.configure(state="disabled")

class MiniChatInterface:
    def __init__(self, parent):
        self.parent = parent
        self.mini_window = Ctk.CTkToplevel()
        self.mini_window.title("")
        self.mini_window.geometry("300x60")

        # Remove window decorations
        self.mini_window.overrideredirect(True)

        # Position at bottom right
        screen_width = self.mini_window.winfo_screenwidth()
        screen_height = self.mini_window.winfo_screenheight()
        x = screen_width - 400
        y = screen_height - 150
        self.mini_window.geometry(f"300x60+{x}+{y}")

        # Make window stay on top
        self.mini_window.attributes('-topmost', True)

        # Create main container
        self.container = Ctk.CTkFrame(self.mini_window)
        self.container.pack(fill="both", expand=True, padx=5, pady=5)

        # Progress bar (hidden by default)
        self.progress_bar = Ctk.CTkProgressBar(self.container)
        self.progress_bar.set(0)

        # Input area container
        self.input_container = Ctk.CTkFrame(self.container)
        self.input_container.pack(fill="x", padx=2, pady=(0,2), side="bottom")

        # Text input
        self.input_field = Ctk.CTkEntry(
            self.input_container,
            placeholder_text="Type a task...",
            height=30
        )
        self.input_field.pack(side="left", fill="x", expand=True, padx=(0,5))

        # Voice button
        self.voice_button = Ctk.CTkButton(
            self.input_container,
            text="🎤",
            width=30,
            height=30,
            command=self.toggle_voice_input
        )
        self.voice_button.pack(side="right")

        # Bind enter key to send message
        self.input_field.bind("<Return>", lambda e: self.send_message())

        # Bind click and drag to move window
        self.mini_window.bind("<Button-1>", self.start_move)
        self.mini_window.bind("<B1-Motion>", self.on_move)

        # Initially hide the mini window
        self.mini_window.withdraw()

        # Initialize voice recognition
        self.recognizer = sr.Recognizer()
        self.voice_active = False

        # Processing flag
        self.is_processing = False

    def start_move(self, event):
        if event.widget not in [self.input_field, self.voice_button]:
            self.x = event.x
            self.y = event.y

    def on_move(self, event):
        if hasattr(self, 'x'):
            deltax = event.x - self.x
            deltay = event.y - self.y
            x = self.mini_window.winfo_x() + deltax
            y = self.mini_window.winfo_y() + deltay
            self.mini_window.geometry(f"+{x}+{y}")

    def show(self):
        self.mini_window.deiconify()

    def hide(self):
        self.mini_window.withdraw()

    def send_message(self):
        message = self.input_field.get().strip()
        if message:
            self.input_field.delete(0, "end")
            self.show_progress()
            self.is_processing = True
            self.parent.add_message(message, is_user=True)
            self.parent.process_message(message)

    def toggle_voice_input(self):
        if not self.voice_active:
            self.voice_active = True
            self.voice_button.configure(fg_color="red")
            threading.Thread(target=self.listen_voice).start()
        else:
            self.voice_active = False
            self.voice_button.configure(fg_color=("gray75", "gray25"))

    def listen_voice(self):
        with sr.Microphone() as source:
            try:
                audio = self.recognizer.listen(source, timeout=5)
                text = self.recognizer.recognize_google(audio)
                self.mini_window.after(0, lambda: self.input_field.insert(0, text))
                self.mini_window.after(0, self.send_message)
            except:
                pass
            finally:
                self.voice_active = False
                self.mini_window.after(0, lambda: self.voice_button.configure(fg_color=("gray75", "gray25")))

    def show_progress(self):
        self.progress_bar.set(0)
        self.progress_bar.pack(fill="x", padx=2, pady=(2,0), side="top")
        self.start_progress()

    def hide_progress(self):
        self.progress_bar.pack_forget()
        self.is_processing = False

    def start_progress(self):
        if self.is_processing:
            current = self.progress_bar.get()
            if current < 1:
                self.progress_bar.set(current + 0.1)
                self.mini_window.after(100, self.start_progress)

class ModernChatInterface:
    def __init__(self):
        self.root = Ctk.CTk()
        self.root.title("Computer Autopilot")
        self.root.geometry("800x600")

        # Create main container
        self.main_container = Ctk.CTkFrame(self.root)
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)

        # Chat history area
        self.chat_frame = Ctk.CTkScrollableFrame(self.main_container)
        self.chat_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Input area container
        self.input_container = Ctk.CTkFrame(self.main_container)
        self.input_container.pack(fill="x", padx=5, pady=(5,10))

        # Text input
        self.input_field = Ctk.CTkEntry(
            self.input_container,
            placeholder_text="Type the task you want to perform...",
            height=40
        )
        self.input_field.pack(side="left", fill="x", expand=True, padx=(0,5))

        # Send button
        self.send_button = Ctk.CTkButton(
            self.input_container,
            text="Send",
            width=100,
            height=40,
            command=self.send_message
        )
        self.send_button.pack(side="right")

        # Voice button
        self.voice_button = Ctk.CTkButton(
            self.input_container,
            text="🎤",
            width=40,
            height=40,
            command=self.toggle_voice_input
        )
        self.voice_button.pack(side="right", padx=5)

        # Add volume control
        self.volume_frame = Ctk.CTkFrame(self.main_container)
        self.volume_frame.pack(fill="x", padx=5, pady=(0,5))

        # Volume slider
        self.volume_slider = Ctk.CTkSlider(
            self.volume_frame,
            from_=0,
            to=100,
            command=self.update_volume
        )
        self.volume_slider.set(80)  # Default volume
        self.volume_slider.pack(side="left", fill="x", expand=True, padx=(5,10))

        # Mute button
        self.mute_button = Ctk.CTkButton(
            self.volume_frame,
            text="🔊",
            width=40,
            height=30,
            command=self.toggle_mute
        )
        self.mute_button.pack(side="right", padx=5)

        # Subtitles toggle
        self.subtitles_var = Ctk.BooleanVar(value=True)
        self.subtitles_checkbox = Ctk.CTkCheckBox(
            self.volume_frame,
            text="Subtitles",
            variable=self.subtitles_var,
            command=self.toggle_subtitles
        )
        self.subtitles_checkbox.pack(side="right", padx=10)

        # Bind enter key to send message
        self.input_field.bind("<Return>", lambda e: self.send_message())

        # Initialize voice recognition
        self.recognizer = sr.Recognizer()
        self.voice_active = False

        # Initialize audio settings
        self.is_muted = False
        set_volume(50)  # Default volume
        set_subtitles(True)  # Default subtitles setting

        keyboard.add_hotkey("alt+m", self.toggle_voice_input)

        self.settings_button = Ctk.CTkButton(self.volume_frame, text="Settings", command=self.open_settings_window)
        self.settings_button.pack(side="left", padx=(10, 5))
        self.settings_window = None  # Store the settings window instance

        # Create mini chat interface
        self.mini_chat = MiniChatInterface(self)

        # Bind window state events
        self.root.bind("<Unmap>", self.handle_minimize)
        self.root.bind("<Map>", self.handle_restore)

    def handle_minimize(self, event):
        self.mini_chat.show()

    def handle_restore(self, event):
        self.mini_chat.hide()

    def open_settings_window(self):
        from core_api import current_llm_model, current_api_key_env_name, current_vision_llm_model, current_vision_api_key_env_name
        if self.settings_window is None or not self.settings_window.settings_window.winfo_exists():
            self.settings_window = LLMSettingsWindow(
            self, 
            current_llm_model, 
            current_api_key_env_name, 
            current_vision_llm_model, 
            current_vision_api_key_env_name
        )
        else:
            self.settings_window.settings_window.deiconify()

    def add_message(self, message, is_user=True):
        # Create message bubble
        message_frame = Ctk.CTkFrame(
            self.chat_frame,
            fg_color=("#DCF8C6" if is_user else "#E8E8E8")
        )

        # Message text
        message_label = Ctk.CTkLabel(
            message_frame,
            text=message,
            text_color="black",
            wraplength=400,
            justify="left"
        )
        message_label.pack(padx=10, pady=5)

        # Pack message with appropriate alignment
        message_frame.pack(
            anchor="e" if is_user else "w",
            padx=10,
            pady=5,
            fill="x"
        )

        # Auto scroll to bottom
        self.chat_frame._parent_canvas.yview_moveto(1.0)

    def send_message(self):
        message = self.input_field.get().strip()
        if message:
            # Add user message to chat
            self.add_message(message, is_user=True)

            # Clear input field
            self.input_field.delete(0, "end")

            # Process message and get AI response
            self.process_message(message)

    def process_message(self, message):
        def process():
            try:
                message_lower = message.lower()
                if "open" in message_lower:
                    if len(message_lower) < 20:
                        window_title = message_lower.split("open ")[-1].strip()
                        activate_window_title(window_title)
                        response = f"Activated Window: {window_title}"
                        self.root.after(0, lambda: self.add_message(response, is_user=False))
                        speaker(response)
                    else:
                        assistant(message_lower)
                elif "scroll" in message_lower:
                    import pyautogui
                    if "up" in message_lower:
                        pyautogui.scroll(800)
                        response = "Scrolled up"
                    else:
                        pyautogui.scroll(-800)
                        response = "Scrolled down"
                    self.root.after(0, lambda: self.add_message(response, is_user=False))
                    speaker(response)
                elif "press" in message_lower:
                    key = message_lower.split("press ")[-1].strip()
                    perform_simulated_keypress(key)
                    response = f"Pressed {key}"
                    self.root.after(0, lambda: self.add_message(response, is_user=False))
                    speaker(response)
                elif "type" in message_lower:
                    text = message_lower.split("type ")[-1].strip()
                    write_action(goal=text, last_step="text_entry")
                    response = f"Typed: {text}"
                    self.root.after(0, lambda: self.add_message(response, is_user=False))
                    speaker(response)
                elif "click" in message_lower:
                    if len(message_lower) < 30:
                        response = fast_act(message_lower)
                        if response:
                            self.root.after(0, lambda: self.add_message(response, is_user=False))
                            speaker(response)
                    else:
                        assistant(message_lower)
                elif "double click" in message_lower:
                    if len(message_lower) < 37:
                        response = fast_act(message_lower.split("double ")[-1].strip(), double_click=True)
                        if response:
                            self.root.after(0, lambda: self.add_message(response, is_user=False))
                            speaker(response)
                    else:
                        assistant(message_lower)
                else:
                    response = auto_role(message)
                    if response:
                        self.root.after(0, lambda: self.add_message(response, is_user=False))
                        speaker(response)
                        if "windows_assistant" in response:
                            assistant_thread = threading.Thread(
                                target=lambda: assistant(assistant_goal=message, called_from="chat")
                            )
                            assistant_thread.start()
                        else:
                            act_thread = threading.Thread(
                                target=lambda: act(message)
                            )
                            act_thread.start()
                if self.root.state() == 'iconic':
                    self.mini_chat.show()
                self.root.after(0, self.mini_chat.hide_progress)
            except Exception as e:
                error_message = f"Error: {str(e)}"
                self.root.after(0, lambda: self.add_message(error_message, is_user=False))
                speaker(error_message)
        threading.Thread(target=process).start()

    def toggle_voice_input(self, event=None):
        if not self.voice_active:
            self.voice_active = True
            self.voice_button.configure(fg_color="red")
            self.mini_chat.voice_button.configure(fg_color="red") # Turn mini chat button red
            threading.Thread(target=self.listen_voice).start()
        else:
            self.voice_active = False
            self.voice_button.configure(fg_color=("gray75", "gray25"))
            self.mini_chat.voice_button.configure(fg_color=("gray75", "gray25")) # Turn mini chat button back

    def listen_voice(self):
        with sr.Microphone() as source:
            try:
                audio = self.recognizer.listen(source, timeout=5)
                text = self.recognizer.recognize_google(audio)
                self.root.after(0, lambda: self.input_field.insert(0, text))
                self.root.after(0, self.send_message)
            except:
                pass
            finally:
                self.voice_active = False
                self.root.after(0, lambda: self.voice_button.configure(fg_color=("gray75", "gray25")))
                self.mini_chat.voice_button.configure(fg_color=("gray75", "gray25")) # Turn mini chat button back

    def update_volume(self, value):
        if not self.is_muted:
            set_volume(int(value))

    def toggle_mute(self):
        self.is_muted = not self.is_muted
        if self.is_muted:
            self.mute_button.configure(text="🔇")
            set_volume(0)
        else:
            self.mute_button.configure(text="🔊")
            set_volume(int(self.volume_slider.get()))

    def toggle_subtitles(self):
        set_subtitles(self.subtitles_var.get())

def create_app():
    load_settings()
    app = ModernChatInterface()
    set_app_instance(app)
    app.root.mainloop()

if __name__ == "__main__":
    create_app()
