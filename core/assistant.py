import speech_recognition as sr
import threading
import json
import pyautogui
import ctypes
import os
import time
import queue
import schedule
import re
import sys
import uuid
import winreg
import tempfile
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton,
                              QVBoxLayout, QHBoxLayout, QLineEdit, QScrollArea,
                              QLabel, QFrame, QSlider, QCheckBox, QDialog, QTabWidget,
                              QTextEdit, QDialogButtonBox, QMessageBox, QSystemTrayIcon, QMenu,
                              QListWidget, QComboBox, QStackedWidget, QSpinBox,
                              QDoubleSpinBox, QListWidgetItem, QGroupBox, QTimeEdit,
                              QDateEdit, QInputDialog)
from PySide6.QtCore import Qt, Signal, QObject, QEvent, QSize, Slot, QMetaObject, Q_ARG, QTime, QDate, QLocale, QDateTime, QLockFile
from PySide6.QtGui import QIcon, QFont, QShortcut, QKeySequence, QAction
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from voice import speaker, set_volume, set_subtitles
from driver import (
    assistant, fast_act, perform_simulated_keypress, write_action,
    request_stop, clear_stop, execute_action, web_assistant, close_browser,
    chat_create_session, chat_rename_session, chat_delete_session,
    chat_clear_session_messages, chat_list_sessions, chat_add_message,
    chat_list_messages
)
from window_focus import activate_window_title
from utils import set_app_instance, get_app_instance, print_to_chat, print_to_web_chat
from core_api import set_llm_model, set_api_key, set_vision_llm_model, set_vision_api_key
from langdetect import detect
from tasks import save_task, delete_task, get_task, load_tasks
from pynput import mouse, keyboard
from datetime import datetime

SETTINGS_FILE = "settings.json"

def load_settings():
    default_settings = {
        "llm_model": "",
        "api_key_env_name": "",
        "vision_llm_model": "",
        "vision_api_key_env_name": "",
        "llm_provider_options": {},
        "vision_provider_options": {},
        "action_delay": 1.5,
        "max_attempts": 20,
        "web_action_delay": 1.5,
        "web_max_attempts": 20,
        "start_with_windows": False
    }

    try:
        with open(SETTINGS_FILE, "r") as f:
            user_settings = json.load(f)
            # Merge with defaults
            for key in user_settings:
                if key in default_settings:
                    default_settings[key] = user_settings[key]

            # Set core API values
            set_llm_model(default_settings["llm_model"], default_settings.get("llm_provider_options", {}))
            set_api_key("", default_settings["api_key_env_name"])
            set_vision_llm_model(default_settings["vision_llm_model"], default_settings.get("vision_provider_options", {}))
            set_vision_api_key("", default_settings["vision_api_key_env_name"])

            return default_settings

    except FileNotFoundError:
        return default_settings
    except Exception as e:
        print(f"Error loading settings: {e}")
        return default_settings


# Helper: format session timestamps to short, local format for UI labels
def _format_session_timestamp(dt_str) -> str:
    try:
        s = (dt_str or "").strip()
        if not s:
            return ""
        # Normalize trailing Z to +00:00 for fromisoformat
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
        except Exception:
            # If parsing fails, show raw
            return dt_str
        # Convert to local time if aware
        if getattr(dt, 'tzinfo', None) is not None:
            dt = dt.astimezone()
        # Prefer Qt locale-aware short format
        try:
            qdt = QDateTime(dt)
            return QLocale.system().toString(qdt, QLocale.ShortFormat)
        except Exception:
            # Fallback to a compact local format
            return dt.strftime("%x %H:%M")
    except Exception:
        return str(dt_str)

class MessageBubble(QFrame):
    def __init__(self, message, is_user=True, parent=None):
        super().__init__(parent)
        self.setObjectName("messageBubble")

        # Create main layout
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 5, 10, 5)

        # Add spacing to align messages
        if is_user:
            main_layout.addStretch()

        # Create message container
        message_container = QFrame()
        message_layout = QVBoxLayout(message_container)
        message_layout.setContentsMargins(10, 5, 10, 5)

        # Add message label with larger font
        message_label = QLabel(message)
        message_label.setWordWrap(True)
        message_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        font = message_label.font()
        font.setPointSize(11)
        message_label.setFont(font)
        message_label.setStyleSheet("color: #000000;")
        message_layout.addWidget(message_label)

        # Add message container to main layout
        main_layout.addWidget(message_container)

        # Add spacing for assistant messages
        if not is_user:
            main_layout.addStretch()

        # Apply styles
        if is_user:
            message_container.setStyleSheet("""
                QFrame {
                    background-color: #DCF8C6;
                    border-radius: 10px;
                    padding: 2px;
                }
            """)
        else:
            message_container.setStyleSheet("""
                QFrame {
                    background-color: #E8E8E8;
                    border-radius: 10px;
                    padding: 2px;
                }
            """)

class ChatArea(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        container = QWidget()
        self.layout = QVBoxLayout(container)
        self.layout.setAlignment(Qt.AlignTop)
        self.layout.setSpacing(10)
        self.setWidget(container)

        # Modern styling
        self.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #FFFFFF;
            }
            QWidget {
                background-color: #FFFFFF;
            }
        """)

class AppContextDialog(QDialog):
    def __init__(self, parent=None, app_name=None, context_data=None, is_shortcut=False):
        super().__init__(parent)
        self.is_shortcut = is_shortcut
        self.setWindowTitle("Shortcuts Context Editor" if is_shortcut else "App Context Editor")
        self.setMinimumSize(600, 500)

        layout = QVBoxLayout(self)

        # App name input
        name_frame = QWidget()
        name_layout = QHBoxLayout(name_frame)
        name_layout.addWidget(QLabel("App Name:"))
        self.name_entry = QLineEdit()
        if app_name:
            self.name_entry.setText(app_name)
        name_layout.addWidget(self.name_entry)
        layout.addWidget(name_frame)

        # Context editor
        self.context_edit = QTextEdit()
        if context_data:
            if isinstance(context_data, list):
                self.context_edit.setText("\n".join(context_data))
            elif isinstance(context_data, dict):
                text = ""
                for key, value in context_data.items():
                    text += f"{key}:\n"
                    if isinstance(value, list):
                        text += "\n".join(f"  {item}" for item in value)
                    text += "\n\n"
                self.context_edit.setText(text)

        # Add helper text
        if self.is_shortcut:
            helper_text = "Enter keyboard shortcuts, one per line:\n"
        else:
            helper_text = "Enter contexts in this format:\n\nContext Name:\n  Identifier 1\n  Identifier 2\n...\n"
        layout.addWidget(QLabel(helper_text))

        layout.addWidget(self.context_edit)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_data(self):
        app_name = self.name_entry.text().strip()
        if not app_name:
            QMessageBox.warning(self, "Error", "App name is required!")
            return None, None

        context_text = self.context_edit.toPlainText()
        if not context_text.strip():
            QMessageBox.warning(self, "Error", "Context cannot be empty!")
            return None, None

        # Parse context based on type
        if self.is_shortcut:
            context_data = [line.strip() for line in context_text.split("\n") if line.strip()]
            if not context_data:
                QMessageBox.warning(self, "Error", "At least one shortcut is required!")
                return None, None
        else:
            context_data = {}
            current_key = None
            current_items = []

            for line in context_text.split("\n"):
                if line.strip().endswith(":"):
                    if current_key and current_items:
                        context_data[current_key] = current_items
                    current_key = line.strip()[:-1]
                    current_items = []
                elif line.strip() and current_key:
                    current_items.append(line.strip())

            if current_key and current_items:
                context_data[current_key] = current_items

            if not context_data:
                QMessageBox.warning(self, "Error", "At least one UI element with identifiers is required!")
                return None, None

        return app_name, context_data

class SettingsDialog(QDialog):
    def __init__(self, parent=None, llm_model="", api_key_env_name="",
                 vision_llm_model="", vision_api_key_env_name="",
                 action_delay=1.5, max_attempts=20, start_with_windows=False,
                 llm_provider_options=None, vision_provider_options=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumSize(1000, 700)
        self.resize(1200, 700)

        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Create scroll area for assistant settings
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Main container inside scroll area
        main_container = QWidget()
        scroll_area.setWidget(main_container)

        main_layout = QHBoxLayout(main_container)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(30)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(20)

        # General Language Model group
        llm_group = QFrame()
        llm_group.setObjectName("settingsGroup")
        llm_layout = QVBoxLayout(llm_group)

        llm_layout.addWidget(QLabel("General Language Model"))

        # LLM Provider dropdown
        providers = [
            "openai","openai_compatible","azure","azure_ai","vertex","gemini","anthropic","aws_sagemaker","bedrock",
            "litellm_proxy","meta_llama","mistral","codestral","cohere","anyscale","huggingface","hyperbolic","databricks",
            "deepgram","watsonx","predibase","nvidia_nim","nscale","xai","moonshot","lm_studio","cerebras","volcano",
            "triton_inference_server","ollama","perplexity","friendliai","galadriel","topaz","groq","deepseek","elevenlabs",
            "fireworks_ai","clarifai","vllm","llamafile","infinity","xinference","aiml","cloudflare_workers","deepinfra",
            "github","github_copilot","ai21","nlp_cloud","recraft","replicate","togetherai","v0","morph","lambda_ai",
            "novita","voyage","jina_ai","aleph_alpha","baseten","openrouter","sambanova","custom_llm_server","petals",
            "snowflake","gradient_ai","featherless_ai","nebius","dashscope","bytez","oci"
        ]
        # Suggested models for general chat per provider
        suggested_models_general = {
            "openai": ["gpt-4o","gpt-4o-mini","o4-mini","o3-mini","gpt-4.1-mini"],
            "gemini": ["gemini-2.0-flash","gemini-1.5-pro","gemini-1.5-flash","gemini-2.0-pro-exp"],
            "anthropic": ["claude-3-5-sonnet-latest","claude-3-opus-latest","claude-3-haiku-latest"],
            "groq": ["llama-3.3-70b-versatile","llama-3.1-70b","mixtral-8x7b"],
            "mistral": ["mistral-large-latest","mistral-small-latest","open-mixtral-8x7b"],
            "openrouter": [],
            "togetherai": ["meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo","Qwen/Qwen2.5-72B-Instruct"],
            "ollama": ["llama3.1","phi4","qwen2.5:latest"],
        }
        # Suggested models for vision per provider
        suggested_models_vision = {
            "openai": ["gpt-4o","gpt-4o-mini"],
            "gemini": ["gemini-2.0-flash","gemini-1.5-pro","gemini-1.5-flash"],
            "anthropic": ["claude-3-5-sonnet-latest","claude-3-haiku-latest"],
            "ollama": ["llava:latest","qwen2-vl:latest"],
        }
        # Map provider to default API key env var (best-known defaults from LiteLLM docs)
        # Note: Some providers use cloud credentials instead of a single API key; we leave those blank.
        default_api_env = {
            # OpenAI family
            "openai": "OPENAI_API_KEY",
            "openai_compatible": "OPENAI_API_KEY",
            "azure": "AZURE_API_KEY",          # Azure OpenAI
            "azure_ai": "AZURE_API_KEY",
            # Google
            "gemini": "GEMINI_API_KEY",       # Google AI Studio (Gemini)
            "vertex": "",                     # Vertex AI uses ADC/service account; no single API key
            # Major labs
            "anthropic": "ANTHROPIC_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "cohere": "COHERE_API_KEY",
            # Infra/aggregators
            "openrouter": "OPENROUTER_API_KEY",
            "togetherai": "TOGETHERAI_API_KEY",
            "replicate": "REPLICATE_API_TOKEN",
            "fireworks_ai": "FIREWORKS_API_KEY",
            "perplexity": "PPLX_API_KEY",
            "groq": "GROQ_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "xai": "XAI_API_KEY",
            "ai21": "AI21_API_KEY",
            "huggingface": "HUGGINGFACE_API_KEY",
            "litellm_proxy": "LITELLM_PROXY_API_KEY",
            "anyscale": "ANYSCALE_API_KEY",
            "hyperbolic": "HYPERBOLIC_API_KEY",
            "databricks": "DATABRICKS_TOKEN",
            "deepgram": "DEEPGRAM_API_KEY",
            "watsonx": "WATSONX_API_KEY",
            "nlp_cloud": "NLP_CLOUD_TOKEN",
            # Local/self-hosted (usually no key)
            "ollama": "",
            "vllm": "",
            "llamafile": "",
            "triton_inference_server": "",
            "lm_studio": "",
            "infinity": "",
            "xinference": "",
            "aiml": "",
            # Cloud/perimeter tokens
            "cloudflare_workers": "CLOUDFLARE_API_TOKEN",
            "github": "GITHUB_TOKEN",         # GitHub Models (Azure inference) typically uses GitHub token
            "github_copilot": "GITHUB_COPILOT_TOKEN",
            "clarifai": "CLARIFAI_API_KEY",
            "deepinfra": "DEEPINFRA_API_KEY",
            "cerebras": "CEREBRAS_API_KEY",
            "predibase": "PREDIBASE_API_KEY",
            "nvidia_nim": "NVIDIA_API_KEY",
            "nscale": "NSCALE_API_KEY",
            "moonshot": "MOONSHOT_API_KEY",
            "friendliai": "FRIENDLIAI_API_KEY",
            "galadriel": "GALADRIEL_API_KEY",
            "topaz": "TOPAZ_API_KEY",
            "recraft": "RECRAFT_API_KEY",
            "v0": "V0_API_KEY",
            "morph": "MORPH_API_KEY",
            "lambda_ai": "LAMBDA_API_KEY",
            "novita": "NOVITA_API_KEY",
            "voyage": "VOYAGE_API_KEY",
            "jina_ai": "JINA_API_KEY",
            "aleph_alpha": "ALEPH_ALPHA_API_KEY",
            "baseten": "BASETEN_API_KEY",
            "sambanova": "SAMBANOVA_API_KEY",
            # Misc / to verify
            "meta_llama": "",
            "volcano": "",
            "custom_llm_server": "",
            "petals": "",
            "snowflake": "",
            "gradient_ai": "GRADIENT_API_KEY",
            "featherless_ai": "FEATHERLESS_API_KEY",
            "nebius": "NEBIUS_API_KEY",
            "dashscope": "DASHSCOPE_API_KEY",
            "bytez": "BYTEZ_API_KEY",
            "oci": "",
            # AWS providers
            "bedrock": "",                    # Uses AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
            "aws_sagemaker": "",              # Uses AWS creds, no single API key
        }

        # LLM Provider dropdown
        llm_layout.addWidget(QLabel("Provider:"))
        self.llm_provider_combo = QComboBox()
        self.llm_provider_combo.addItems(providers)
        llm_layout.addWidget(self.llm_provider_combo)

        # LLM Model dropdown
        llm_layout.addWidget(QLabel("Model:"))
        self.llm_model_combo = QComboBox()
        self.llm_model_combo.setEditable(True)
        llm_layout.addWidget(self.llm_model_combo)

        # Dynamic provider-specific options UI (editable)
        provider_specific_fields = {
            # provider: list of (label, key, placeholder)
            "openai_compatible": [("API Base", "api_base", "https://your-openai-compatible-base")],
            "azure": [("API Base", "api_base", "https://{resource}.openai.azure.com"), ("API Version", "api_version", "2024-06-01"),],
            "azure_ai": [("API Base", "api_base", "https://{resource}.openai.azure.com"), ("API Version", "api_version", "2024-06-01"),],
            "vertex": [("Project", "vertex_project", "your-gcp-project"), ("Location", "vertex_location", "us-central1")],
            "bedrock": [("AWS Region", "aws_region_name", "us-east-1")],
            "ollama": [("API Base", "api_base", "http://localhost:11434")],
            "openrouter": [("API Base", "api_base", "https://openrouter.ai/api/v1")],
            "togetherai": [("API Base", "api_base", "https://api.together.xyz/v1")],
            "huggingface": [("API Base", "api_base", "https://api-inference.huggingface.co")],
            "vllm": [("API Base", "api_base", "http://localhost:8000/v1")],
            "cloudflare_workers": [("API Base", "api_base", "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run")],
            "github": [("API Base", "api_base", "https://models.inference.ai.azure.com")],
        }

        # Make provider options widget
        def make_provider_options_widget(prefix: str):
            w = QWidget(); w.setObjectName(f"{prefix}_provider_opts")
            layout = QVBoxLayout(w)
            layout.setContentsMargins(0, 10, 0, 10)
            layout.setSpacing(8)
            w._fields = {}
            w._layout = layout
            return w

        # Show provider options
        def show_provider_options(container, provider):
            # Clear existing widgets
            for child in container.findChildren(QLineEdit):
                child.deleteLater()
            for child in container.findChildren(QLabel):
                child.deleteLater()

            # Rebuild with consistent styling
            fields = provider_specific_fields.get(provider, [])
            layout = container._layout
            container._fields = {}

            for label, key, placeholder in fields:
                # Create label with consistent styling
                label_widget = QLabel(f"{label}:")
                label_widget.setMinimumHeight(20)

                # Create input field with consistent styling
                le = QLineEdit()
                le.setPlaceholderText(placeholder)
                le.setMinimumHeight(30)
                le.setMaximumHeight(30)

                layout.addWidget(label_widget)
                layout.addWidget(le)
                container._fields[key] = le

        # LLM provider options container
        self.llm_provider_opts_container = make_provider_options_widget("llm")
        llm_layout.addWidget(self.llm_provider_opts_container)

        # Update provider options when LLM provider changes
        self.llm_provider_combo.currentTextChanged.connect(lambda p: show_provider_options(self.llm_provider_opts_container, p))

        # API Key env + key
        self.api_key_env_name_entry = QLineEdit()
        self.api_key_env_name_entry.setText(api_key_env_name)
        llm_layout.addWidget(QLabel("API Key Environment Name:"))
        llm_layout.addWidget(self.api_key_env_name_entry)

        self.api_key_entry = QLineEdit()
        self.api_key_entry.setEchoMode(QLineEdit.Password)
        llm_layout.addWidget(QLabel("API Key (Optional):"))
        llm_layout.addWidget(self.api_key_entry)

        # Parse initial llm_model into provider/model
        init_llm_provider = "openai"
        init_llm_model = llm_model or "gpt-4o"
        if "/" in init_llm_model:
            parts = init_llm_model.split("/", 1)
            if parts[0]:
                init_llm_provider = parts[0]
                init_llm_model = parts[1]
        if init_llm_provider in providers:
            self.llm_provider_combo.setCurrentText(init_llm_provider)

        # Online models index cache
        self._models_index = None

        # Fetch models index cache
        def fetch_models_index():
            if self._models_index is not None:
                return True
            try:
                import json as _json
                from urllib.request import urlopen
                # Use LiteLLM official model catalog JSON from GitHub
                url = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
                with urlopen(url, timeout=10) as resp:
                    data = resp.read()
                self._models_index = _json.loads(data.decode("utf-8", errors="ignore"))
                return True
            except Exception:
                self._models_index = None
                return False

        # List models for provider
        def list_models_for_provider(provider: str, vision_only: bool=False):
            models = []
            idx = self._models_index or {}
            for model_id, meta in idx.items():
                if not isinstance(meta, dict):
                    continue
                if meta.get("litellm_provider") != provider:
                    continue
                if meta.get("mode") not in ("chat", "responses"):
                    continue
                if vision_only:
                    # heuristic: supports_vision or supported_modalities includes image
                    if not (meta.get("supports_vision") or ("supported_modalities" in meta and "image" in meta.get("supported_modalities", []))):
                        continue
                # exclude fine-tune namespaces from suggestion list (still user can type)
                if str(model_id).startswith("ft:"):
                    continue
                models.append(model_id)
            # stable sort - latest first if contains 'latest' or recent date
            def sort_key(m):
                s = m.lower()
                score = 0
                if "latest" in s: score -= 100
                # prefer newer dates in id (yyyy-mm or yyyy-mm-dd)
                import re
                dates = re.findall(r"(20\d{2}[-_](?:0[1-9]|1[0-2])(?:[-_](?:0[1-9]|[12]\d|3[01]))?)", s)
                if dates:
                    score -= 50
                # prefer larger models by common tokens
                if any(x in s for x in ["gpt-5","gpt-4.5","gpt-4o","sonnet","opus","70b","72b","405b","large","pro"]):
                    score -= 10
                return (score, s)
            models = sorted(set(models), key=sort_key)
            return models

        # Populate suggestions
        def populate_llm_models(provider):
            self.llm_model_combo.clear()
            added = False
            if fetch_models_index():
                for m in list_models_for_provider(provider, vision_only=False):
                    disp = m.split("/", 1)[1] if m.startswith(provider + "/") else m
                    self.llm_model_combo.addItem(disp)
                    added = True
            if not added:
                for m in suggested_models_general.get(provider, []):
                    self.llm_model_combo.addItem(m)
        populate_llm_models(init_llm_provider)
        # Select the saved model if it's present in the list
        self.llm_model_combo.setEditText(init_llm_model)

        # Auto-fill API env var on provider change
        def on_llm_provider_changed(p):
            # Prefer saved model when switching back to saved provider
            if p == init_llm_provider:
                populate_llm_models(p)
                self.llm_model_combo.setEditText(init_llm_model)
            else:
                populate_llm_models(p)
                # Select the first available model for the newly selected provider
                if self.llm_model_combo.count() > 0:
                    self.llm_model_combo.setCurrentIndex(0)
                    self.llm_model_combo.setEditText(self.llm_model_combo.currentText())
                else:
                    self.llm_model_combo.setEditText("")
            # Update provider-specific env var (may be empty for some providers)
            default_env = default_api_env.get(p, "")
            self.api_key_env_name_entry.setText(default_env)
        self.llm_provider_combo.currentTextChanged.connect(on_llm_provider_changed)


        # Vision Model group
        vision_group = QFrame()
        vision_group.setObjectName("settingsGroup")
        vision_layout = QVBoxLayout(vision_group)

        vision_layout.addWidget(QLabel("Vision Model"))

        # Vision provider dropdown
        vision_layout.addWidget(QLabel("Provider:"))

        self.vision_provider_combo = QComboBox()
        self.vision_provider_combo.addItems(providers)
        vision_layout.addWidget(self.vision_provider_combo)

        # Vision model dropdown
        vision_layout.addWidget(QLabel("Model (Vision-capable):"))
        self.vision_model_combo = QComboBox()
        self.vision_model_combo.setEditable(True)
        vision_layout.addWidget(self.vision_model_combo)

        # Vision provider options container
        self.vision_provider_opts_container = make_provider_options_widget("vision")
        vision_layout.addWidget(self.vision_provider_opts_container)
        self.vision_provider_combo.currentTextChanged.connect(lambda p: show_provider_options(self.vision_provider_opts_container, p))

        self.vision_api_key_env_name_entry = QLineEdit()
        self.vision_api_key_env_name_entry.setText(vision_api_key_env_name)
        vision_layout.addWidget(QLabel("API Key Environment Name:"))
        vision_layout.addWidget(self.vision_api_key_env_name_entry)

        self.vision_api_key_entry = QLineEdit()
        self.vision_api_key_entry.setEchoMode(QLineEdit.Password)
        vision_layout.addWidget(QLabel("API Key (Optional):"))
        vision_layout.addWidget(self.vision_api_key_entry)


        # Parse initial vision model and set combos
        init_vis_provider = "openai"
        init_vis_model = vision_llm_model or "gpt-4o"
        if "/" in init_vis_model:
            vparts = init_vis_model.split("/", 1)
            if vparts[0]:
                init_vis_provider = vparts[0]
                init_vis_model = vparts[1]
        if init_vis_provider in providers:
            self.vision_provider_combo.setCurrentText(init_vis_provider)

        def populate_vision_models(provider):
            self.vision_model_combo.clear()
            added = False
            if fetch_models_index():
                for m in list_models_for_provider(provider, vision_only=True):
                    disp = m.split("/", 1)[1] if m.startswith(provider + "/") else m
                    self.vision_model_combo.addItem(disp)
                    added = True
            if not added:
                for m in suggested_models_vision.get(provider, []):
                    self.vision_model_combo.addItem(m)
        populate_vision_models(init_vis_provider)
        # Set initial vision model text from saved settings
        self.vision_model_combo.setEditText(init_vis_model)

        # Auto-fill Vision API env var on provider change
        def on_vision_provider_changed(p):
            # Prefer saved model when switching back to saved provider
            if p == init_vis_provider:
                populate_vision_models(p)
                # Select the saved vision model if it's present in the list
                self.vision_model_combo.setEditText(init_vis_model)
            else:
                populate_vision_models(p)
                if self.vision_model_combo.count() > 0:
                    self.vision_model_combo.setCurrentIndex(0)
                    self.vision_model_combo.setEditText(self.vision_model_combo.currentText())
                else:
                    self.vision_model_combo.setEditText("")
            # Update provider-specific env var (may be empty for some providers)
            default_env = default_api_env.get(p, "")
            self.vision_api_key_env_name_entry.setText(default_env)
        self.vision_provider_combo.currentTextChanged.connect(on_vision_provider_changed)


        left_layout.addWidget(llm_group)
        left_layout.addWidget(vision_group)
        left_layout.addStretch()

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(20)

        performance_group = QFrame()
        performance_group.setObjectName("settingsGroup")
        performance_layout = QVBoxLayout(performance_group)

        # Action Delay
        delay_container = QWidget()
        delay_layout = QHBoxLayout(delay_container)
        delay_layout.addWidget(QLabel("Delay Between Actions (seconds):"))
        self.action_delay_spin = QDoubleSpinBox()
        self.action_delay_spin.setRange(0.1, 10.0)
        self.action_delay_spin.setValue(action_delay)
        self.action_delay_spin.setSingleStep(0.1)
        delay_layout.addWidget(self.action_delay_spin)
        performance_layout.addWidget(delay_container)

        # Max Attempts
        attempts_container = QWidget()
        attempts_layout = QHBoxLayout(attempts_container)
        attempts_layout.addWidget(QLabel("Max Attempts:"))
        self.max_attempts_spin = QSpinBox()
        self.max_attempts_spin.setRange(1, 100)
        self.max_attempts_spin.setValue(max_attempts)
        attempts_layout.addWidget(self.max_attempts_spin)
        performance_layout.addWidget(attempts_container)

        # Add stretch to push controls up
        performance_layout.addStretch()

        right_layout.addWidget(performance_group)

        # Create a new Web Agent settings group
        web_group = QFrame()
        web_group.setObjectName("settingsGroup")
        web_layout = QVBoxLayout(web_group)

        web_layout.addWidget(QLabel("Web Agent Settings"))

        # Web Action Delay
        web_delay_container = QWidget()
        web_delay_layout = QHBoxLayout(web_delay_container)
        web_delay_layout.addWidget(QLabel("Delay Between Web Actions (seconds):"))
        self.web_action_delay_spin = QDoubleSpinBox()
        self.web_action_delay_spin.setRange(0.1, 10.0)
        self.web_action_delay_spin.setValue(action_delay)  # Default to same as regular action delay
        self.web_action_delay_spin.setSingleStep(0.1)
        web_delay_layout.addWidget(self.web_action_delay_spin)
        web_layout.addWidget(web_delay_container)

        # Web Max Attempts
        web_attempts_container = QWidget()
        web_attempts_layout = QHBoxLayout(web_attempts_container)
        web_attempts_layout.addWidget(QLabel("Web Max Attempts:"))
        self.web_max_attempts_spin = QSpinBox()
        self.web_max_attempts_spin.setRange(1, 100)
        self.web_max_attempts_spin.setValue(max_attempts)  # Default to same as regular max attempts
        web_attempts_layout.addWidget(self.web_max_attempts_spin)
        web_layout.addWidget(web_attempts_container)

        # Add stretch to push controls up
        web_layout.addStretch()

        right_layout.addWidget(web_group)
        right_layout.addStretch()

        # Add panels to main layout with better proportions
        main_layout.addWidget(left_panel, 65)
        main_layout.addWidget(right_panel, 35)

        # Create tab widget and add scrollable content
        tab_widget = QTabWidget()

        # Create scrollable assistant settings tab
        assistant_tab = QWidget()
        assistant_tab_layout = QVBoxLayout(assistant_tab)
        assistant_tab_layout.setContentsMargins(0, 0, 0, 0)
        assistant_tab_layout.addWidget(scroll_area)

        tab_widget.addTab(assistant_tab, "Assistant Settings")

        # App Space tab
        app_space_widget = QWidget()
        app_space_layout = QVBoxLayout(app_space_widget)

        # Create two sections
        general_apps_group = QGroupBox("General App Contexts")
        shortcuts_group = QGroupBox("Keyboard Shortcuts")

        # General Apps section
        general_layout = QVBoxLayout(general_apps_group)
        self.general_app_list = QScrollArea()
        self.general_app_list.setWidgetResizable(True)
        general_list_container = QWidget()
        self.general_app_list_layout = QVBoxLayout(general_list_container)
        self.general_app_list.setWidget(general_list_container)

        general_buttons = QHBoxLayout()
        add_general_btn = QPushButton("Add App")
        edit_general_btn = QPushButton("Edit")
        remove_general_btn = QPushButton("Remove")
        add_general_btn.clicked.connect(lambda: self.add_app("UI Elements"))
        edit_general_btn.clicked.connect(lambda: self.edit_app("UI Elements"))
        remove_general_btn.clicked.connect(lambda: self.remove_app("UI Elements"))

        general_buttons.addWidget(add_general_btn)
        general_buttons.addWidget(edit_general_btn)
        general_buttons.addWidget(remove_general_btn)
        general_buttons.addStretch()

        general_layout.addWidget(self.general_app_list)
        general_layout.addLayout(general_buttons)

        # Keyboard Shortcuts section
        shortcuts_layout = QVBoxLayout(shortcuts_group)
        self.shortcuts_app_list = QScrollArea()
        self.shortcuts_app_list.setWidgetResizable(True)
        shortcuts_list_container = QWidget()
        self.shortcuts_app_list_layout = QVBoxLayout(shortcuts_list_container)
        self.shortcuts_app_list.setWidget(shortcuts_list_container)

        shortcuts_buttons = QHBoxLayout()
        add_shortcut_btn = QPushButton("Add Shortcuts")
        edit_shortcut_btn = QPushButton("Edit")
        remove_shortcut_btn = QPushButton("Remove")
        add_shortcut_btn.clicked.connect(lambda: self.add_app("Keyboard Shortcuts"))
        edit_shortcut_btn.clicked.connect(lambda: self.edit_app("Keyboard Shortcuts"))
        remove_shortcut_btn.clicked.connect(lambda: self.remove_app("Keyboard Shortcuts"))

        shortcuts_buttons.addWidget(add_shortcut_btn)
        shortcuts_buttons.addWidget(edit_shortcut_btn)
        shortcuts_buttons.addWidget(remove_shortcut_btn)
        shortcuts_buttons.addStretch()

        shortcuts_layout.addWidget(self.shortcuts_app_list)
        shortcuts_layout.addLayout(shortcuts_buttons)

        # Add both sections to main layout
        app_space_layout.addWidget(general_apps_group)
        app_space_layout.addWidget(shortcuts_group)

        tab_widget.addTab(app_space_widget, "App Space")

        # Add tab widget to main layout
        layout.addWidget(tab_widget)

        # Dialog buttons with proper spacing
        button_container = QWidget()
        button_container_layout = QHBoxLayout(button_container)
        button_container_layout.setContentsMargins(10, 10, 10, 10)

        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        button_container_layout.addWidget(button_box)
        layout.addWidget(button_container)

        # Load app list
        self.load_app_list()

        # Apply styling
        self.setStyleSheet("""
            QFrame#settingsGroup {
                background-color: #F5F5F5;
                border-radius: 10px;
                padding: 15px;
                margin: 5px;
            }
            QSpinBox, QDoubleSpinBox {
                min-width: 100px;
                max-width: 100px;
            }
            QHBoxLayout {
                margin: 5px;
            }
            QLabel {
                font-weight: bold;
                margin-top: 10px;
            }
            QLineEdit {
                padding: 5px;
                border: 1px solid #CCCCCC;
                border-radius: 5px;
            }
        """)

        # Add startup option to performance group
        startup_container = QWidget()
        startup_layout = QHBoxLayout(startup_container)
        startup_layout.addWidget(QLabel("Start with Windows:"))
        self.startup_check = QCheckBox()
        # Reflect actual registry state rather than only saved settings
        self.startup_check.setChecked(is_startup_enabled())
        startup_layout.addWidget(self.startup_check)
        performance_layout.addWidget(startup_container)

    def load_app_list(self):
        try:
            with open("app_space_map.json", "r") as f:
                app_space_data = json.load(f)

            # Clear existing items
            for i in reversed(range(self.general_app_list_layout.count())):
                self.general_app_list_layout.itemAt(i).widget().setParent(None)
            for i in reversed(range(self.shortcuts_app_list_layout.count())):
                self.shortcuts_app_list_layout.itemAt(i).widget().setParent(None)

            # Add apps with selection handling
            def create_app_button(name, section):
                btn = QPushButton(name)
                btn.setProperty("selected", False)
                btn.setProperty("section", section)
                btn.clicked.connect(lambda: self.select_app_button(btn))
                btn.mouseDoubleClickEvent = lambda e: self.edit_app(section)
                return btn

            # Add general apps
            for app_name, data in app_space_data.items():
                if app_name != "keyboard_shortcuts":
                    self.general_app_list_layout.addWidget(
                        create_app_button(app_name, "UI Elements")
                    )

            # Add keyboard shortcuts
            if "keyboard_shortcuts" in app_space_data:
                for app_name in app_space_data["keyboard_shortcuts"]:
                    self.shortcuts_app_list_layout.addWidget(
                        create_app_button(app_name, "Keyboard Shortcuts")
                    )
        except:
            pass

    def select_app_button(self, clicked_button):
        # Deselect all buttons in both sections
        for btn in self.general_app_list.widget().findChildren(QPushButton):
            btn.setProperty("selected", False)
            btn.setStyleSheet("")
        for btn in self.shortcuts_app_list.widget().findChildren(QPushButton):
            btn.setProperty("selected", False)
            btn.setStyleSheet("")

        # Select clicked button
        clicked_button.setProperty("selected", True)
        clicked_button.setStyleSheet("""
            QPushButton {
                background-color: #1976D2;
                color: white;
            }
        """)

    def add_app(self, section_type):
        dialog = AppContextDialog(self, is_shortcut=(section_type == "Keyboard Shortcuts"))

        if dialog.exec() == QDialog.Accepted:
            app_name, context_data = dialog.get_data()
            if app_name is None or context_data is None:
                return

            try:
                with open("app_space_map.json", "r") as f:
                    app_space_data = json.load(f)
            except:
                app_space_data = {"keyboard_shortcuts": {}}

            if section_type == "Keyboard Shortcuts":
                if "keyboard_shortcuts" not in app_space_data:
                    app_space_data["keyboard_shortcuts"] = {}
                app_space_data["keyboard_shortcuts"][app_name] = context_data
            else:
                app_space_data[app_name] = context_data

            with open("app_space_map.json", "w") as f:
                json.dump(app_space_data, f, indent=2)

            self.load_app_list()

    def edit_app(self, section_type=None):
        if section_type is None:
            selected = next((btn for btn in self.general_app_list.widget().findChildren(QPushButton)
                            if btn.property("selected")), None)
            if not selected:
                selected = next((btn for btn in self.shortcuts_app_list.widget().findChildren(QPushButton)
                               if btn.property("selected")), None)
        else:
            if section_type == "UI Elements":
                selected = next((btn for btn in self.general_app_list.widget().findChildren(QPushButton)
                               if btn.property("selected")), None)
            else:
                selected = next((btn for btn in self.shortcuts_app_list.widget().findChildren(QPushButton)
                               if btn.property("selected")), None)

        if not selected:
            return

        old_app_name = selected.text()
        section_type = selected.property("section")
        is_shortcut = section_type == "Keyboard Shortcuts"

        try:
            with open("app_space_map.json", "r") as f:
                app_space_data = json.load(f)

            if is_shortcut:
                context_data = app_space_data["keyboard_shortcuts"].get(old_app_name, [])
            else:
                context_data = app_space_data.get(old_app_name, {})

            dialog = AppContextDialog(self, old_app_name, context_data, is_shortcut)

            if dialog.exec() == QDialog.Accepted:
                new_app_name, new_context_data = dialog.get_data()

                # Remove old entry
                if is_shortcut:
                    if old_app_name in app_space_data["keyboard_shortcuts"]:
                        del app_space_data["keyboard_shortcuts"][old_app_name]
                else:
                    if old_app_name in app_space_data:
                        del app_space_data[old_app_name]

                # Add new entry
                if section_type == "Keyboard Shortcuts":
                    if "keyboard_shortcuts" not in app_space_data:
                        app_space_data["keyboard_shortcuts"] = {}
                    app_space_data["keyboard_shortcuts"][new_app_name] = new_context_data
                else:
                    app_space_data[new_app_name] = new_context_data

                with open("app_space_map.json", "w") as f:
                    json.dump(app_space_data, f, indent=2)

                self.load_app_list()
        except Exception as e:
            QMessageBox.warning(
                self,
                "Error",
                f"Error editing app: {str(e)}"
            )

    def remove_app(self, section_type=None):
        if section_type is None:
            selected = next((btn for btn in self.general_app_list.widget().findChildren(QPushButton)
                            if btn.property("selected")), None)
            if not selected:
                selected = next((btn for btn in self.shortcuts_app_list.widget().findChildren(QPushButton)
                               if btn.property("selected")), None)
        else:
            if section_type == "UI Elements":
                selected = next((btn for btn in self.general_app_list.widget().findChildren(QPushButton)
                               if btn.property("selected")), None)
            else:
                selected = next((btn for btn in self.shortcuts_app_list.widget().findChildren(QPushButton)
                               if btn.property("selected")), None)

        if not selected:
            return

        # Show confirmation dialog
        confirm = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete '{selected.text()}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if confirm == QMessageBox.Yes:
            app_name = selected.text()
            section_type = selected.property("section")
            is_shortcut = section_type == "Keyboard Shortcuts"

            try:
                with open("app_space_map.json", "r") as f:
                    app_space_data = json.load(f)

                if is_shortcut:
                    del app_space_data["keyboard_shortcuts"][app_name]
                else:
                    del app_space_data[app_name]

                with open("app_space_map.json", "w") as f:
                    json.dump(app_space_data, f, indent=2)

                self.load_app_list()
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Error",
                    f"Error removing app: {str(e)}"
                )

    def accept(self):
        # Get current startup state
        current_startup = is_startup_enabled()

        # Get new startup state
        new_startup = self.startup_check.isChecked()

        # Update startup if changed
        if current_startup != new_startup:
            if new_startup:
                enable_startup()
            else:
                disable_startup()

        super().accept()

def _enable_startup_powershell():
    """Create Startup shortcut via PowerShell (no pywin32 dependency)."""
    try:
        # Determine target, working dir, and args
        if getattr(sys, 'frozen', False):
            exe_path = sys.executable
            working_dir = os.path.dirname(sys.executable)
            args = ""
        else:
            exe_path = sys.executable
            working_dir = os.path.dirname(os.path.abspath(__file__))
            args = f'"{os.path.abspath(sys.argv[0])}"'

        startup_folder = os.path.join(
            os.environ['APPDATA'], 'Microsoft', 'Windows', 'Start Menu', 'Programs', 'Startup'
        )
        shortcut_path = os.path.join(startup_folder, 'ComputerAutopilot.lnk')

        # Build a PowerShell one-liner to create the .lnk
        ps = (
            f"$WshShell = New-Object -ComObject WScript.Shell; "
            f"$Shortcut = $WshShell.CreateShortcut('{shortcut_path}'); "
            f"$Shortcut.TargetPath = '{exe_path}'; "
            f"$Shortcut.WorkingDirectory = '{working_dir}'; "
            f"$Shortcut.Arguments = '{args}'; "
            f"$Shortcut.Description = 'Computer Autopilot - AI Assistant'; "
            f"$Shortcut.Save()"
        )

        import subprocess
        result = subprocess.run([
            "powershell", "-NoProfile", "-Command", ps
        ], capture_output=True, text=True)
        if result.returncode != 0:
            if result.stderr:
                print(f"PowerShell error creating startup shortcut: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"Error in PowerShell fallback for startup: {e}")
        return False

def enable_startup():
    """Enable app to start with Windows using Startup folder shortcut"""
    try:
        import win32com.client

        # Get the correct executable path
        if getattr(sys, 'frozen', False):
            # PyInstaller executable
            exe_path = sys.executable
            working_dir = os.path.dirname(sys.executable)
        else:
            # Running from Python script
            exe_path = sys.executable
            working_dir = os.path.dirname(os.path.abspath(__file__))
            # For .py files, we need to pass the script as an argument; handled below

        # Get Startup folder path
        startup_folder = os.path.join(
            os.environ['APPDATA'],
            'Microsoft', 'Windows', 'Start Menu', 'Programs', 'Startup'
        )

        # Create shortcut path
        shortcut_path = os.path.join(startup_folder, 'ComputerAutopilot.lnk')

        # Create shortcut using COM
        shell = win32com.client.Dispatch("WScript.Shell")
        shortcut = shell.CreateShortCut(shortcut_path)

        if getattr(sys, 'frozen', False):
            # For PyInstaller executable
            shortcut.Targetpath = exe_path
            shortcut.WorkingDirectory = working_dir
            shortcut.Arguments = ""
        else:
            # For Python script
            shortcut.Targetpath = exe_path
            shortcut.WorkingDirectory = working_dir
            shortcut.Arguments = f'"{os.path.abspath(sys.argv[0])}"'

        shortcut.Description = "Computer Autopilot - AI Assistant"
        shortcut.save()

        return True
    except ImportError:
        # Fallback to PowerShell if win32com is not available
        return _enable_startup_powershell()
    except Exception as e:
        print(f"Error enabling startup: {e}")
        return False

def disable_startup():
    """Disable app from starting with Windows by removing Startup shortcut (and legacy registry value)."""
    try:
        startup_folder = os.path.join(
            os.environ['APPDATA'], 'Microsoft', 'Windows', 'Start Menu', 'Programs', 'Startup'
        )
        shortcut_path = os.path.join(startup_folder, 'ComputerAutopilot.lnk')
        if os.path.exists(shortcut_path):
            os.remove(shortcut_path)
        # Also remove legacy registry value if present (best-effort)
        try:
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\\Microsoft\\Windows\\CurrentVersion\\Run",
                0,
                winreg.KEY_SET_VALUE
            )
            try:
                winreg.DeleteValue(key, "ComputerAutopilot")
            except WindowsError:
                pass
            winreg.CloseKey(key)
        except Exception:
            pass
        return True
    except Exception as e:
        print(f"Error disabling startup: {e}")
        return False

def is_startup_enabled():
    """Check if app is set to start with Windows by verifying Startup shortcut exists."""
    try:
        startup_folder = os.path.join(
            os.environ['APPDATA'], 'Microsoft', 'Windows', 'Start Menu', 'Programs', 'Startup'
        )
        shortcut_path = os.path.join(startup_folder, 'ComputerAutopilot.lnk')
        return os.path.exists(shortcut_path)
    except Exception as e:
        print(f"Error checking startup: {e}")
        return False

class VoiceHandler(QObject):
    textReceived = Signal(str)

class ModernChatInterface(QMainWindow):
    def __init__(self):
        super().__init__()

        # Store reference to browser cleanup
        self.close_browser = close_browser

        # Application settings
        self.setWindowTitle("Computer Autopilot")
        self.setMinimumSize(1000, 700)

        # Set window icon (this ensures consistency with taskbar)
        self.setWindowIcon(QApplication.windowIcon())

        # System tray support
        self._tray_quit = False
        self._tray_message_shown = False
        if QSystemTrayIcon.isSystemTrayAvailable():
            # Call helper if present; otherwise create a minimal tray icon inline
            if hasattr(self, "_create_tray_icon"):
                self._create_tray_icon()
            else:
                try:
                    self.tray_icon = QSystemTrayIcon(QApplication.windowIcon(), self)
                    self.tray_icon.setToolTip("Computer Autopilot")
                    self.tray_icon.show()
                except Exception:
                    pass


        # Initialize main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Chat area
        self.chat_area = ChatArea()
        layout.addWidget(self.chat_area)

        # In-memory chat history (list of {role, content}) for conversation context
        self.chat_history = []

        # In-memory chat history for context to the assistant
        self.chat_history = []

        # Input area
        input_container = QWidget()
        input_layout = QHBoxLayout(input_container)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type the task you want to perform...")
        self.input_field.setMinimumHeight(40)
        input_layout.addWidget(self.input_field)

        # Control buttons
        control_container = QWidget()
        control_layout = QHBoxLayout(control_container)
        control_layout.setContentsMargins(0, 0, 0, 0)

        self.send_stop_button = QPushButton("Send")
        self.send_stop_button.setMinimumHeight(40)
        self.pause_resume_button = QPushButton("Pause")
        self.pause_resume_button.setMinimumHeight(40)
        self.pause_resume_button.setEnabled(False)

        control_layout.addWidget(self.send_stop_button)
        control_layout.addWidget(self.pause_resume_button)
        input_layout.addWidget(control_container)

        # Voice button with improved styling
        self.voice_button = QPushButton("🎤")
        self.voice_button.setFixedSize(40, 40)
        emoji_font = QFont()
        emoji_font.setPointSize(16)
        emoji_font.setFamily("Segoe UI Emoji")
        self.voice_button.setFont(emoji_font)
        self.voice_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 0px;
                margin: 0px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        input_layout.addWidget(self.voice_button)

        layout.addWidget(input_container)

        # Volume controls
        volume_container = QWidget()
        volume_layout = QHBoxLayout(volume_container)

        self.clear_chat_button = QPushButton("Clear Chat")
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(80)

        self.mute_button = QPushButton("🔊")
        self.mute_button.setFixedSize(40, 30)

        self.subtitles_checkbox = QCheckBox("Subtitles")
        self.subtitles_checkbox.setChecked(True)

        self.settings_button = QPushButton("Settings")

        volume_layout.addWidget(self.settings_button)
        volume_layout.addWidget(self.clear_chat_button)  # Add clear chat button
        volume_layout.addWidget(self.volume_slider)
        volume_layout.addWidget(self.mute_button)
        volume_layout.addWidget(self.subtitles_checkbox)

        layout.addWidget(volume_container)

        # Initialize state variables
        self.is_processing = False
        self.paused_execution = False
        self.voice_active = False
        self.is_muted = False
        self.current_language = 'en'
        self.paused_goal = None
        self.paused_actions = []
        self.additional_context = None

        # Add voice handler
        self.voice_handler = VoiceHandler()
        self.voice_handler.textReceived.connect(self.handle_voice_text)

        # Connect signals
        self.input_field.returnPressed.connect(self.send_message)
        self.send_stop_button.clicked.connect(self.handle_send_stop)
        self.pause_resume_button.clicked.connect(self.handle_pause_resume)
        self.voice_button.clicked.connect(self.toggle_voice_input)
        self.volume_slider.valueChanged.connect(self.update_volume)
        self.mute_button.clicked.connect(self.toggle_mute)
        self.subtitles_checkbox.stateChanged.connect(self.toggle_subtitles)
        self.settings_button.clicked.connect(self.open_settings_window)

        # Connect clear chat button (integrated with session clear)
        self.clear_chat_button.clicked.connect(lambda: self._on_desktop_clear_messages())

        # Apply modern styling
        self.apply_styling()

        # Set application icon
        self.setWindowIcon(QIcon("media/icon.png"))

        # Add voice shortcut
        self.voice_shortcut = QShortcut(QKeySequence("Alt+V"), self)
        self.voice_shortcut.activated.connect(self.toggle_voice_input)

        # Create mini chat interface
        self.mini_chat = MiniChatInterface(self)
        self.mini_chat.input_field.returnPressed.connect(self.handle_mini_chat_input)
        self.mini_chat.voice_button.clicked.connect(self.toggle_voice_input)

        # Create mini control interface
        self.mini_control = MiniControlInterface(self)

        # Create tab widget
        self.tab_widget = QTabWidget()

        # Create chat tab
        chat_tab = QWidget()
        chat_layout = QVBoxLayout(chat_tab)

        # Sessions bar for Desktop Chat
        self.desktop_session_combo = QComboBox()
        self.desktop_new_btn = QPushButton("New Session")
        self.desktop_rename_btn = QPushButton("Rename")
        self.desktop_delete_btn = QPushButton("Delete")
        sess_bar = QWidget()
        sess_bar_layout = QHBoxLayout(sess_bar)
        sess_bar_layout.setContentsMargins(0, 0, 0, 0)
        sess_bar_layout.addWidget(QLabel("Session:"))
        sess_bar_layout.addWidget(self.desktop_session_combo)
        sess_bar_layout.addStretch(1)
        sess_bar_layout.addWidget(self.desktop_new_btn)
        sess_bar_layout.addWidget(self.desktop_rename_btn)
        sess_bar_layout.addWidget(self.desktop_delete_btn)
        chat_layout.addWidget(sess_bar)

        # Desktop chat session state
        self.desktop_current_session_id = None
        self._desktop_restoring = False

        # Connect session controls
        self.desktop_session_combo.currentIndexChanged.connect(self._on_desktop_session_changed)
        self.desktop_new_btn.clicked.connect(self._on_desktop_new_session)
        self.desktop_rename_btn.clicked.connect(self._on_desktop_rename_session)
        self.desktop_delete_btn.clicked.connect(self._on_desktop_delete_session)

        # Initialize sessions per preference (always start new)
        self._init_desktop_sessions()

        # Move existing chat widgets to chat tab
        chat_layout.addWidget(self.chat_area)
        chat_layout.addWidget(input_container)
        chat_layout.addWidget(volume_container)

        # Create RPA tab
        rpa_tab = RPATab()

        # Create Web Agent tab
        web_tab = WebAgentTab()

        # Add tabs
        self.tab_widget.addTab(chat_tab, "Chat")
        self.tab_widget.addTab(web_tab, "Web Agent")
        self.tab_widget.addTab(rpa_tab, "RPA")

        # Set central widget to tab widget
        self.setCentralWidget(self.tab_widget)

        # Connect the status update signal from tabs to MiniControlInterface
        rpa_tab.statusUpdate.connect(self.mini_control.update_status)
        web_tab.statusUpdate.connect(self.mini_control.update_status)

        # Connect mini control signals
        self.mini_control.play_pause_button.clicked.connect(self.handle_mini_control_play_pause)
        self.mini_control.stop_button.clicked.connect(self.handle_mini_control_stop)

        # Connect tab change signal
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

    def _create_tray_icon(self):
        """Create and show the system tray icon with context menu."""
        try:
            icon = QApplication.windowIcon()
            self.tray_icon = QSystemTrayIcon(icon, self)
            self.tray_icon.setToolTip("Computer Autopilot")

            # Context menu
            menu = QMenu()
            action_show = QAction("Show", self)
            action_show.triggered.connect(self.restore_from_tray)
            menu.addAction(action_show)

            action_quit = QAction("Quit", self)
            action_quit.triggered.connect(self.quit_from_tray)
            menu.addAction(action_quit)

            self.tray_icon.setContextMenu(menu)
            self.tray_icon.activated.connect(self.on_tray_activated)
            self.tray_icon.show()
        except Exception as e:
            print(f"Error creating system tray icon: {e}")

    def restore_from_tray(self):
        try:
            self.showNormal()
            self.raise_()
            self.activateWindow()
        except Exception:
            pass

    def quit_from_tray(self):
        """Quit the application from the tray menu, ensuring background threads stop."""
        self._tray_quit = True
        # Hide tray icon if present
        try:
            if hasattr(self, 'tray_icon'):
                self.tray_icon.hide()
        except Exception:
            pass
        # Attempt to stop any running tasks/threads gracefully
        try:
            if hasattr(self, 'tab_widget'):
                for i in range(self.tab_widget.count()):
                    tab = self.tab_widget.widget(i)
                    if hasattr(tab, 'stop_execution'):
                        try:
                            tab.stop_execution()
                        except Exception:
                            pass
        except Exception:
            pass
        # Close browser/resources if available
        try:
            if hasattr(self, 'close_browser'):
                self.close_browser()
        except Exception:
            pass
        # Request application quit
        try:
            QApplication.instance().quit()
        except Exception:
            # As last resort
            sys.exit(0)

    def on_tray_activated(self, reason):
        try:
            # Single click or double click restores window
            if reason in (QSystemTrayIcon.Trigger, QSystemTrayIcon.DoubleClick):
                self.restore_from_tray()
        except Exception:
            pass

    def changeEvent(self, event):
        if (event.type() == QEvent.Type.WindowStateChange):
            if self.windowState() & Qt.WindowState.WindowMinimized:
                # Show appropriate mini interface based on active tab
                if self.tab_widget.currentIndex() == 2:  # RPA tab is now index 2
                    self.mini_control.show()
                    self.update_mini_control()
                else:
                    self.mini_chat.show()
            else:
                self.mini_chat.hide()
                self.mini_control.hide()
        super().changeEvent(event)

    def handle_mini_chat_input(self):
        """Handle input from mini chat interface"""
        message = self.mini_chat.input_field.text().strip()
        if message:
            self.mini_chat.input_field.clear()
            # Process message without restoring main window
            self.process_mini_chat_message(message)

    def process_mini_chat_message(self, message):
        """Process message from mini chat without restoring main window"""
        if not self.is_processing:
            # Detect language of input
            try:
                self.current_language = detect(message)
                if self.current_language not in ['en', 'vi']:
                    self.current_language = 'en'
            except:
                self.current_language = 'en'

            # Check which tab is currently active to route the message appropriately
            current_tab_index = self.tab_widget.currentIndex()

            # If Web Agent tab is active (index 1), send to web assistant
            if current_tab_index == 1:  # Web Agent tab
                web_tab = self.tab_widget.widget(1)
                if web_tab and not web_tab.is_processing:
                    # Add user message to web tab
                    web_tab.messageReceived.emit(message, True)
                    # Process in web agent tab
                    web_tab.set_stop_mode()
                    threading.Thread(target=web_tab.process_message, args=(message,)).start()
                return

            # Otherwise, process in the main chat tab
            # Add user message to chat
            print_to_chat(message, is_user=True)

            # Process message in the main chat interface
            self.process_message(message)

    def apply_styling(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #FFFFFF;
            }
            QLineEdit {
                padding: 8px;
                border: 1px solid #CCCCCC;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #FFFFFF;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #2196F3;
                border: none;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """)

    def add_message(self, message, is_user=True):
        """Add a message bubble to the chat area and store in chat history"""
        bubble = MessageBubble(message, is_user)
        self.chat_area.layout.addWidget(bubble)

        # Update in-memory chat history for context building
        role = "user" if is_user else "assistant"
        prev_len = len(getattr(self, 'chat_history', []) or [])
        try:
            self.chat_history.append({"role": role, "content": str(message)})
        except Exception:
            # Best effort; do not break UI if chat_history isn't available
            pass

        # Persist to current desktop session if applicable and not in restore mode
        try:
            if hasattr(self, 'desktop_current_session_id') and self.desktop_current_session_id and not getattr(self, '_desktop_restoring', False):
                chat_add_message(self.desktop_current_session_id, role, str(message))
                # If this is the first user message in a new session, auto-name it
                if is_user and prev_len == 0:
                    first_line = str(message).strip()
                    if first_line:
                        chat_rename_session(self.desktop_current_session_id, first_line[:60])
                        self._refresh_desktop_sessions(select_id=self.desktop_current_session_id)
        except Exception:
            pass

        # Auto scroll to bottom
        self.chat_area.verticalScrollBar().setValue(
            self.chat_area.verticalScrollBar().maximum()
        )

    def build_conversation_context(self, max_messages: int = 20, max_chars: int = 6000) -> str:
        """Build a compact conversation history string for model context.
        Includes the most recent messages up to limits, labeled by speaker.
        """
        try:
            history = getattr(self, "chat_history", []) or []
            recent = history[-max_messages:]
            lines = []
            for msg in recent:
                role = msg.get("role", "assistant")
                speaker = "User" if role == "user" else "Assistant"
                content = str(msg.get("content", ""))
                lines.append(f"{speaker}: {content}")
            text = "\n".join(lines)
            # Enforce rough char budget by trimming from the start
            if len(text) > max_chars:
                text = text[-max_chars:]
            return f"Conversation history (most recent first at bottom):\n{text}"
        except Exception:
            return ""


    def handle_send_stop(self):
        """Handle send/stop button click based on current state"""
        if self.is_processing:
            self.stop_execution()
        else:
            self.send_message()

    @Slot()
    def set_send_mode(self):
        """Set button to send mode"""
        self.send_stop_button.setText("Send")
        self.send_stop_button.setStyleSheet("QPushButton { background-color: #2196F3; }")
        # self.send_stop_button.setStyleSheet("""
        #     QPushButton {
        #         background-color: #2196F3;
        #     }
        #     QPushButton:hover {
        #         background-color: #1976D2;
        #     }
        # """)
        self.pause_resume_button.setEnabled(False)
        self.pause_resume_button.setText("Pause")
        self.input_field.setEnabled(True)
        self.voice_button.setEnabled(True)
        self.is_processing = False
        self.paused_execution = False
        self.paused_goal = None
        self.paused_actions = None
        self.additional_context = None

    @Slot()
    def set_stop_mode(self):
        """Set button to stop mode"""
        self.send_stop_button.setText("Stop")
        self.send_stop_button.setStyleSheet("QPushButton { background-color: #F44336; }")
        # self.send_stop_button.setStyleSheet("""
        #     QPushButton {
        #         background-color: #F44336;
        #     }
        #     QPushButton:hover {
        #         background-color: #D32F2F;
        #     }
        # """)
        self.pause_resume_button.setEnabled(True)
        self.input_field.setEnabled(False)
        self.voice_button.setEnabled(False)
        self.is_processing = True

    def stop_execution(self):
        """Stop the current task execution"""
        request_stop()
        self.set_send_mode()

    def complete_task(self, result=None, speak=True):
        """Helper function to handle task completion"""
        if not self.paused_execution:
            self.set_send_mode()
            if result:
                print_to_chat(result, is_user=False)
                if speak:
                    speaker(result)

    def handle_pause_resume(self):
        """Handle pause/resume button click"""
        if self.paused_execution:
            # Resume execution
            self.pause_resume_button.setText("Pause")
            self.paused_execution = False
            additional_info = self.input_field.text().strip()
            self.input_field.clear()
            if additional_info:
                self.additional_context = additional_info
            # Resume the task execution
            if self.paused_goal:
                threading.Thread(target=self.process_message, args=(self.paused_goal, True)).start()
        else:
            # Pause execution
            self.pause_resume_button.setText("Resume")
            self.paused_execution = True
            request_stop()
            # Use signal for thread-safe message display
            self.messageReceived.emit("Task execution paused. Enter additional context if needed.", False)
            self.input_field.setEnabled(True)

    def send_message(self):
        message = self.input_field.text().strip()
        if message and not self.is_processing:
            # Detect language of input
            try:
                self.current_language = detect(message)
                if self.current_language not in ['en', 'vi']:
                    self.current_language = 'en'
            except:
                self.current_language = 'en'

            # Add user message to chat
            print_to_chat(message, is_user=True)

            # Clear input field
            self.input_field.clear()

            # Process message
            self.process_message(message)

    def process_message(self, message, resumed=False):
        """Process message and execute assistant"""
        def process():
            try:
                if not resumed:
                    if not self.is_processing:
                        # self.set_stop_mode()
                        QMetaObject.invokeMethod(
                            self,
                            "set_stop_mode",
                            Qt.ConnectionType.QueuedConnection
                        )

                    # Store goal for potential pause/resume
                    self.paused_goal = message
                    self.paused_actions = []

                message_lower = message.lower()

                # Handle simple tasks first
                if message_lower.startswith("open ") and len(message_lower) < 20:
                    window_title = message_lower.split("open ")[-1].strip()
                    activate_window_title(window_title)
                    self.complete_task(f"Activated Window: {window_title}")
                elif message_lower.startswith("scroll ") and len(message_lower) < 20:
                    scroll_amount = 850
                    if "up" in message_lower:
                        pyautogui.scroll(scroll_amount)
                        self.complete_task("Scrolled up")
                    elif "right" in message_lower:
                        pyautogui.hscroll(-scroll_amount)
                        self.complete_task("Scrolled right")
                    elif "left" in message_lower:
                        pyautogui.hscroll(scroll_amount)
                        self.complete_task("Scrolled left")
                    else:
                        pyautogui.scroll(-scroll_amount)
                        self.complete_task("Scrolled down")
                elif message_lower.startswith("press ") and len(message_lower) < 20:
                    key = message_lower.split("press ")[-1].strip()
                    perform_simulated_keypress(key)
                    self.complete_task(f"Pressed {key}")
                elif message_lower.startswith("type "):
                    text = message_lower.split("type ")[-1].strip()
                    write_action(goal=text)
                    self.complete_task(f"Typed: {text}")
                elif message_lower.startswith("click ") and len(message_lower) < 30:
                    response = fast_act(message_lower, action_type="single")
                    self.complete_task(response)
                elif message_lower.startswith("double click ") and len(message_lower) < 37:
                    response = fast_act(message_lower.split("double ")[-1].strip(), action_type="double")
                    self.complete_task(response)
                else:
                    # For complex tasks, determine role and use assistant
                    # response = auto_role(message)
                    # if response:
                    #     if "windows_assistant" in response:
                    #         result = assistant(
                    #             goal=message,
                    #             executed_actions=self.paused_actions,
                    #             additional_context=self.additional_context,
                    #             resumed=resumed
                    #         )
                    #         self.complete_task(result)
                    #     else:
                    #         joyful_response = api_call(
                    #             [{"role": "user", "content": message}],
                    #             temperature=0.7,
                    #             max_tokens=1000
                    #         )
                    #         self.complete_task(joyful_response, speak=False)
                    # Build conversation context to include entire chat history (not just executed actions)
                    conv_context = self.build_conversation_context(max_messages=20, max_chars=6000)
                    addl = self.additional_context if self.additional_context else ""
                    combined_context = conv_context if not addl else f"{conv_context}\n\nAdditional Context: {addl}"
                    result = assistant(
                        goal=message,
                        executed_actions=self.paused_actions,
                        additional_context=combined_context,
                        resumed=resumed
                    )
                    self.complete_task(result)

                # if not self.isActiveWindow():
                #     self.show()
                #     self.raise_()
                #     self.activateWindow()

            except Exception as e:
                error_message = f"Error: {str(e)}"
                print_to_chat(error_message, is_user=False)
                speaker(error_message)
                # self.set_send_mode()
                QMetaObject.invokeMethod(
                    self,
                    "set_send_mode",
                    Qt.ConnectionType.QueuedConnection
                )
            finally:
                if not self.paused_execution:
                    clear_stop()

        threading.Thread(target=process).start()

    def toggle_voice_input(self):
        """Toggle voice input for web agent"""
        if not self.voice_active:
            self.voice_active = True
            self.voice_button.setStyleSheet("""
                QPushButton {
                    background-color: #F44336;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    padding: 0px;
                    margin: 0px;
                    font-size: 16px;
                }
                QPushButton:hover {
                    background-color: #D32F2F;
                }
            """)
            # Start listening using the existing voice_handler
            threading.Thread(target=self.listen_voice, daemon=True).start()
        else:
            self.voice_active = False
            self.voice_button.setStyleSheet("""
                QPushButton {
                    background-color: #2196F3;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    padding: 0px;
                    margin: 0px;
                    font-size: 16px;
                }
                QPushButton:hover {
                    background-color: #1976D2;
                }
            """)

    def listen_voice(self):
        """Listen for voice input and convert to text"""
        with sr.Microphone() as source:
            try:
                # Don't use global speaker, use signal to add message
                self.messageReceived.emit("I'm listening...", False)

                recognizer = sr.Recognizer()
                audio = recognizer.listen(source, timeout=5)
                try:
                    text = recognizer.recognize_google(audio, language='vi-VN')
                    self.current_language = 'vi'
                except:
                    text = recognizer.recognize_google(audio, language='en-US')
                    self.current_language = 'en'

                # Thread-safe way to handle recognized text
                if text:
                    # Set text in input field
                    QMetaObject.invokeMethod(
                        self.input_field,
                        "setText",
                        Qt.ConnectionType.QueuedConnection,
                        Q_ARG(str, text)
                    )

                    # Call send_message method
                    QMetaObject.invokeMethod(
                        self,
                        "send_message",
                        Qt.ConnectionType.QueuedConnection
                    )
            except Exception as e:
                self.messageReceived.emit(f"Voice recognition error: {str(e)}", False)
            finally:
                self.voice_active = False

                # Update button style in main thread
                QMetaObject.invokeMethod(
                    self,
                    "reset_voice_button",
                    Qt.ConnectionType.QueuedConnection
                )

    def handle_voice_text(self, text):
        """Handle voice recognition text in the main thread"""
        self.input_field.setText(text)
        self.send_message()

    def update_volume(self, value):
        if not self.is_muted:
            set_volume(value)

    def toggle_mute(self):
        self.is_muted = not self.is_muted
        if self.is_muted:
            self.mute_button.setText("🔇")
            set_volume(0)
        else:
            self.mute_button.setText("🔊")
            set_volume(self.volume_slider.value())

    def toggle_subtitles(self):
        set_subtitles(self.subtitles_checkbox.isChecked())

    def open_settings_window(self):
        """Open settings dialog"""
        # Load current settings
        settings = load_settings()

        # Create settings dialog with current settings
        dialog = SettingsDialog(
            self,
            llm_model=settings.get("llm_model", ""),
            api_key_env_name=settings.get("api_key_env_name", ""),
            vision_llm_model=settings.get("vision_llm_model", ""),
            vision_api_key_env_name=settings.get("vision_api_key_env_name", ""),
            action_delay=settings.get("action_delay", 1.5),
            max_attempts=settings.get("max_attempts", 20),
            start_with_windows=settings.get("start_with_windows", False)
        )

        # Set web-specific settings if they exist
        if "web_action_delay" in settings:
            dialog.web_action_delay_spin.setValue(settings.get("web_action_delay"))
        if "web_max_attempts" in settings:
            dialog.web_max_attempts_spin.setValue(settings.get("web_max_attempts"))

        # If dialog is accepted, save settings
        if dialog.exec() == QDialog.Accepted:
            # Collect provider-specific options
            def collect_opts(container):
                opts = {}
                if hasattr(container, "_fields"):
                    for k, le in container._fields.items():
                        v = le.text().strip()
                        if v:
                            opts[k] = v
                return opts

            llm_opts = collect_opts(dialog.llm_provider_opts_container)
            vis_opts = collect_opts(dialog.vision_provider_opts_container)

            # Get settings from dialog
            new_settings = {
                "llm_model": f"{dialog.llm_provider_combo.currentText()}/{dialog.llm_model_combo.currentText().strip()}",
                "api_key_env_name": dialog.api_key_env_name_entry.text(),
                "vision_llm_model": f"{dialog.vision_provider_combo.currentText()}/{dialog.vision_model_combo.currentText().strip()}",
                "vision_api_key_env_name": dialog.vision_api_key_env_name_entry.text(),
                "llm_provider_options": llm_opts,
                "vision_provider_options": vis_opts,
                "action_delay": dialog.action_delay_spin.value(),
                "max_attempts": dialog.max_attempts_spin.value(),
                "web_action_delay": dialog.web_action_delay_spin.value(),
                "web_max_attempts": dialog.web_max_attempts_spin.value(),
                "start_with_windows": dialog.startup_check.isChecked()
            }

            # Set API credentials
            api_key = dialog.api_key_entry.text()
            vision_api_key = dialog.vision_api_key_entry.text()

            set_llm_model(new_settings["llm_model"], new_settings.get("llm_provider_options", {}))
            set_api_key(api_key, new_settings["api_key_env_name"])
            set_vision_llm_model(new_settings["vision_llm_model"], new_settings.get("vision_provider_options", {}))
            set_vision_api_key(vision_api_key, new_settings["vision_api_key_env_name"])

            # Save settings to file
            try:
                with open(SETTINGS_FILE, "w") as f:
                    json.dump(new_settings, f, indent=2)
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Error",
                    f"Error saving settings: {str(e)}"
                )

    def update_mini_control(self):
        """Update mini control interface state"""
        try:
            rpa_tab = self.tab_widget.widget(2)  # RPA tab is now index 2
            if rpa_tab:
                current_task = rpa_tab.task_list.currentItem()
                if current_task:
                    self.mini_control.update_task(current_task.text())
                    self.mini_control.update_execution_state(
                        rpa_tab.is_executing,
                        rpa_tab.is_paused
                    )
                    self.mini_control.update_status(
                        rpa_tab.status_label.text(),
                        rpa_tab.is_executing,
                        rpa_tab.is_paused
                    )
        except Exception as e:
            print(f"Error in update_mini_control: {str(e)}")

    def on_tab_changed(self, index):
        """Handle tab changes to update mini interface behavior"""
        self.mini_chat.hide()
        self.mini_control.hide()

        # Update mini control if RPA tab is selected
        if index == 2:  # RPA tab is now index 2
            rpa_tab = self.tab_widget.widget(2)
            current_task = rpa_tab.task_list.currentItem()
            if current_task:
                self.update_mini_control()

    def handle_mini_control_play_pause(self):
        """Handle play/pause button click for mini control interface"""
        rpa_tab = self.tab_widget.widget(2)  # RPA tab is now index 2
        if rpa_tab:
            if rpa_tab.is_executing:
                # Handle pause/resume
                if rpa_tab.is_paused:
                    rpa_tab.handle_play_pause()
                else:
                    rpa_tab.handle_play_pause()
            else:
                # Start execution of selected task
                rpa_tab.execute_selected_task()
            self.update_mini_control()
    # --- Desktop chat session management ---
    def _init_desktop_sessions(self):
        """Start with a fresh session and load sessions list (preference 3.B)."""
        # Create a brand-new session with empty name; it'll be renamed on first user msg
        sid = chat_create_session('desktop', "")
        self.desktop_current_session_id = sid
        self._refresh_desktop_sessions(select_id=sid)
        # Nothing to load for a new session

    def _refresh_desktop_sessions(self, select_id=None):
        sessions = chat_list_sessions('desktop')
        self.desktop_session_combo.blockSignals(True)
        self.desktop_session_combo.clear()
        for sess in sessions:
            label_name = sess['name'] if sess['name'] else 'New chat'
            dt_str = _format_session_timestamp(sess.get('created_at'))
            label = f"{label_name} — {dt_str}" if dt_str else label_name
            self.desktop_session_combo.addItem(label, sess['id'])
        self.desktop_session_combo.blockSignals(False)
        # Select current
        if select_id is not None:
            for i in range(self.desktop_session_combo.count()):
                if self.desktop_session_combo.itemData(i) == select_id:
                    self.desktop_session_combo.setCurrentIndex(i)
                    break

    def _delete_desktop_session_if_empty(self, session_id: int) -> bool:
        """Delete the given desktop session if it has no messages. Returns True if deleted."""
        try:
            if not session_id:
                return False
            msgs = chat_list_messages(session_id)
            if not msgs:
                chat_delete_session(session_id)
                return True
            return False
        except Exception:
            return False

    def _on_desktop_session_changed(self, index):
        if index < 0:
            return
        new_id = self.desktop_session_combo.itemData(index)
        if not new_id or new_id == self.desktop_current_session_id:
            return
        old_id = getattr(self, 'desktop_current_session_id', None)
        if old_id and old_id != new_id:
            if self._delete_desktop_session_if_empty(old_id):
                # Refresh list to drop the deleted session
                self._refresh_desktop_sessions(select_id=new_id)
        self.desktop_current_session_id = new_id
        self._load_desktop_session_messages(new_id)

    def _on_desktop_new_session(self):
        # If current is empty, remove it so empty sessions are not saved
        curr_id = getattr(self, 'desktop_current_session_id', None)
        if curr_id:
            self._delete_desktop_session_if_empty(curr_id)
        sid = chat_create_session('desktop', "")
        self.desktop_current_session_id = sid
        self._refresh_desktop_sessions(select_id=sid)
        self.clear_chat()

    def _on_desktop_rename_session(self):
        if not self.desktop_current_session_id:
            return
        current_name = ""
        try:
            for sess in chat_list_sessions('desktop'):
                if sess['id'] == self.desktop_current_session_id:
                    current_name = sess.get('name') or ""
                    break
        except Exception:
            pass
        new_name, ok = QInputDialog.getText(
            self, "Rename Session", "New name:", QLineEdit.Normal, current_name
        )
        if ok and new_name.strip():
            chat_rename_session(self.desktop_current_session_id, new_name.strip())
            self._refresh_desktop_sessions(select_id=self.desktop_current_session_id)

    def _on_desktop_delete_session(self):
        if not self.desktop_current_session_id:
            return
        confirm = QMessageBox.question(self, "Delete Session", "Delete this session and all messages?", QMessageBox.Yes|QMessageBox.No)
        if confirm == QMessageBox.Yes:
            chat_delete_session(self.desktop_current_session_id)
            # Start a new one after delete
            sid = chat_create_session('desktop', "")
            self.desktop_current_session_id = sid
            self._refresh_desktop_sessions(select_id=sid)
            self.clear_chat()

    def _on_desktop_clear_messages(self):
        if not self.desktop_current_session_id:
            return
        confirm = QMessageBox.question(self, "Clear Messages", "Clear all messages in this session?", QMessageBox.Yes|QMessageBox.No)
        if confirm == QMessageBox.Yes:
            chat_clear_session_messages(self.desktop_current_session_id)
            self.clear_chat()

    def _load_desktop_session_messages(self, session_id: int):
        # Restore messages to UI and in-memory context without re-saving
        self.clear_chat()
        self._desktop_restoring = True
        try:
            msgs = chat_list_messages(session_id)
            for m in msgs:
                is_user = (m['role'] == 'user')
                self.add_message(m['content'], is_user)
        finally:
            self._desktop_restoring = False

    def handle_mini_control_stop(self):
        """Handle stop button click for mini control interface"""
        rpa_tab = self.tab_widget.widget(2)  # RPA tab is now index 2
        if rpa_tab:
            rpa_tab.stop_execution()
            self.update_mini_control()

    def clear_chat(self):
        """Clear all messages from the chat area and reset conversation history."""
        # Remove all message bubbles
        while self.chat_area.layout.count():
            child = self.chat_area.layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Also clear in-memory conversation history used for context
        try:
            if hasattr(self, "chat_history") and isinstance(self.chat_history, list):
                self.chat_history.clear()
            else:
                self.chat_history = []
        except Exception:
            # Ensure we at least reset the reference
            self.chat_history = []

    def closeEvent(self, event):
        """Minimize to system tray on window close, unless quitting from tray."""
        # Prune any empty sessions so they aren't saved
        try:
            curr_desktop = getattr(self, 'desktop_current_session_id', None)
            if curr_desktop:
                self._delete_desktop_session_if_empty(curr_desktop)
        except Exception:
            pass
        try:
            web_tab = None
            if hasattr(self, 'tab_widget') and self.tab_widget.count() > 1:
                web_tab = self.tab_widget.widget(1)
            if web_tab and getattr(web_tab, 'web_current_session_id', None):
                web_tab._delete_web_session_if_empty(web_tab.web_current_session_id)
        except Exception:
            pass
        try:
            if QSystemTrayIcon.isSystemTrayAvailable() and not getattr(self, "_tray_quit", False):
                # Hide instead of closing
                event.ignore()
                self.hide()
                # Inform user once
                if not getattr(self, "_tray_message_shown", False) and hasattr(self, 'tray_icon'):
                    try:
                        self.tray_icon.showMessage(
                            "Computer Autopilot",
                            "Still running in the background. Use the tray icon to restore or quit.",
                            QSystemTrayIcon.Information,

                            3000
                        )
                        self._tray_message_shown = True
                    except Exception:
                        pass
                return
        except Exception:
            # If tray is not available or any error occurs, proceed to close
            pass

        # Clean up browser resources when app is closing
        try:
            if hasattr(self, 'close_browser'):
                self.close_browser()
        except Exception as e:
            print(f"Error closing browser: {str(e)}")

        # Accept the close event to quit the app
        event.accept()
        # If this close was initiated as a real quit (from tray), request app quit
        if getattr(self, "_tray_quit", False):
            try:
                QApplication.instance().quit()
            except Exception:
                pass

class MiniChatInterface(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent, Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setWindowTitle("")
        self.setFixedSize(300, 60)

        # Position at bottom right
        screen = QApplication.primaryScreen().geometry()
        self.move(screen.width() - 320, screen.height() - 120)

        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Input container
        input_container = QWidget()
        input_layout = QHBoxLayout(input_container)
        input_layout.setContentsMargins(0, 0, 0, 0)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type a task...")
        self.input_field.setMinimumHeight(30)
        input_layout.addWidget(self.input_field)

        self.voice_button = QPushButton("🎤")
        self.voice_button.setFixedSize(30, 30)
        # Improve emoji visibility
        emoji_font = QFont()
        emoji_font.setPointSize(14)
        emoji_font.setFamily("Segoe UI Emoji")
        self.voice_button.setFont(emoji_font)
        self.voice_button.setStyleSheet(self.get_voice_button_style(False))
        input_layout.addWidget(self.voice_button)

        layout.addWidget(input_container)

        # Style
        self.setStyleSheet("""
            QWidget {
                background-color: #FFFFFF;
                border: 1px solid #CCCCCC;
                border-radius: 5px;
            }
            QLineEdit {
                border: 1px solid #CCCCCC;
                border-radius: 3px;
                padding: 5px;
            }

            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 0px;
                margin: 0px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)

        # Make window draggable
        self.oldPos = None

    def mousePressEvent(self, event):
        self.oldPos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        if self.oldPos:
            delta = event.globalPosition().toPoint() - self.oldPos
            self.move(self.pos() + delta)
            self.oldPos = event.globalPosition().toPoint()

    def get_voice_button_style(self, is_active):
        """Get voice button style based on active state"""
        color = "#F44336" if is_active else "#2196F3"
        return f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                border-radius: 3px;
                padding: 0px;
                margin: 0px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: {("#D32F2F" if is_active else "#1976D2")};
            }}
        """

    def update_voice_button_state(self, is_active):
        """Update voice button appearance based on active state"""
        self.voice_button.setStyleSheet(self.get_voice_button_style(is_active))

class WebAgentTab(QWidget):
    statusUpdate = Signal(str, bool, bool)
    messageReceived = Signal(str, bool)  # Signal for thread-safe message display

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)

        # Chat area
        self.chat_area = ChatArea()
        layout.addWidget(self.chat_area)

        # In-memory chat history for Web Agent (list of {role, content})
        self.chat_history = []


        # Input area
        input_container = QWidget()
        input_layout = QHBoxLayout(input_container)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type your web agent request...")
        self.input_field.setMinimumHeight(40)
        input_layout.addWidget(self.input_field)

        # Control buttons
        control_container = QWidget()
        control_layout = QHBoxLayout(control_container)
        control_layout.setContentsMargins(0, 0, 0, 0)

        self.send_stop_button = QPushButton("Send")
        self.send_stop_button.setMinimumHeight(40)
        self.pause_resume_button = QPushButton("Pause")
        self.pause_resume_button.setMinimumHeight(40)
        self.pause_resume_button.setEnabled(False)

        control_layout.addWidget(self.send_stop_button)
        control_layout.addWidget(self.pause_resume_button)
        input_layout.addWidget(control_container)
        # Sessions bar for Web Agent
        self.web_session_combo = QComboBox()
        self.web_new_btn = QPushButton("New Session")
        self.web_rename_btn = QPushButton("Rename")
        self.web_delete_btn = QPushButton("Delete")
        web_sess_bar = QWidget()
        web_sess_layout = QHBoxLayout(web_sess_bar)
        web_sess_layout.setContentsMargins(0, 0, 0, 0)
        web_sess_layout.addWidget(QLabel("Session:"))
        web_sess_layout.addWidget(self.web_session_combo)
        web_sess_layout.addStretch(1)
        web_sess_layout.addWidget(self.web_new_btn)
        web_sess_layout.addWidget(self.web_rename_btn)
        web_sess_layout.addWidget(self.web_delete_btn)
        layout.addWidget(web_sess_bar)

        # Web chat session state
        self.web_current_session_id = None
        self._web_restoring = False

        # Connect session controls
        self.web_session_combo.currentIndexChanged.connect(self._on_web_session_changed)
        self.web_new_btn.clicked.connect(self._on_web_new_session)
        self.web_rename_btn.clicked.connect(self._on_web_rename_session)
        self.web_delete_btn.clicked.connect(self._on_web_delete_session)

        # Initialize sessions per preference (always start new)
        self._init_web_sessions()


        # Voice button with improved styling
        self.voice_button = QPushButton("🎤")
        self.voice_button.setFixedSize(40, 40)
        emoji_font = QFont()
        emoji_font.setPointSize(16)
        emoji_font.setFamily("Segoe UI Emoji")
        self.voice_button.setFont(emoji_font)
        self.voice_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;

                border: none;
                border-radius: 5px;
                padding: 0px;
                margin: 0px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        input_layout.addWidget(self.voice_button)

        layout.addWidget(input_container)

        # Initialize state variables
        self.is_processing = False
        self.paused_execution = False
        self.voice_active = False
        self.paused_goal = None
        self.paused_actions = []
        self.additional_context = None

        # Voice handler for this tab
        self.voice_handler = VoiceHandler()
        self.voice_handler.textReceived.connect(self.handle_voice_text)

        # Connect the message signal to add_message method
        self.messageReceived.connect(self.add_message)

        # Browser settings
        options = QGroupBox("Options")
        options_layout = QVBoxLayout(options)

        # Headless mode option
        headless_option = QHBoxLayout()
        self.headless_checkbox = QCheckBox("Headless Mode")
        headless_option.addWidget(self.headless_checkbox)

        # Add vision mode option
        self.vision_checkbox = QCheckBox("Use Vision")
        self.vision_checkbox.setToolTip("Use visual analysis to help generate web actions")
        headless_option.addWidget(self.vision_checkbox)

        options_layout.addLayout(headless_option)

        # Add browser settings to layout
        layout.addWidget(options)

        # Volume controls container (similar to chat tab)
        volume_container = QWidget()
        volume_layout = QHBoxLayout(volume_container)

        # Settings button
        self.settings_button = QPushButton("Settings")
        self.settings_button.clicked.connect(self.open_settings_window)

        # Clear chat button
        self.clear_chat_button = QPushButton("Clear Chat")
        # Integrate with session clear for Web
        self.clear_chat_button.clicked.connect(lambda: self._on_web_clear_messages())

        # Mute button
        self.mute_button = QPushButton("🔊")
        self.mute_button.setFixedSize(40, 30)
        self.mute_button.clicked.connect(self.toggle_mute)

        # Volume slider
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(80)
        self.volume_slider.valueChanged.connect(self.update_volume)

        # Subtitles checkbox
        self.subtitles_checkbox = QCheckBox("Subtitles")
        self.subtitles_checkbox.setChecked(True)
        self.subtitles_checkbox.stateChanged.connect(self.toggle_subtitles)

        volume_layout.addWidget(self.settings_button)
        volume_layout.addWidget(self.clear_chat_button)
        volume_layout.addWidget(self.volume_slider)
        volume_layout.addWidget(self.mute_button)
        volume_layout.addWidget(self.subtitles_checkbox)

        layout.addWidget(volume_container)

        # Connect signals
        self.input_field.returnPressed.connect(self.send_message)
        self.send_stop_button.clicked.connect(self.handle_send_stop)
        self.pause_resume_button.clicked.connect(self.handle_pause_resume)
        self.voice_button.clicked.connect(self.toggle_voice_input)

        # Apply styling
        self.apply_styling()

        # Add voice shortcut
        self.voice_shortcut = QShortcut(QKeySequence("Alt+V"), self)
        self.voice_shortcut.activated.connect(self.toggle_voice_input)

    def apply_styling(self):
        """Apply modern styling to the web agent tab"""
        self.setStyleSheet("""
            QPushButton {
                padding: 8px 16px;
                border-radius: 4px;
                background-color: #2196F3;
                color: white;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QLineEdit {
                padding: 8px;
                border: 1px solid #CCCCCC;
                border-radius: 5px;
                font-size: 14px;
            }
            QGroupBox {
                border: 1px solid #CCCCCC;
                border-radius: 5px;
                padding: 10px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
        """)

    def add_message(self, message, is_user=True):
        """Add a message bubble to the chat area and store in chat history"""
        bubble = MessageBubble(message, is_user)
        self.chat_area.layout.addWidget(bubble)

        # Update in-memory chat history for context building
        role = "user" if is_user else "assistant"
        prev_len = len(getattr(self, 'chat_history', []) or [])
        try:
            self.chat_history.append({"role": role, "content": str(message)})
        except Exception:
            pass

        # Persist to current web session if applicable and not in restore mode
        try:
            if hasattr(self, 'web_current_session_id') and self.web_current_session_id and not getattr(self, '_web_restoring', False):
                chat_add_message(self.web_current_session_id, role, str(message))
                # If this is the first user message in a new session, auto-name it
                if is_user and prev_len == 0:
                    first_line = str(message).strip()
                    if first_line:
                        chat_rename_session(self.web_current_session_id, first_line[:60])
                        self._refresh_web_sessions(select_id=self.web_current_session_id)
        except Exception:
            pass

        # Auto scroll to bottom
        self.chat_area.verticalScrollBar().setValue(
            self.chat_area.verticalScrollBar().maximum()
        )

    def build_conversation_context(self, max_messages: int = 20, max_chars: int = 6000) -> str:
        """Build a compact conversation history string for Web Agent context."""
        try:
            history = getattr(self, "chat_history", []) or []
            recent = history[-max_messages:]
            lines = []
            for msg in recent:
                role = msg.get("role", "assistant")
                speaker = "User" if role == "user" else "Assistant"
                content = str(msg.get("content", ""))
                lines.append(f"{speaker}: {content}")
            text = "\n".join(lines)
            if len(text) > max_chars:
                text = text[-max_chars:]
            return f"Conversation history (most recent first at bottom):\n{text}"
        except Exception:
            return ""


    def handle_send_stop(self):
        """Handle send/stop button click based on current state"""
        if self.is_processing:
            self.stop_execution()
        else:
            self.send_message()

    def send_message(self):
        """Send message to the web assistant"""
        message = self.input_field.text().strip()
        if message:
            self.input_field.clear()

            # Detect language of input
            try:
                current_language = detect(message)
                if current_language not in ['en', 'vi']:
                    current_language = 'en'

                # Sync language with main window for correct speech/subtitles
                main_window = get_app_instance()
                if main_window:
                    main_window.current_language = current_language
            except:
                main_window = get_app_instance()
                if main_window:
                    main_window.current_language = 'en'

            # Add user message to chat using signal for thread safety
            self.messageReceived.emit(message, True)

            # Process message in a separate thread
            self.set_stop_mode()
            threading.Thread(target=self.process_message, args=(message,), daemon=True).start()

    def process_message(self, message, resumed=False):
        """Process message and execute web actions"""
        try:
            if not resumed:
                # Store goal for potential pause/resume
                self.paused_goal = message
                self.paused_actions = []

            headless = self.headless_checkbox.isChecked()
            use_vision = self.vision_checkbox.isChecked()

            # Build conversation context to include entire chat history
            conv_context = self.build_conversation_context(max_messages=20, max_chars=6000)
            addl = self.additional_context if self.additional_context else ""
            combined_context = conv_context if not addl else f"{conv_context}\n\nContext: {addl}"

            result = web_assistant(
                goal=message,
                executed_actions=self.paused_actions if resumed else None,
                headless=headless,
                use_vision=use_vision,
                additional_context=combined_context
            )

            if not self.paused_execution:
                # Use invokeMethod safely
                QMetaObject.invokeMethod(
                    self,
                    "complete_task",
                    Qt.ConnectionType.QueuedConnection,
                    Q_ARG(str, result or "")
                )

        except Exception as e:
            error_message = f"Error: {str(e)}"
            # Send error via signal for thread safety
            self.messageReceived.emit(error_message, False)

            # Set send mode in main thread
            QMetaObject.invokeMethod(
                self,
                "set_send_mode",

                Qt.ConnectionType.QueuedConnection
            )
        finally:
            if not self.paused_execution:
                clear_stop()

    def set_stop_mode(self):
        """Set button to stop mode"""
        self.send_stop_button.setText("Stop")
        self.send_stop_button.setStyleSheet("QPushButton { background-color: #F44336; }")
        self.pause_resume_button.setEnabled(True)
        self.input_field.setEnabled(False)
        self.voice_button.setEnabled(False)
        self.is_processing = True
        self.statusUpdate.emit("Processing web task...", True, False)

    def set_send_mode(self):
        """Set button to send mode"""
        self.send_stop_button.setText("Send")
        self.send_stop_button.setStyleSheet("QPushButton { background-color: #2196F3; }")
        self.pause_resume_button.setEnabled(False)
        self.pause_resume_button.setText("Pause")
        self.input_field.setEnabled(True)
        self.voice_button.setEnabled(True)
        self.is_processing = False
        self.paused_execution = False
        self.paused_goal = None
        self.paused_actions = None
        self.additional_context = None
        self.statusUpdate.emit("Ready", False, False)

    def stop_execution(self):
        """Stop the current task execution"""
        request_stop()
        self.set_send_mode()

    def handle_pause_resume(self):
        """Handle pause/resume button click"""
        if self.paused_execution:
            # Resume execution
            self.pause_resume_button.setText("Pause")
            self.paused_execution = False
            additional_info = self.input_field.text().strip()
            self.input_field.clear()
            if additional_info:
                self.additional_context = additional_info
            # Resume the task execution
            if self.paused_goal:
                threading.Thread(target=self.process_message, args=(self.paused_goal, True), daemon=True).start()
        else:
            # Pause execution
            self.pause_resume_button.setText("Resume")
            self.paused_execution = True
            request_stop()
            msg = "Task execution paused. Enter additional context if needed."
            self.add_message(msg, is_user=False)
            self.input_field.setEnabled(True)

    @Slot(str)
    def complete_task(self, result=None):
        """Complete the task and update UI"""
        self.set_send_mode()
        # Display the final result message
        if result:
            self.add_message(result, is_user=False)
            speaker(result)

    # --- Web chat session management ---
    def _init_web_sessions(self):
        sid = chat_create_session('web', "")
        self.web_current_session_id = sid
        self._refresh_web_sessions(select_id=sid)

    def _refresh_web_sessions(self, select_id=None):
        sessions = chat_list_sessions('web')
        self.web_session_combo.blockSignals(True)
        self.web_session_combo.clear()
        for sess in sessions:
            label_name = sess['name'] if sess['name'] else 'New chat'
            dt_str = _format_session_timestamp(sess.get('created_at'))
            label = f"{label_name} — {dt_str}" if dt_str else label_name
            self.web_session_combo.addItem(label, sess['id'])
        self.web_session_combo.blockSignals(False)
        if select_id is not None:
            for i in range(self.web_session_combo.count()):
                if self.web_session_combo.itemData(i) == select_id:
                    self.web_session_combo.setCurrentIndex(i)
                    break

    def _delete_web_session_if_empty(self, session_id: int) -> bool:
        try:
            if not session_id:
                return False
            msgs = chat_list_messages(session_id)
            if not msgs:
                chat_delete_session(session_id)
                return True
            return False
        except Exception:
            return False

    def _on_web_session_changed(self, index):
        if index < 0:
            return
        new_id = self.web_session_combo.itemData(index)
        if not new_id or new_id == self.web_current_session_id:
            return
        old_id = getattr(self, 'web_current_session_id', None)
        if old_id and old_id != new_id:
            if self._delete_web_session_if_empty(old_id):
                self._refresh_web_sessions(select_id=new_id)
        self.web_current_session_id = new_id
        self._load_web_session_messages(new_id)

    def _on_web_new_session(self):
        curr_id = getattr(self, 'web_current_session_id', None)
        if curr_id:
            self._delete_web_session_if_empty(curr_id)
        sid = chat_create_session('web', "")
        self.web_current_session_id = sid
        self._refresh_web_sessions(select_id=sid)
        self.clear_chat()

    def _on_web_rename_session(self):
        if not self.web_current_session_id:
            return
        current_name = ""
        try:
            for sess in chat_list_sessions('web'):
                if sess['id'] == self.web_current_session_id:
                    current_name = sess.get('name') or ""
                    break
        except Exception:
            pass
        new_name, ok = QInputDialog.getText(
            self, "Rename Session", "New name:", QLineEdit.Normal, current_name
        )
        if ok and new_name.strip():
            chat_rename_session(self.web_current_session_id, new_name.strip())
            self._refresh_web_sessions(select_id=self.web_current_session_id)

    def _on_web_delete_session(self):
        if not self.web_current_session_id:
            return
        confirm = QMessageBox.question(self, "Delete Session", "Delete this session and all messages?", QMessageBox.Yes|QMessageBox.No)
        if confirm == QMessageBox.Yes:
            chat_delete_session(self.web_current_session_id)
            sid = chat_create_session('web', "")
            self.web_current_session_id = sid
            self._refresh_web_sessions(select_id=sid)
            self.clear_chat()

    def _on_web_clear_messages(self):
        if not self.web_current_session_id:
            return
        confirm = QMessageBox.question(self, "Clear Messages", "Clear all messages in this session?", QMessageBox.Yes|QMessageBox.No)
        if confirm == QMessageBox.Yes:
            chat_clear_session_messages(self.web_current_session_id)
            self.clear_chat()

    def _load_web_session_messages(self, session_id: int):
        self.clear_chat()
        self._web_restoring = True
        try:
            msgs = chat_list_messages(session_id)
            for m in msgs:
                is_user = (m['role'] == 'user')
                self.add_message(m['content'], is_user)
        finally:
            self._web_restoring = False

    def toggle_voice_input(self):
        """Toggle voice input for web agent"""
        if not self.voice_active:
            self.voice_active = True
            self.voice_button.setStyleSheet("""
                QPushButton {
                    background-color: #F44336;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    padding: 0px;
                    margin: 0px;

                    font-size: 16px;
                }
                QPushButton:hover {
                    background-color: #D32F2F;
                }
            """)
            # Start listening using the existing voice_handler
            threading.Thread(target=self.listen_voice, daemon=True).start()
        else:
            self.voice_active = False
            self.voice_button.setStyleSheet("""
                QPushButton {
                    background-color: #2196F3;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    padding: 0px;
                    margin: 0px;
                    font-size: 16px;
                }
                QPushButton:hover {
                    background-color: #1976D2;
                }
            """)

    def listen_voice(self):
        """Listen for voice input and convert to text"""
        with sr.Microphone() as source:
            try:
                # Don't use global speaker, use signal to add message
                self.messageReceived.emit("I'm listening...", False)

                recognizer = sr.Recognizer()
                audio = recognizer.listen(source, timeout=5)
                try:
                    text = recognizer.recognize_google(audio, language='vi-VN')
                    current_language = 'vi'
                    main_window = get_app_instance()
                    if main_window:
                        main_window.current_language = current_language
                except:
                    text = recognizer.recognize_google(audio, language='en-US')
                    current_language = 'en'
                    main_window = get_app_instance()
                    if main_window:
                        main_window.current_language = current_language

                # Handle recognized text in main thread
                if text:
                    QMetaObject.invokeMethod(
                        self,
                        "handle_voice_text",
                        Qt.ConnectionType.QueuedConnection,
                        Q_ARG(str, text)
                    )
            except Exception as e:
                self.messageReceived.emit(f"Voice recognition error: {str(e)}", False)
            finally:
                self.voice_active = False

                # Update button style in main thread
                QMetaObject.invokeMethod(
                    self,
                    "reset_voice_button",
                    Qt.ConnectionType.QueuedConnection
                )

    @Slot()
    def reset_voice_button(self):
        """Reset voice button style - thread safe method"""
        self.voice_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 0px;
                margin: 0px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)

    def clear_chat(self):
        """Clear all messages from the chat area and reset conversation history."""
        # Remove all message bubbles
        while self.chat_area.layout.count():
            child = self.chat_area.layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Also clear in-memory conversation history used for context
        try:
            if hasattr(self, "chat_history") and isinstance(self.chat_history, list):
                self.chat_history.clear()
            else:
                self.chat_history = []
        except Exception:
            self.chat_history = []

    def handle_voice_text(self, text):
        """Handle text received from voice recognition"""
        if text:
            self.input_field.setText(text)
            self.send_message()

    def update_volume(self, value):
        """Update volume via the global function"""
        # Get main window to check mute state
        main_window = get_app_instance()
        if main_window and not main_window.is_muted:
            set_volume(value)

    def toggle_mute(self):
        """Toggle mute state"""
        main_window = get_app_instance()
        if main_window:
            main_window.is_muted = not main_window.is_muted
            if main_window.is_muted:
                self.mute_button.setText("🔇")
                set_volume(0)
            else:
                self.mute_button.setText("🔊")
                set_volume(self.volume_slider.value())

    def toggle_subtitles(self):
        """Toggle subtitles via the global function"""
        set_subtitles(self.subtitles_checkbox.isChecked())

    def open_settings_window(self):
        """Open settings dialog from parent window"""
        # Get the main window (parent of our tab)
        main_window = self.parent()
        while main_window and not isinstance(main_window, ModernChatInterface):
            main_window = main_window.parent()

        # If we found the main window, call its settings method
        if main_window and hasattr(main_window, 'open_settings_window'):
            main_window.open_settings_window()

class RPATab(QWidget):
    statusUpdate = Signal(str, bool, bool)

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)

        # Create Tasks group
        tasks_group = QGroupBox("Tasks")
        tasks_layout = QVBoxLayout(tasks_group)

        # Task list with improved visibility
        self.task_list = QListWidget()
        # Add double click handler for tasks
        self.task_list.itemDoubleClicked.connect(self.edit_selected_task)
        self.task_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #CCCCCC;
                border-radius: 4px;
                padding: 5px;
            }
            QListWidget::item {
                border-bottom: 1px solid #EEEEEE;
                padding: 10px 5px;
                margin: 2px 0;
            }
            QListWidget::item:selected {
                background-color: #2196F3;
                color: white;
                border-radius: 4px;
                outline: none;
            }
            QListWidget::item:hover {
                background-color: #E3F2FD;
                border-radius: 4px;
            }
        """)
        tasks_layout.addWidget(self.task_list)

        # Task buttons
        task_buttons = QHBoxLayout()
        self.add_task = QPushButton("New Task")
        self.edit_task = QPushButton("Edit Task")
        self.delete_task = QPushButton("Delete Task")
        task_buttons.addWidget(self.add_task)
        task_buttons.addWidget(self.edit_task)
        task_buttons.addWidget(self.delete_task)
        tasks_layout.addLayout(task_buttons)

        # Add tasks group to main layout
        layout.addWidget(tasks_group)

        # Task execution settings
        settings_frame = QFrame()
        settings_frame.setObjectName("settingsFrame")
        settings_layout = QVBoxLayout(settings_frame)

        # Execution settings row
        exec_settings = QHBoxLayout()
        exec_settings.setSpacing(20)

        # Repeat settings
        repeat_container = QWidget()
        repeat_layout = QVBoxLayout(repeat_container)
        repeat_layout.setContentsMargins(0, 0, 0, 0)

        repeat_header = QHBoxLayout()
        repeat_header.addWidget(QLabel("Task Repeats:"))
        self.repeat_count = QSpinBox()
        self.repeat_count.setRange(1, 1000)
        self.repeat_count.setValue(1)
        self.repeat_count.setToolTip("Number of times to repeat the entire task")
        self.repeat_count.setMinimumWidth(100)
        repeat_header.addWidget(self.repeat_count)
        repeat_layout.addLayout(repeat_header)

        # Add repeat delay setting
        repeat_delay = QHBoxLayout()
        repeat_delay.addWidget(QLabel("Delay Between Repeats (seconds):"))
        self.repeat_delay = QDoubleSpinBox()
        self.repeat_delay.setRange(0, 3600)
        self.repeat_delay.setValue(0.0)
        self.repeat_delay.setSingleStep(1.0)
        self.repeat_delay.setToolTip("Delay between each repeat of the task")
        repeat_delay.addWidget(self.repeat_delay)
        repeat_layout.addLayout(repeat_delay)

        exec_settings.addWidget(repeat_container, 1)

        # Delay settings
        delay_container = QWidget()
        delay_layout = QHBoxLayout(delay_container)
        delay_layout.setContentsMargins(0, 0, 0, 0)
        delay_layout.addWidget(QLabel("Delay Between Actions (seconds):"))
        self.action_delay = QDoubleSpinBox()
        self.action_delay.setRange(0.1, 60.0)
        self.action_delay.setValue(0.1)
        self.action_delay.setSingleStep(0.1)
        self.action_delay.setToolTip("Delay between each action in the task")
        self.action_delay.setMinimumWidth(100)
        delay_layout.addWidget(self.action_delay)
        exec_settings.addWidget(delay_container, 1)

        settings_layout.addLayout(exec_settings)

        # Add scheduling section
        schedule_group = QGroupBox("Task Scheduling")
        schedule_layout = QVBoxLayout(schedule_group)

        # Schedule list
        self.schedule_list = QListWidget()
        self.schedule_list.itemDoubleClicked.connect(self.edit_schedule)
        schedule_layout.addWidget(self.schedule_list)

        # Schedule buttons
        schedule_buttons = QHBoxLayout()
        self.add_schedule = QPushButton("Add Schedule")
        self.edit_schedule_btn = QPushButton("Edit Schedule")
        self.delete_schedule_btn = QPushButton("Delete Schedule")

        self.add_schedule.clicked.connect(self.add_new_schedule)
        self.edit_schedule_btn.clicked.connect(self.edit_schedule)
        self.delete_schedule_btn.clicked.connect(self.delete_selected_schedule)

        schedule_buttons.addWidget(self.add_schedule)
        schedule_buttons.addWidget(self.edit_schedule_btn)
        schedule_buttons.addWidget(self.delete_schedule_btn)
        schedule_layout.addLayout(schedule_buttons)

        layout.addWidget(schedule_group)

        # Initialize scheduler
        self.scheduled_jobs = {}
        self.scheduler_thread = None
        self.start_scheduler()

        # Load existing schedules
        self.load_schedules()

        # Schedule status
        self.schedule_status = QLabel()
        layout.addWidget(self.schedule_status)

        # Initialize scheduler
        self.scheduled_jobs = {}
        self.scheduler_thread = None
        self.start_scheduler()

        layout.addWidget(settings_frame)

        # Execution controls - modify this section
        exec_controls = QHBoxLayout()
        self.play_pause_button = QPushButton("▶")
        self.stop_button = QPushButton("◼")
        exec_controls.addWidget(self.play_pause_button)
        exec_controls.addWidget(self.stop_button)
        layout.addLayout(exec_controls)

        # Add execution status label
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        # Initialize execution state
        self.current_task = None
        self.is_paused = False
        self.is_executing = False

        # Connect signals - modify this section
        self.add_task.clicked.connect(self.create_task)
        self.edit_task.clicked.connect(self.edit_selected_task)
        self.delete_task.clicked.connect(self.delete_selected_task)
        self.play_pause_button.clicked.connect(self.handle_play_pause)
        self.stop_button.clicked.connect(self.stop_execution)

        # Disable stop button initially
        self.stop_button.setEnabled(False)

        # Load existing tasks
        self.load_tasks()

        # Apply styling
        self.apply_styling()

        # Add styling for settings frame with improved spinner heights
        settings_frame.setStyleSheet("""
            QFrame#settingsFrame {
                background-color: #F5F5F5;
                border-radius: 8px;
                padding: 10px;
                margin: 5px;
            }
            QLabel {
                color: #424242;
                font-weight: bold;
            }
            QSpinBox, QDoubleSpinBox {
                padding: 8px;
                border: 1px solid #CCCCCC;
                border-radius: 4px;
                min-width: 80px;
                min-height: 20px;
                font-size: 12px;
            }
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                min-height: 10px;
            }
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                min-height: 10px;
            }
        """)

        # Add task selection change handler
        self.task_list.itemSelectionChanged.connect(self.on_task_selection_changed)

    def on_task_selection_changed(self):
        """Handle task selection change"""
        try:
            if hasattr(self.parent(), 'parent'):
                parent = self.parent()
                if hasattr(parent, 'update_mini_control'):
                    parent.update_mini_control()
        except Exception as e:
            print(f"Error updating mini control: {str(e)}")

    def apply_styling(self):
        # Add modern styling for the RPA tab
        self.setStyleSheet("""
            QPushButton {
                padding: 8px 16px;
                border-radius: 4px;
                background-color: #2196F3;
                color: white;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QListWidget {
                border: 1px solid #CCCCCC;
                border-radius: 4px;
                padding: 5px;
            }
        """)

    def load_tasks(self):
        """Load tasks into list widget"""
        self.task_list.clear()
        tasks = load_tasks()
        for task_name in tasks:
            item = QListWidgetItem(task_name)
            item.setSizeHint(item.sizeHint() + QSize(0, 20))
            self.task_list.addItem(item)

    def create_task(self):
        dialog = TaskEditor(parent=self)
        if dialog.exec() == QDialog.Accepted:
            self.load_tasks()

    def edit_selected_task(self):
        current = self.task_list.currentItem()
        if current:
            dialog = TaskEditor(current.text(), self)
            if dialog.exec() == QDialog.Accepted:
                self.load_tasks()

    def delete_selected_task(self):
        current = self.task_list.currentItem()
        if current:
            reply = QMessageBox.question(
                self, 'Delete Task',
                f'Are you sure you want to delete "{current.text()}"?',
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                if delete_task(current.text()):
                    self.load_tasks()

    def handle_play_pause(self):
        """Handle play/pause button click based on current state"""
        if self.is_executing:
            if self.is_paused:
                # Resume execution
                self.is_paused = False
                self.play_pause_button.setText("⏸")
            else:
                # Pause execution
                self.is_paused = True
                self.play_pause_button.setText("▶")
                self.status_label.setText(f"Paused: {self.current_task}")
        else:
            # Start execution
            self.execute_selected_task()

    def execute_selected_task(self):
        """Execute the selected task"""
        current = self.task_list.currentItem()
        if current and not self.is_executing:
            self.current_task = current.text()
            actions = get_task(self.current_task)
            if actions:
                self.is_executing = True
                self.is_paused = False
                self.status_label.setText(f"Executing: {self.current_task}")

                # Update button states
                self.play_pause_button.setText("⏸")
                self.stop_button.setEnabled(True)

                # Start execution thread
                threading.Thread(target=self.execute_actions, args=(self.current_task,)).start()

    def stop_execution(self):
        """Stop task execution"""
        if self.is_executing:
            self.is_executing = False
            self.is_paused = False
            self.status_label.setText("Stopped")
            self.play_pause_button.setText("▶")
            self.stop_button.setEnabled(False)

    def execute_actions(self, task_name):
        """Execute actions for the selected task"""
        try:
            # Get task data
            task_data = get_task(task_name)

            # Get actions from task data
            if isinstance(task_data, dict):
                actions = task_data.get('actions', [])
            else:
                # Handle legacy format
                actions = task_data

            if not actions:
                self.update_status(f"No actions found for task: {task_name}")
                return

            # Start execution thread
            self.execution_thread = threading.Thread(
                target=self.execute_action_sequence,
                args=(actions,),
                daemon=True
            )
            self.execution_thread.start()

        except Exception as e:
            self.update_status(f"Error starting execution: {str(e)}")
            self.stop_execution()

    def execute_action_sequence(self, actions):
        """Execute a sequence of actions"""
        try:
            self.is_executing = True
            self.update_execution_state()

            # Get execution settings
            total_repeats = self.repeat_count.value()
            action_delay = self.action_delay.value()
            repeat_delay = self.repeat_delay.value()

            for repeat in range(total_repeats):
                if not self.is_executing:
                    break

                # Update status with repeat count if multiple repeats
                if total_repeats > 1:
                    self.status_label.setText(f"Executing: {self.current_task} (Repeat {repeat + 1}/{total_repeats})")

                # Execute each action in the task
                for i, action in enumerate(actions):
                    if not self.is_executing:
                        break

                    if self.is_paused:
                        while self.is_paused and self.is_executing:
                            time.sleep(0.1)
                        if not self.is_executing:
                            break

                    action_status = f"Action {i+1}/{len(actions)}"
                    if total_repeats > 1:
                        action_status = f"Repeat {repeat + 1}/{total_repeats}, {action_status}"

                    # Emit signal for status update
                    self.statusUpdate.emit(f"Executing: {self.current_task} ({action_status})", self.is_executing, self.is_paused)

                    # Execute single action
                    success = execute_action(json.dumps({"action": [action]}))
                    if not success:
                        self.update_status("Action execution failed!")
                        self.stop_execution()
                        return

                    time.sleep(action_delay)

                # Add delay between task repeats if needed
                if repeat < total_repeats - 1 and self.is_executing:
                    time.sleep(repeat_delay)

            if self.is_executing:
                completed_msg = f"Completed: {self.current_task}"
                if total_repeats > 1:
                    completed_msg += f" ({total_repeats} times)"
                self.update_status(completed_msg)
                self.stop_execution()

        except Exception as e:
            self.update_status(f"Error during execution: {str(e)}")
            self.stop_execution()

    def update_execution_state(self):
        """Update the execution state of the task"""
        self.status_label.setText(f"Executing: {self.current_task}")
        self.play_pause_button.setText(f"{'⏸' if self.is_paused else '▶'}")
        self.stop_button.setEnabled(self.is_executing)

    def on_task_selection_changed(self):
        """Handle task selection change"""
        if hasattr(self.parent(), 'update_mini_control'):
            self.parent().parent().update_mini_control()

    def on_schedule_type_changed(self, schedule_type):
        """Handle schedule type changes"""
        type_to_index = {
            "Daily": 0,
            "Weekly": 1,
            "Monthly": 2,
            "Yearly": 3,
            "None": 4
        }
        self.schedule_stack.setCurrentIndex(type_to_index.get(schedule_type, 4))

    def get_schedule_settings(self):
        """Get current schedule settings"""
        schedule_type = self.schedule_combo.currentText()
        if schedule_type == "None":
            return None

        settings = {"type": schedule_type}

        if schedule_type == "Daily":
            settings["time"] = self.daily_time.time()
        elif schedule_type == "Weekly":
            settings["day"] = self.weekday_combo.currentText()
            settings["time"] = self.weekly_time.time()
        elif schedule_type == "Monthly":
            settings["day"] = self.monthly_day.value()
            settings["time"] = self.monthly_time.time()
        elif schedule_type == "Yearly":
            settings["date"] = self.yearly_date.date()
            settings["time"] = self.yearly_time.time()

        return settings

    def schedule_task(self):
        """Schedule current task with selected settings"""
        current = self.task_list.currentItem()
        if not current:
            return

        settings = self.get_schedule_settings()
        if not settings:
            return

        task_name = current.text()

        # Cancel existing schedule for this task
        if task_name in self.scheduled_jobs:
            schedule.cancel_job(self.scheduled_jobs[task_name])

        # Create new schedule
        job = None
        if settings["type"] == "Daily":
            time_str = settings["time"].toString("HH:mm")
            job = schedule.every().day.at(time_str).do(self.execute_selected_task)
        elif settings["type"] == "Weekly":
            time_str = settings["time"].toString("HH:mm")
            day = settings["day"].lower()
            job = getattr(schedule.every(), day).at(time_str).do(self.execute_selected_task)
        elif settings["type"] == "Monthly":
            time_str = settings["time"].toString("HH:mm")
            job = schedule.every().month.at(f"{settings['day']}:{time_str}").do(self.execute_selected_task)
        elif settings["type"] == "Yearly":
            date = settings["date"]
            time = settings["time"]
            job = schedule.every().year.at(f"{date.month()}-{date.day()} {time.toString('HH:mm')}").do(self.execute_selected_task)

        if job:
            self.scheduled_jobs[task_name] = job
            self.update_schedule_status()

    def start_scheduler(self):
        """Start the scheduler thread"""
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(1)

        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()

    def update_schedule_status(self):
        """Update the schedule status display"""
        current = self.task_list.currentItem()
        if not current or current.text() not in self.scheduled_jobs:
            self.schedule_status.setText("No schedule set")
            return

        job = self.scheduled_jobs[current.text()]
        next_run = job.next_run
        self.schedule_status.setText(f"Next run: {next_run.strftime('%Y-%m-%d %H:%M')}")

    def add_new_schedule(self):
        """Add a new schedule for the selected task"""
        current = self.task_list.currentItem()
        if not current:
            QMessageBox.warning(self, "Error", "Please select a task to schedule")
            return

        dialog = ScheduleEditor(current.text())
        if dialog.exec() == QDialog.Accepted:
            settings = dialog.get_schedule_settings()
            self.create_schedule(current.text(), settings)
            self.load_schedules()

    def edit_schedule(self, item=None):
        """Edit existing schedule"""
        if not item:
            item = self.schedule_list.currentItem()
        if not item:
            return

        data = item.data(Qt.UserRole)
        task_name = data["task"]
        settings = data["settings"]
        schedule_id = settings["id"]

        dialog = ScheduleEditor(task_name, settings)
        if dialog.exec() == QDialog.Accepted:
            new_settings = dialog.get_schedule_settings()
            # Pass schedule_id to update_schedule
            self.update_schedule(task_name, schedule_id, new_settings)
            self.load_schedules()

    def delete_selected_schedule(self):
        """Delete selected schedule"""
        item = self.schedule_list.currentItem()
        if not item:
            return

        data = item.data(Qt.UserRole)
        task_name = data["task"]
        schedule_id = data["settings"]["id"]

        reply = QMessageBox.question(
            self, 'Delete Schedule',
            f'Are you sure you want to delete this schedule for "{task_name}"?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.delete_schedule(task_name, schedule_id)
            # Remove the item from the list widget directly
            row = self.schedule_list.row(item)
            self.schedule_list.takeItem(row)

    def delete_schedule(self, task_name, schedule_id):
        """Delete a specific schedule"""
        if task_name in self.scheduled_jobs and schedule_id in self.scheduled_jobs[task_name]:
            # Cancel the scheduled job
            schedule.cancel_job(self.scheduled_jobs[task_name][schedule_id]["job"])
            # Remove from scheduled jobs dictionary
            del self.scheduled_jobs[task_name][schedule_id]

            # Remove task entry if no more schedules
            if not self.scheduled_jobs[task_name]:
                del self.scheduled_jobs[task_name]

            # Save updated schedules to file
            self.save_schedules()

    def create_schedule(self, task_name, settings):
        """Create a new schedule"""

        # Generate unique ID for new schedule if not exists
        if "id" not in settings:
            settings["id"] = str(uuid.uuid4())

        schedule_id = settings["id"]

        # Get time string for scheduler
        if isinstance(settings["time"], QTime):
            time_str = settings["time"].toString("HH:mm")
        else:
            time_str = settings["time"]

        # Create job based on schedule type
        job = None
        if settings["type"] == "Daily":
            job = schedule.every().day.at(time_str).do(
                self.execute_task_by_name, task_name
            )
        elif settings["type"] == "Weekly":
            day = settings["day"].lower()
            job = getattr(schedule.every(), day).at(time_str).do(
                self.execute_task_by_name, task_name
            )
        elif settings["type"] == "Monthly":
            def monthly_check():
                now = datetime.now()
                if now.day == settings['day']:
                    self.execute_task_by_name(task_name)
            job = schedule.every().day.at(time_str).do(monthly_check)
        elif settings["type"] == "Yearly":
            def yearly_check():
                now = datetime.now()
                date_parts = settings['date'].split('-')
                if len(date_parts) == 2 and now.month == int(date_parts[0]) and now.day == int(date_parts[1]):
                    self.execute_task_by_name(task_name)
            job = schedule.every().day.at(time_str).do(yearly_check)

        if job:
            # Initialize dict for task if not exists
            if task_name not in self.scheduled_jobs:
                self.scheduled_jobs[task_name] = {}

            # Store schedule info with unique ID
            self.scheduled_jobs[task_name][schedule_id] = {
                "job": job,
                "settings": settings
            }
            self.save_schedules()

    def update_schedule(self, task_name, schedule_id, new_settings):
        """Update existing schedule"""
        # Delete old schedule
        if task_name in self.scheduled_jobs and schedule_id in self.scheduled_jobs[task_name]:
            schedule.cancel_job(self.scheduled_jobs[task_name][schedule_id]["job"])

        # Create new schedule with same ID
        new_settings["id"] = schedule_id
        self.create_schedule(task_name, new_settings)

    def delete_schedule(self, task_name, schedule_id):
        """Delete a specific schedule"""
        if task_name in self.scheduled_jobs and schedule_id in self.scheduled_jobs[task_name]:
            # Cancel the scheduled job
            schedule.cancel_job(self.scheduled_jobs[task_name][schedule_id]["job"])
            # Remove from scheduled jobs dictionary
            del self.scheduled_jobs[task_name][schedule_id]

            # Remove task entry if no more schedules
            if not self.scheduled_jobs[task_name]:
                del self.scheduled_jobs[task_name]

            # Save updated schedules to file
            self.save_schedules

    def create_job(self, task_name, settings):
        """Create a schedule job"""
        job = None
        if settings["type"] == "Daily":
            time_str = settings["time"].toString("HH:mm")
            job = schedule.every().day.at(time_str).do(
                self.execute_task_by_name, task_name
            )
        elif settings["type"] == "Weekly":
            time_str = settings["time"].toString("HH:mm")
            day = settings["day"].lower()
            job = getattr(schedule.every(), day).at(time_str).do(
                self.execute_task_by_name, task_name
            )
        elif settings["type"] == "Monthly":
            time_str = settings["time"].toString("HH:mm")
            job = schedule.every().month.at(f"{settings['day']}:{time_str}").do(
                self.execute_task_by_name, task_name
            )
        elif settings["type"] == "Yearly":
            date = settings["date"]
            time = settings["time"]
            job = schedule.every().year.at(f"{date.month()}-{date.day()} {time.toString('HH:mm')}").do(
                self.execute_task_by_name, task_name
            )
        return job

    def execute_task_by_name(self, task_name):
        """Execute a task by name (used by scheduler)"""
        for i in range(self.task_list.count()):
            item = self.task_list.item(i)
            if item.text() == task_name:
                self.task_list.setCurrentItem(item)
                self.execute_selected_task()
                break

    def save_schedules(self):
        """Save schedules to file"""
        data = {}
        for task_name, schedules in self.scheduled_jobs.items():
            data[task_name] = [schedule_info["settings"] for schedule_info in schedules.values()]

        try:
            with open("task_schedules.json", "w") as f:
                json.dump(data, f, default=str)
        except Exception as e:
            print(f"Error saving schedules: {e}")

    def load_schedules(self):
        """Load schedules from file and update list"""
        try:
            with open("task_schedules.json", "r") as f:
                data = json.load(f)

            if not data:  # If file is empty or contains empty object
                return

            self.schedule_list.clear()
            for task_name, schedules in data.items():
                for schedule_settings in schedules:
                    self.create_schedule(task_name, schedule_settings)
                    self.add_schedule_to_list(task_name, schedule_settings)

        except FileNotFoundError:
            # Create empty file if it doesn't exist
            with open("task_schedules.json", "w") as f:
                json.dump({}, f)
        except json.JSONDecodeError:
            # Reset file if it's corrupted
            with open("task_schedules.json", "w") as f:
                json.dump({}, f)
        except Exception as e:
            print(f"Error loading schedules: {e}")

    def add_schedule_to_list(self, task_name, settings):
        """Add schedule to the list widget"""
        schedule_text = self.format_schedule_text(task_name, settings)
        item = QListWidgetItem(schedule_text)
        # Store settings with unique ID
        item.setData(Qt.UserRole, {
            "task": task_name,
            "settings": settings
        })
        self.schedule_list.addItem(item)

    def format_schedule_text(self, task_name, settings):
        """Format schedule display text"""
        schedule_type = settings["type"]
        if schedule_type == "Daily":
            return f"{task_name} - Daily at {settings['time']}"
        elif schedule_type == "Weekly":
            return f"{task_name} - Every {settings['day']} at {settings['time']}"
        elif schedule_type == "Monthly":
            return f"{task_name} - Monthly on day {settings['day']} at {settings['time']}"
        elif schedule_type == "Yearly":
            return f"{task_name} - Yearly on {settings['date']} at {settings['time']}"
        return task_name

    def update_status(self, message):
        """Update status label and emit status signal"""
        self.status_label.setText(message)
        self.statusUpdate.emit(message, self.is_executing, self.is_paused)

class TaskRecorder(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Task Recorder")
        self.setMinimumSize(400, 300)

        layout = QVBoxLayout(self)

        # Recording status
        self.status_label = QLabel("Not Recording")
        layout.addWidget(self.status_label)

        # Control buttons
        controls = QHBoxLayout()
        self.record_button = QPushButton("Start Recording")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        controls.addWidget(self.record_button)
        controls.addWidget(self.stop_button)
        layout.addLayout(controls)

        # Recorded actions list
        self.actions_list = QListWidget()
        layout.addWidget(self.actions_list)

        # Add buttons to manage recorded actions
        action_buttons = QHBoxLayout()

        self.remove_action = QPushButton("Remove Action")
        self.remove_action.clicked.connect(self.remove_selected_action)
        self.remove_action.setEnabled(False)

        self.clear_actions = QPushButton("Clear All")
        self.clear_actions.clicked.connect(self.clear_all_actions)
        self.clear_actions.setEnabled(False)

        action_buttons.addWidget(self.remove_action)
        action_buttons.addWidget(self.clear_actions)
        layout.addLayout(action_buttons)

        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        # Connect signals
        self.record_button.clicked.connect(self.toggle_recording)
        self.stop_button.clicked.connect(self.stop_recording)
        self.actions_list.itemSelectionChanged.connect(self.on_selection_changed)

        self.recording = False
        self.recorded_actions = []
        self.listener_thread = None

        # Add recorded events queue
        self.events_queue = queue.Queue()
        self.keyboard_events = []
        self.last_time = time.time()
        self.mouse_listener = None
        self.keyboard_listener = None
        self.drag_start = None
        self.recording_drag = False
        self.drag_threshold = 10
        self.drag_start_time = None
        self.current_text = ""
        self.last_text_time = time.time()
        self.text_input_delay = 0.5  # Delay to batch text input
        self.current_text_action = None  # Track the current text action
        self.last_move_time = time.time()
        self.move_threshold = 0.5  # Seconds between move actions

    def on_selection_changed(self):
        """Enable/disable action buttons based on selection"""
        has_selection = len(self.actions_list.selectedItems()) > 0
        self.remove_action.setEnabled(has_selection)
        self.clear_actions.setEnabled(self.actions_list.count() > 0)

    def remove_selected_action(self):
        """Remove selected action from list"""
        current = self.actions_list.currentRow()
        if current >= 0:
            self.actions_list.takeItem(current)
            if current < len(self.recorded_actions):
                del self.recorded_actions[current]
        self.on_selection_changed()

    def clear_all_actions(self):
        """Clear all recorded actions"""
        reply = QMessageBox.question(
            self, 'Clear Actions',
            'Are you sure you want to clear all recorded actions?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.actions_list.clear()
            self.recorded_actions.clear()
            self.on_selection_changed()

    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.recording = True
        self.record_button.setText("Pause")
        self.stop_button.setEnabled(True)
        self.status_label.setText("Recording...")
        self.status_label.setStyleSheet("color: red")

        # Clear previous recordings
        self.recorded_actions = []
        self.events_queue = queue.Queue()
        self.keyboard_events = []
        self.last_time = time.time()

        # Start mouse listener
        self.mouse_listener = mouse.Listener(
            on_click=self.on_click,
            on_scroll=self.on_scroll,
            on_move=self.on_move
        )
        self.mouse_listener.start()

        # Start keyboard listener
        self.keyboard_listener = keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release
        )
        self.keyboard_listener.start()

        # Start processing thread
        self.listener_thread = threading.Thread(target=self.process_events)
        self.listener_thread.start()

    def stop_recording(self):
        """Stop recording and handle any pending text"""
        if self.recording:
            # Flush any pending text input
            if self.current_text:
                self.finalize_text_entry()

            # ...rest of existing stop_recording code...
            self.recording = False
            self.record_button.setText("Start Recording")
            self.stop_button.setEnabled(False)
            self.status_label.setText("Recording stopped")
            self.status_label.setStyleSheet("")

            # Stop listeners
            if self.mouse_listener:
                self.mouse_listener.stop()
            if self.keyboard_listener:
                self.keyboard_listener.stop()

            # Add any remaining keyboard combination
            self.add_keyboard_combination()

            self.on_selection_changed()

    def on_click(self, x, y, button, pressed):
        """Handle mouse clicks and drags"""
        if not self.recording:
            return

        # Add any pending keyboard combination before mouse action
        self.add_keyboard_combination()

        if pressed:
            if button == mouse.Button.left:
                self.drag_start = (x, y)
                self.drag_start_time = time.time()
                self.recording_drag = False
                return
        else:  # Button released
            if button == mouse.Button.left:
                if self.drag_start and self.recording_drag:
                    # Only create drag action if we've moved past threshold
                    dist_x = abs(x - self.drag_start[0])
                    dist_y = abs(y - self.drag_start[1])
                    if dist_x > self.drag_threshold or dist_y > self.drag_threshold:
                        # Create drag action
                        action = {
                            "act": "drag",
                            "detail": f"Drag from ({self.drag_start[0]}, {self.drag_start[1]}) to ({x}, {y})",
                            "coordinates": f"x={self.drag_start[0]}, y={self.drag_start[1]} to x={x}, y={y}",
                            "repeat": 1
                        }
                        self.events_queue.put(action)
                        self.drag_start = None
                        self.recording_drag = False
                        return

                # If we get here, it was a click, not a drag
                action = {
                    "act": "click",
                    "detail": f"Click at ({x}, {y})",
                    "coordinates": f"x={x}, y={y}",
                    "repeat": 1
                }
                self.events_queue.put(action)
                self.drag_start = None
                self.recording_drag = False
            elif button == mouse.Button.right:
                action = {
                    "act": "right_click",
                    "detail": f"Right click at ({x}, {y})",
                    "coordinates": f"x={x}, y={y}",
                    "repeat": 1
                }
                self.events_queue.put(action)

    def on_scroll(self, x, y, dx, dy):
        """Handle mouse scroll"""
        if not self.recording:
            return

        # Add scroll action
        direction = "up" if dy > 0 else "down"
        action = {
            "act": "scroll",
            "detail": direction,
            "repeat": 1
        }
        self.events_queue.put(action)

    def on_key_press(self, key):
        """Handle key press with immediate text updates"""
        if not self.recording:
            return

        try:
            # Handle modifier keys first
            if hasattr(key, 'name'):
                key_str = str(key).replace('Key.', '')

                # Handle modifier keys
                if key_str.startswith(('ctrl', 'alt', 'shift', 'cmd')):
                    modifier_map = {
                        'ctrl': 'Ctrl',
                        'alt': 'Alt',
                        'shift': 'Shift',
                        'cmd': 'Windows'
                    }
                    for mod in modifier_map:
                        if key_str.startswith(mod):
                            key_str = modifier_map[mod]
                            break

                    if key_str not in self.keyboard_events:
                        self.keyboard_events.append(key_str)
                # For other special keys
                elif key_str not in self.keyboard_events:
                    self.keyboard_events.append(key_str)

            # Handle regular character keys
            else:
                try:
                    # Handle control characters
                    if hasattr(key, 'char'):
                        char = key.char
                        if isinstance(char, str) and len(char) == 1:
                            # Convert control character to regular character
                            if ord(char) < 32:
                                char = chr(ord(char) + 64)
                            if char not in self.keyboard_events:
                                self.keyboard_events.append(char)
                except AttributeError:
                    pass

        except Exception as e:
            print(f"Error in key press: {str(e)}")

    def update_text_entry(self):
        """Update or create text entry action in real-time"""
        if not self.current_text:
            return

        action = {
            "act": "text_entry",
            "detail": self.current_text,
            "repeat": 1
        }

        if self.current_text_action is None:
            # Add new text entry action
            self.current_text_action = action
            self.recorded_actions.append(action)
            QMetaObject.invokeMethod(self, "add_action_to_list",
                                   Qt.QueuedConnection,
                                   Q_ARG(str, str(action)))
        else:
            # Update existing text entry action
            self.current_text_action["detail"] = self.current_text
            # Find and update the list widget item
            for i in range(self.actions_list.count()):
                item = self.actions_list.item(i)
                if eval(item.text())["act"] == "text_entry":
                    QMetaObject.invokeMethod(self, "update_action_in_list",
                                           Qt.QueuedConnection,
                                           Q_ARG(int, i),
                                           Q_ARG(str, str(action)))
                    break

    def finalize_text_entry(self):
        """Finalize current text entry"""
        if self.current_text:
            self.current_text = ""
            self.current_text_action = None

    def on_key_release(self, key):
        """Handle key release"""
        if not self.recording:
            return

        try:
            # Wait a bit to ensure all keys in combination are recorded
            time.sleep(0.1)

            # Add keyboard combination if we have recorded keys
            if self.keyboard_events:
                self.add_keyboard_combination()
                self.keyboard_events = []  # Clear events after adding combination

        except Exception as e:
            print(f"Error in key release: {e}")

    def add_text_entry(self):
        """Add accumulated text as a text_entry action"""
        if self.current_text:
            action = {
                "act": "text_entry",
                "detail": self.current_text,
                "repeat": 1
            }
            self.events_queue.put(action)
            self.current_text = ""

    def add_keyboard_combination(self):
        """Add current keyboard combination as an action"""
        if not self.keyboard_events:
            return

        # Define modifier keys
        modifiers = {'Ctrl', 'Alt', 'Shift', 'Windows'}
        has_modifier = any(k in modifiers for k in self.keyboard_events)
        all_printable = all(isinstance(k, str) and len(k) == 1 and k.isprintable() and not k.isspace() for k in self.keyboard_events)

        if has_modifier:
            # Sort to ensure modifiers come first
            def key_sort(k):
                if k in modifiers:
                    return 0, k
                return 1, k
            sorted_keys = sorted(self.keyboard_events, key=key_sort)
            key_str = " + ".join(sorted_keys)
            action = {
                "act": "press_key",
                "detail": key_str,
                "repeat": 1
            }
            self.events_queue.put(action)
        elif all_printable:
            # Record as text_entry for consecutive printable keys
            text = ''.join(self.keyboard_events)
            action = {
                "act": "text_entry",
                "detail": text,
                "repeat": 1
            }
            self.events_queue.put(action)
        else:
            # Fallback: record each as a single key press
            for k in self.keyboard_events:
                action = {
                    "act": "press_key",
                    "detail": k,
                    "repeat": 1
                }
                self.events_queue.put(action)

    def process_events(self):
        """Process recorded events with realtime display - merge similar consecutive actions"""
        while self.recording or not self.events_queue.empty():
            try:
                action = self.events_queue.get(timeout=0.1)

                # Check if we can merge with the last action
                merged = False
                if self.recorded_actions:
                    last_action = self.recorded_actions[-1]
                    if self.actions_match(action, last_action):
                        # Update repeat count
                        last_action['repeat'] = last_action.get('repeat', 1) + 1
                        # Update list widget display
                        if self.actions_list.count() > 0:  # Kiểm tra xem có item nào không
                            last_item_idx = self.actions_list.count() - 1
                            QMetaObject.invokeMethod(
                                self,
                                "update_action_in_list",
                                Qt.QueuedConnection,
                                Q_ARG(int, last_item_idx),
                                Q_ARG(str, str(last_action))
                            )
                        merged = True

                if not merged:
                    # Add as new action if couldn't merge
                    action['repeat'] = 1
                    self.recorded_actions.append(action)
                    # Add to list widget
                    QMetaObject.invokeMethod(
                        self,
                        "add_action_to_list",
                        Qt.QueuedConnection,
                        Q_ARG(str, str(action))
                    )

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing event: {str(e)}")
                continue

        if self.listener_thread:
            self.listener_thread = None

    def actions_match(self, action1, action2):
        """Compare two actions for equality, ignoring repeat count"""
        if action1['act'] != action2['act']:
            return False

        # For coordinate-based actions
        if action1['act'] in ['move_to', 'click', 'right_click',
                           'double_click', 'drag']:
            return action1['coordinates'] == action2['coordinates']

        # For text/key based actions
        elif action1['act'] in ['press_key', 'text_entry']:
            return action1['detail'] == action2['detail']

        # For time-based actions
        elif action1['act'] == 'time_sleep':
            return float(action1['detail']) == float(action2['detail'])

        # For app-related actions
        elif action1['act'] == 'open_app':
            return action1['detail'].lower() == action2['detail'].lower()

        # For scroll actions
        elif action1['act'] == 'scroll':
            return action1['detail'] == action2['detail']

        return False

    @Slot(int, str)
    def update_action_in_list(self, index, action_str):
        """Update existing action in list widget"""
        if hasattr(self, 'actions_list') and index < self.actions_list.count():
            self.actions_list.item(index).setText(action_str)
            self.actions_list.scrollToBottom()

    def on_move(self, x, y):
        """Handle mouse movement to detect drags and moves"""
        if not self.recording:
            return

        current_time = time.time()

        # Handle drag detection first
        if self.drag_start and not self.recording_drag:
            dist_x = abs(x - self.drag_start[0])
            dist_y = abs(y - self.drag_start[1])
            time_elapsed = current_time - self.drag_start_time

            if (dist_x > self.drag_threshold or dist_y > self.drag_threshold) and time_elapsed < 0.5:
                self.recording_drag = True
            return

        # Record move_to action if enough time has passed
        if current_time - self.last_move_time > self.move_threshold:
            action = {
                "act": "move_to",
                "detail": f"Move to ({x}, {y})",
                "coordinates": f"x={x}, y={y}",
                "repeat": 1
            }
            self.events_queue.put(action)
            self.last_move_time = current_time

    @Slot(str)
    def add_action_to_list(self, action_str):
        """Add action to list widget"""
        if hasattr(self, 'actions_list'):
            self.actions_list.addItem(action_str)
            self.actions_list.scrollToBottom()

    def reject(self):
        """Clean up before closing"""
        self.stop_recording()
        super().reject()

class ActionEditor(QDialog):
    def __init__(self, action=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Action Editor")
        self.setMinimumSize(400, 300)

        layout = QVBoxLayout(self)

        # Action type selector
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Action Type:"))
        self.action_type = QComboBox()
        self.action_type.addItems([
            "move_to", "click", "text_entry", "press_key",
            "open_app", "time_sleep", "right_click", "double_click",
            "hold_key_and_click", "scroll", "drag", "assistant_task"  # Added assistant_task
        ])
        type_layout.addWidget(self.action_type)
        layout.addLayout(type_layout)

        # Action description
        desc_layout = QHBoxLayout()
        desc_layout.addWidget(QLabel("Description:"))
        self.act_desc = QLineEdit()
        desc_layout.addWidget(self.act_desc)
        layout.addLayout(desc_layout)

        # Add repeat count for all actions
        repeat_widget = QWidget()
        repeat_layout = QHBoxLayout(repeat_widget)
        repeat_layout.addWidget(QLabel("Repeat count:"))
        self.repeat_count = QSpinBox()
        self.repeat_count.setRange(1, 100)
        self.repeat_count.setValue(1)
        repeat_layout.addWidget(self.repeat_count)
        repeat_layout.addStretch()
        layout.addWidget(repeat_widget)

        # Parameter widgets for different action types
        self.param_stack = QStackedWidget()
        self.create_parameter_widgets()
        layout.addWidget(self.param_stack)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Connect signals
        self.action_type.currentIndexChanged.connect(self.on_action_type_changed)

        # Load action if editing
        if action:
            self.load_action(action)

    def create_parameter_widgets(self):
        # Coordinates widget (move_to, click, etc)
        coord_widget = QWidget()
        coord_layout = QVBoxLayout(coord_widget)

        # Method selection
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        self.coord_method = QComboBox()
        self.coord_method.addItems(["Coordinates", "Element Description"])
        self.coord_method.currentTextChanged.connect(self.on_coord_method_changed)
        method_layout.addWidget(self.coord_method)
        coord_layout.addLayout(method_layout)

        # Coordinate input section
        self.coord_input_widget = QWidget()
        coord_input_layout = QVBoxLayout(self.coord_input_widget)
        self.x_coord = QSpinBox()
        self.y_coord = QSpinBox()
        self.x_coord.setRange(0, 9999)
        self.y_coord.setRange(0, 9999)
        coord_input_layout.addWidget(QLabel("X:"))
        coord_input_layout.addWidget(self.x_coord)
        coord_input_layout.addWidget(QLabel("Y:"))
        coord_input_layout.addWidget(self.y_coord)
        coord_layout.addWidget(self.coord_input_widget)

        # Element description section
        self.element_input_widget = QWidget()
        element_input_layout = QVBoxLayout(self.element_input_widget)
        element_input_layout.addWidget(QLabel("Element Description:"))
        self.element_description = QTextEdit()
        self.element_description.setPlaceholderText("Describe the element you want to interact with (e.g., 'red button with text Save', 'search box at top of page')")
        self.element_description.setMaximumHeight(80)
        element_input_layout.addWidget(self.element_description)
        coord_layout.addWidget(self.element_input_widget)

        # Initially show coordinates
        self.element_input_widget.hide()

        self.param_stack.addWidget(coord_widget)

        # Text entry widget
        text_widget = QWidget()
        text_layout = QVBoxLayout(text_widget)
        self.text_input = QTextEdit()
        text_layout.addWidget(QLabel("Text to enter:"))
        text_layout.addWidget(self.text_input)
        self.param_stack.addWidget(text_widget)

        # Key press widget with modifier support
        key_widget = QWidget()
        key_layout = QVBoxLayout(key_widget)

        # Modifier keys
        modifier_layout = QHBoxLayout()
        self.ctrl_check = QCheckBox("Ctrl")
        self.alt_check = QCheckBox("Alt")
        self.shift_check = QCheckBox("Shift")
        self.win_check = QCheckBox("Win")
        modifier_layout.addWidget(self.ctrl_check)
        modifier_layout.addWidget(self.alt_check)
        modifier_layout.addWidget(self.shift_check)
        modifier_layout.addWidget(self.win_check)

        # Main key dropdown
        self.key_combo = QComboBox()
        self.key_combo.addItems([
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
            "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12",
            "Enter", "Space", "Tab", "Backspace", "Delete", "Escape",
            "Home", "End", "Page Up", "Page Down",
            "Left", "Right", "Up", "Down"
        ])

        key_layout.addWidget(QLabel("Select modifiers (if needed):"))
        key_layout.addLayout(modifier_layout)
        key_layout.addWidget(QLabel("Select main key:"))
        key_layout.addWidget(self.key_combo)
        self.param_stack.addWidget(key_widget)

        # Wait time widget
        wait_widget = QWidget()
        wait_layout = QVBoxLayout(wait_widget)
        self.wait_time = QSpinBox()
        self.wait_time.setRange(1, 60)
        wait_layout.addWidget(QLabel("Wait time (seconds):"))
        wait_layout.addWidget(self.wait_time)
        self.param_stack.addWidget(wait_widget)

        # App name widget
        app_widget = QWidget()
        app_layout = QVBoxLayout(app_widget)
        self.app_input = QLineEdit()
        app_layout.addWidget(QLabel("Application name:"))
        app_layout.addWidget(self.app_input)
        self.param_stack.addWidget(app_widget)

        # Scroll widget
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        # Scroll method selection
        scroll_method_layout = QHBoxLayout()
        scroll_method_layout.addWidget(QLabel("Scroll Method:"))
        self.scroll_method = QComboBox()
        self.scroll_method.addItems(["Direction", "Until Element Visible"])
        self.scroll_method.currentTextChanged.connect(self.on_scroll_method_changed)
        scroll_method_layout.addWidget(self.scroll_method)
        scroll_layout.addLayout(scroll_method_layout)

        # Direction scroll section
        self.scroll_direction_widget = QWidget()
        scroll_direction_layout = QVBoxLayout(self.scroll_direction_widget)
        self.scroll_direction = QComboBox()
        self.scroll_direction.addItems(["up", "down", "left", "right"])
        scroll_direction_layout.addWidget(QLabel("Scroll direction:"))
        scroll_direction_layout.addWidget(self.scroll_direction)
        scroll_layout.addWidget(self.scroll_direction_widget)

        # Element scroll section
        self.scroll_element_widget = QWidget()
        scroll_element_layout = QVBoxLayout(self.scroll_element_widget)
        scroll_element_layout.addWidget(QLabel("Element to find:"))
        self.scroll_element_description = QTextEdit()
        self.scroll_element_description.setPlaceholderText("Describe the element to scroll until visible (e.g., 'Submit button', 'footer text')")
        self.scroll_element_description.setMaximumHeight(60)
        scroll_element_layout.addWidget(self.scroll_element_description)

        # Scroll direction for element search
        scroll_elem_dir_layout = QHBoxLayout()
        scroll_elem_dir_layout.addWidget(QLabel("Scroll Direction:"))
        self.scroll_element_direction = QComboBox()
        self.scroll_element_direction.addItems(["down", "up", "right", "left"])
        scroll_elem_dir_layout.addWidget(self.scroll_element_direction)
        scroll_element_layout.addLayout(scroll_elem_dir_layout)

        scroll_layout.addWidget(self.scroll_element_widget)

        # Initially show direction scroll
        self.scroll_element_widget.hide()

        self.param_stack.addWidget(scroll_widget)

        # Modify hold key and click widget
        hold_key_widget = QWidget()
        hold_key_layout = QVBoxLayout(hold_key_widget)

        # Hold key selection
        hold_key_container = QWidget()
        hold_key_controls = QHBoxLayout(hold_key_container)

        self.hold_key_input = QComboBox()
        self.hold_key_input.addItems([
            "Ctrl", "Alt", "Shift", "Win",
            "Left Ctrl", "Right Ctrl",
            "Left Alt", "Right Alt",
            "Left Shift", "Right Shift",
            "Left Win", "Right Win"
        ])

        hold_key_controls.addWidget(QLabel("Key to hold:"))
        hold_key_controls.addWidget(self.hold_key_input)
        hold_key_layout.addWidget(hold_key_container)

        # Method selection for hold key and click
        hold_method_layout = QHBoxLayout()
        hold_method_layout.addWidget(QLabel("Click Method:"))
        self.hold_coord_method = QComboBox()
        self.hold_coord_method.addItems(["Coordinates", "Element Description"])
        self.hold_coord_method.currentTextChanged.connect(self.on_hold_coord_method_changed)
        hold_method_layout.addWidget(self.hold_coord_method)
        hold_key_layout.addLayout(hold_method_layout)

        # Coordinate input section for hold key and click
        self.hold_coord_input_widget = QWidget()
        hold_coord_input_layout = QVBoxLayout(self.hold_coord_input_widget)
        hold_coord_input_layout.addWidget(QLabel("Click coordinates:"))
        self.hold_x_coord = QSpinBox()
        self.hold_y_coord = QSpinBox()
        self.hold_x_coord.setRange(0, 9999)
        self.hold_y_coord.setRange(0, 9999)
        hold_coord_input_layout.addWidget(QLabel("X:"))
        hold_coord_input_layout.addWidget(self.hold_x_coord)
        hold_coord_input_layout.addWidget(QLabel("Y:"))
        hold_coord_input_layout.addWidget(self.hold_y_coord)
        hold_key_layout.addWidget(self.hold_coord_input_widget)

        # Element description section for hold key and click
        self.hold_element_input_widget = QWidget()
        hold_element_input_layout = QVBoxLayout(self.hold_element_input_widget)
        hold_element_input_layout.addWidget(QLabel("Element Description:"))
        self.hold_element_description = QTextEdit()
        self.hold_element_description.setPlaceholderText("Describe the element to click while holding key (e.g., 'folder icon in sidebar')")
        self.hold_element_description.setMaximumHeight(80)
        hold_element_input_layout.addWidget(self.hold_element_description)
        hold_key_layout.addWidget(self.hold_element_input_widget)

        # Initially show coordinates
        self.hold_element_input_widget.hide()

        self.param_stack.addWidget(hold_key_widget)

        # Add drag coordinates widget
        drag_widget = QWidget()
        drag_layout = QVBoxLayout(drag_widget)

        # Drag method selection
        drag_method_layout = QHBoxLayout()
        drag_method_layout.addWidget(QLabel("Drag Method:"))
        self.drag_method = QComboBox()
        self.drag_method.addItems(["Coordinates", "Elements"])
        self.drag_method.currentTextChanged.connect(self.on_drag_method_changed)
        drag_method_layout.addWidget(self.drag_method)
        drag_layout.addLayout(drag_method_layout)

        # Coordinate-based drag section
        self.drag_coord_widget = QWidget()
        drag_coord_layout = QVBoxLayout(self.drag_coord_widget)

        # Start position
        drag_coord_layout.addWidget(QLabel("Start Position:"))
        start_pos = QWidget()
        start_layout = QHBoxLayout(start_pos)
        self.start_x = QSpinBox()
        self.start_y = QSpinBox()
        self.start_x.setRange(0, 9999)
        self.start_y.setRange(0, 9999)
        start_layout.addWidget(QLabel("X:"))
        start_layout.addWidget(self.start_x)
        start_layout.addWidget(QLabel("Y:"))
        start_layout.addWidget(self.start_y)
        drag_coord_layout.addWidget(start_pos)

        # End position
        drag_coord_layout.addWidget(QLabel("End Position:"))
        end_pos = QWidget()
        end_layout = QHBoxLayout(end_pos)
        self.end_x = QSpinBox()
        self.end_y = QSpinBox()
        self.end_x.setRange(0, 9999)
        self.end_y.setRange(0, 9999)
        end_layout.addWidget(QLabel("X:"))
        end_layout.addWidget(self.end_x)
        end_layout.addWidget(QLabel("Y:"))
        end_layout.addWidget(self.end_y)
        drag_coord_layout.addWidget(end_pos)

        drag_layout.addWidget(self.drag_coord_widget)

        # Element-based drag section
        self.drag_element_widget = QWidget()
        drag_element_layout = QVBoxLayout(self.drag_element_widget)

        # Start element
        drag_element_layout.addWidget(QLabel("Start Element:"))
        self.start_element_description = QTextEdit()
        self.start_element_description.setPlaceholderText("Describe the element to drag from (e.g., 'file icon named document.pdf')")
        self.start_element_description.setMaximumHeight(60)
        drag_element_layout.addWidget(self.start_element_description)

        # End element
        drag_element_layout.addWidget(QLabel("End Element:"))
        self.end_element_description = QTextEdit()
        self.end_element_description.setPlaceholderText("Describe the element to drag to (e.g., 'trash bin icon', 'folder named Downloads')")
        self.end_element_description.setMaximumHeight(60)
        drag_element_layout.addWidget(self.end_element_description)

        drag_layout.addWidget(self.drag_element_widget)

        # Initially show coordinates
        self.drag_element_widget.hide()

        self.param_stack.addWidget(drag_widget)

        # Assistant task widget
        assistant_widget = QWidget()
        assistant_layout = QVBoxLayout(assistant_widget)

        assistant_layout.addWidget(QLabel("Task description:"))
        self.task_description = QTextEdit()
        self.task_description.setPlaceholderText("Enter the task you want the assistant to perform...")
        self.task_description.setMaximumHeight(100)
        assistant_layout.addWidget(self.task_description)

        self.param_stack.addWidget(assistant_widget)

    def on_coord_method_changed(self, method):
        """Handle coordinate method change"""
        if method == "Coordinates":
            self.coord_input_widget.show()
            self.element_input_widget.hide()
        else:  # Element Description
            self.coord_input_widget.hide()
            self.element_input_widget.show()

    def on_scroll_method_changed(self, method):
        """Handle scroll method change"""
        if method == "Direction":
            self.scroll_direction_widget.show()
            self.scroll_element_widget.hide()
        else:  # Until Element Visible
            self.scroll_direction_widget.hide()
            self.scroll_element_widget.show()

    def on_drag_method_changed(self, method):
        """Handle drag method change"""
        if method == "Coordinates":
            self.drag_coord_widget.show()
            self.drag_element_widget.hide()
        else:  # Elements
            self.drag_coord_widget.hide()
            self.drag_element_widget.show()

    def on_hold_coord_method_changed(self, method):
        """Handle hold key and click coordinate method change"""
        if method == "Coordinates":
            self.hold_coord_input_widget.show()
            self.hold_element_input_widget.hide()
        else:  # Element Description
            self.hold_coord_input_widget.hide()
            self.hold_element_input_widget.show()

    def on_action_type_changed(self, index):
        """Handle action type change"""
        action_type = self.action_type.currentText()

        # Map action types to parameter widget indices
        type_to_widget = {
            "move_to": 0,
            "click": 0,
            "right_click": 0,
            "double_click": 0,
            "hold_key_and_click": 0,
            "text_entry": 1,
            "press_key": 2,
            "time_sleep": 3,
            "open_app": 4,
            "scroll": 5,
            "hold_key_and_click": 6,
            "drag": 7,
            "assistant_task": 8
        }

        self.param_stack.setCurrentIndex(type_to_widget.get(action_type, 0))

    def load_action(self, action):
        """Load action data into editor"""
        try:
            if isinstance(action, str):
                action = eval(action)

            # Set action type
            self.action_type.setCurrentText(action['act'])

            # Set description
            self.act_desc.setText(action.get('detail', ''))

            # Set repeat count
            self.repeat_count.setValue(action.get('repeat', 1))

            # Set parameters based on action type
            if 'coordinates' in action:
                coords = action['coordinates'].split(',')
                x = int(coords[0].split('=')[1])
                y = int(coords[1].split('=')[1])
                if action['act'] == 'hold_key_and_click':
                    self.hold_x_coord.setValue(x)
                    self.hold_y_coord.setValue(y)
                    key = action['detail'].split(" and click")[0]
                    self.hold_key_input.setCurrentText(key)
                elif action['act'] == 'drag':
                    # Parse drag coordinates
                    start_coords, end_coords = action['coordinates'].split(" to ")
                    start_x, start_y = map(float, re.findall(r'x=(\d+\.?\d*), y=(\d+\.?\d*)', start_coords)[0])
                    end_x, end_y = map(float, re.findall(r'x=(\d+\.?\d*), y=(\d+\.?\d*)', end_coords)[0])
                    self.start_x.setValue(int(start_x))
                    self.start_y.setValue(int(start_y))
                    self.end_x.setValue(int(end_x))
                    self.end_y.setValue(int(end_y))
                else:
                    self.x_coord.setValue(x)
                    self.y_coord.setValue(y)
            elif action['act'] == 'text_entry':
                self.text_input.setText(action['detail'])
            elif action['act'] == 'press_key':
                key_parts = action['detail'].split(' + ')
                for part in key_parts:
                    part = part.strip()
                    # Update the check to handle Windows key variations
                    if part.lower() in ['ctrl', 'alt', 'shift', 'windows', 'win', 'cmd']:
                        if part.lower() in ['windows', 'win', 'cmd']:
                            self.win_check.setChecked(True)
                        elif part == 'Ctrl':
                            self.ctrl_check.setChecked(True)
                        elif part == 'Alt':
                            self.alt_check.setChecked(True)
                        elif part == 'Shift':
                            self.shift_check.setChecked(True)
                    else:
                        self.key_combo.setCurrentText(part)
            elif action['act'] == 'time_sleep':
                self.wait_time.setValue(int(action['detail']))
            elif action['act'] == 'open_app':
                self.app_input.setText(action['detail'])
            elif action['act'] == 'scroll':
                # Handle both old and new scroll formats
                if 'scroll_method' in action:
                    if action['scroll_method'] == 'element':
                        self.scroll_method.setCurrentText("Until Element Visible")
                        self.scroll_element_description.setText(action.get('element_description', ''))
                        self.scroll_element_direction.setCurrentText(action.get('scroll_direction', 'down'))
                    else:
                        self.scroll_method.setCurrentText("Direction")
                        self.scroll_direction.setCurrentText(action['detail'])
                else:
                    # Legacy format - just direction
                    self.scroll_method.setCurrentText("Direction")
                    self.scroll_direction.setCurrentText(action['detail'])
            elif action['act'] == 'assistant_task':
                self.task_description.setText(action['detail'])

            # Handle element descriptions for mouse actions
            if 'element_description' in action and action['act'] in ['move_to', 'click', 'right_click', 'double_click']:
                self.coord_method.setCurrentText("Element Description")
                self.element_description.setText(action['element_description'])
            elif 'coordinates' in action and action['act'] in ['move_to', 'click', 'right_click', 'double_click']:
                self.coord_method.setCurrentText("Coordinates")

            # Handle hold_key_and_click element descriptions separately
            if action['act'] == 'hold_key_and_click':
                if 'element_description' in action:
                    self.hold_coord_method.setCurrentText("Element Description")
                    self.hold_element_description.setText(action['element_description'])
                else:
                    self.hold_coord_method.setCurrentText("Coordinates")

            # Handle drag element descriptions
            if action['act'] == 'drag':
                if 'start_element_description' in action and 'end_element_description' in action:
                    self.drag_method.setCurrentText("Elements")
                    self.start_element_description.setText(action['start_element_description'])
                    self.end_element_description.setText(action['end_element_description'])
                else:
                    self.drag_method.setCurrentText("Coordinates")

        except Exception as e:
            print_to_chat(f"Error loading action: {str(e)}")

    def get_action_data(self):
        """Get action data from editor"""
        try:
            action_type = self.action_type.currentText()

            # Base action data with repeat count
            action = {
                "act": action_type,
                "detail": self.act_desc.text() or "Custom action detail",
                "repeat": self.repeat_count.value()
            }

            # Add parameters based on action type
            if action_type == "hold_key_and_click":
                key = self.hold_key_input.currentText()
                if self.hold_coord_method.currentText() == "Element Description":
                    action["element_description"] = self.hold_element_description.toPlainText()
                    action["method"] = "element"
                else:
                    action["coordinates"] = f"x={self.hold_x_coord.value()}, y={self.hold_y_coord.value()}"
                    action["method"] = "coordinates"
                action["detail"] = f"{key} and click"
            elif action_type in ["move_to", "click", "right_click", "double_click"]:
                if self.coord_method.currentText() == "Element Description":
                    action["element_description"] = self.element_description.toPlainText()
                    action["method"] = "element"
                else:
                    action["coordinates"] = f"x={self.x_coord.value()}, y={self.y_coord.value()}"
                    action["method"] = "coordinates"
            elif action_type == "text_entry":
                action["detail"] = self.text_input.toPlainText()
            elif action_type == "press_key":
                key_parts = []
                if self.ctrl_check.isChecked(): key_parts.append("Ctrl")
                if self.alt_check.isChecked(): key_parts.append("Alt")
                if self.shift_check.isChecked(): key_parts.append("Shift")
                if self.win_check.isChecked(): key_parts.append("Windows")
                key_parts.append(self.key_combo.currentText())
                action["detail"] = " + ".join(key_parts)
            elif action_type == "time_sleep":
                action["detail"] = str(self.wait_time.value())
            elif action_type == "open_app":
                action["detail"] = self.app_input.text()
            elif action_type == "scroll":
                if self.scroll_method.currentText() == "Until Element Visible":
                    action["scroll_method"] = "element"
                    action["element_description"] = self.scroll_element_description.toPlainText()
                    action["scroll_direction"] = self.scroll_element_direction.currentText()
                    action["detail"] = f"scroll until {self.scroll_element_description.toPlainText()}"
                else:
                    action["scroll_method"] = "direction"
                    action["detail"] = self.scroll_direction.currentText()
            elif action_type == "drag":
                if self.drag_method.currentText() == "Elements":
                    action["method"] = "elements"
                    action["start_element_description"] = self.start_element_description.toPlainText()
                    action["end_element_description"] = self.end_element_description.toPlainText()
                    action["detail"] = f"drag from {self.start_element_description.toPlainText()} to {self.end_element_description.toPlainText()}"
                else:
                    action["method"] = "coordinates"
                    action["coordinates"] = f"x={self.start_x.value()}, y={self.start_y.value()} to x={self.end_x.value()}, y={self.end_y.value()}"
            elif action_type == "assistant_task":
                action["detail"] = self.task_description.toPlainText()

            return action

        except Exception as e:
            QMessageBox.warning(
                self,
                "Error",
                f"Error getting action data: {str(e)}"
            )
            return None

class TaskEditor(QDialog):
    def __init__(self, task_name=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Task Editor")
        self.setMinimumSize(600, 400)

        layout = QVBoxLayout(self)

        # Task name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Task Name:"))
        self.task_name = QLineEdit()
        name_layout.addWidget(self.task_name)
        layout.addLayout(name_layout)

        # Add task description
        desc_layout = QHBoxLayout()
        desc_layout.addWidget(QLabel("Description:"))
        self.task_description = QTextEdit()
        self.task_description.setMaximumHeight(100)
        desc_layout.addWidget(self.task_description)
        layout.addLayout(desc_layout)

        # Actions list
        self.actions_list = QListWidget()
        self.actions_list.itemDoubleClicked.connect(self.edit_selected_action)
        layout.addWidget(self.actions_list)

        # Action buttons
        action_buttons = QHBoxLayout()
        self.add_action = QPushButton("Add Action")
        self.edit_action = QPushButton("Edit Action")
        self.remove_action = QPushButton("Remove Action")
        self.move_up = QPushButton("Move Up")
        self.move_down = QPushButton("Move Down")
        self.record_actions = QPushButton("Record Actions")
        self.clear_all = QPushButton("Clear All")  # Thêm nút Clear All

        action_buttons.addWidget(self.add_action)
        action_buttons.addWidget(self.edit_action)
        action_buttons.addWidget(self.remove_action)
        action_buttons.addWidget(self.move_up)
        action_buttons.addWidget(self.move_down)
        action_buttons.addWidget(self.record_actions)
        action_buttons.addWidget(self.clear_all)  # Thêm nút vào layout
        layout.addLayout(action_buttons)

        # Dialog buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Connect signals
        self.add_action.clicked.connect(self.add_new_action)
        self.edit_action.clicked.connect(self.edit_selected_action)
        self.remove_action.clicked.connect(self.remove_selected_action)
        self.move_up.clicked.connect(self.move_action_up)
        self.move_down.clicked.connect(self.move_action_down)
        self.record_actions.clicked.connect(self.start_recording)
        self.clear_all.clicked.connect(self.clear_all_actions)  # Kết nối sự kiện

        # Load task if editing
        if task_name:
            self.task_name.setText(task_name)
            # Load task actions and description
            task_data = get_task(task_name)
            if isinstance(task_data, dict):
                actions = task_data.get('actions', [])
                description = task_data.get('description', '')
                self.task_description.setText(description)
                for action in actions:
                    self.actions_list.addItem(str(action))
            else:
                # Handle legacy format
                for action in task_data:
                    self.actions_list.addItem(str(action))

    def clear_all_actions(self):
        """Clear all actions from the list after confirmation"""
        if self.actions_list.count() > 0:
            reply = QMessageBox.question(
                self, 'Clear All Actions',
                'Are you sure you want to clear all actions?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self.actions_list.clear()

    def add_new_action(self):
        """Add a new action to the task"""
        try:
            dialog = ActionEditor(parent=self)
            if dialog.exec() == QDialog.Accepted:
                action = dialog.get_action_data()
                if action:
                    # Add action directly to list widget
                    self.actions_list.addItem(str(action))
        except Exception as e:
            QMessageBox.warning(
                self,
                "Error",
                f"Error adding action: {str(e)}"
            )

    def edit_selected_action(self):
        current = self.actions_list.currentItem()
        if current:
            dialog = ActionEditor(eval(current.text()), self)
            if dialog.exec() == QDialog.Accepted:
                current.setText(str(dialog.get_action_data()))

    def remove_selected_action(self):
        current = self.actions_list.currentRow()
        if current >= 0:
            self.actions_list.takeItem(current)

    def start_recording(self):
        dialog = TaskRecorder(self)
        if dialog.exec() == QDialog.Accepted:
            for action in dialog.recorded_actions:
                self.actions_list.addItem(str(action))

    def move_action_up(self):
        """Move the selected action up in the list"""
        current_row = self.actions_list.currentRow()
        if current_row > 0:  # Can't move the first item up
            # Take the current item
            item = self.actions_list.takeItem(current_row)
            # Insert it one position up
            self.actions_list.insertItem(current_row - 1, item)
            # Keep the item selected
            self.actions_list.setCurrentRow(current_row - 1)

    def move_action_down(self):
        """Move the selected action down in the list"""
        current_row = self.actions_list.currentRow()
        if current_row >= 0 and current_row < self.actions_list.count() - 1:  # Can't move the last item down
            # Take the current item
            item = self.actions_list.takeItem(current_row)
            # Insert it one position down
            self.actions_list.insertItem(current_row + 1, item)
            # Keep the item selected
            self.actions_list.setCurrentRow(current_row + 1)

    def clear_all_actions(self):
        """Clear all actions from the list"""
        reply = QMessageBox.question(
            self,
            "Clear All Actions",
            "Are you sure you want to clear all actions?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.actions_list.clear()

    def accept(self):
        """Save task when OK is clicked"""
        name = self.task_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Error", "Task name is required!")
            return

        # Collect actions and description
        actions = []
        try:
            if self.actions_list.count() > 0:
                for i in range(self.actions_list.count()):
                    item = self.actions_list.item(i)
                    if item:
                        action = eval(item.text())
                        actions.append(action)

            # --- Merge consecutive press_key actions with typable characters ---
            def is_typable_press_key(action):
                if action.get('act') != 'text_entry':
                    return False
                val = action.get('detail', '')
                # Only merge if it's a single visible character (not special keys)
                return isinstance(val, str) and val.isprintable()

            merged_actions = []
            buffer = ''
            for action in actions:
                if is_typable_press_key(action):
                    buffer += action['detail']
                else:
                    if buffer:
                        merged_actions.append({'act': 'text_entry', 'detail': buffer, 'repeat': 1})
                        buffer = ''
                    merged_actions.append(action)
            if buffer:
                merged_actions.append({'act': 'text_entry', 'detail': buffer, 'repeat': 1})

            # Create task data with description
            task_data = {
                'description': self.task_description.toPlainText(),
                'actions': merged_actions
            }
            # Save task
            save_task(name, task_data)
            super().accept()
        except Exception as e:
            QMessageBox.warning(
                self,
                "Error",
                f"Error saving task: {str(e)}"
            )

class ScheduleEditor(QDialog):
    def __init__(self, task_name="", schedule_settings=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Schedule Editor")
        self.setMinimumSize(400, 300)

        layout = QVBoxLayout(self)

        # Schedule type selector
        schedule_type = QHBoxLayout()
        schedule_type.addWidget(QLabel("Schedule Type:"))
        self.schedule_combo = QComboBox()
        self.schedule_combo.addItems(["Daily", "Weekly", "Monthly", "Yearly"])
        self.schedule_combo.currentTextChanged.connect(self.on_schedule_type_changed)
        schedule_type.addWidget(self.schedule_combo)
        layout.addLayout(schedule_type)

        # Schedule options stack
        self.schedule_stack = QStackedWidget()

        # Daily options (just time)
        daily_widget = QWidget()
        daily_layout = QHBoxLayout(daily_widget)
        daily_layout.addWidget(QLabel("Time:"))
        self.daily_time = QTimeEdit()
        self.daily_time.setDisplayFormat("HH:mm")
        daily_layout.addWidget(self.daily_time)
        self.schedule_stack.addWidget(daily_widget)

        # Weekly options
        weekly_widget = QWidget()
        weekly_layout = QVBoxLayout(weekly_widget)

        weekday_layout = QHBoxLayout()
        weekday_layout.addWidget(QLabel("Day:"))
        self.weekday_combo = QComboBox()
        self.weekday_combo.addItems(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        weekday_layout.addWidget(self.weekday_combo)
        weekly_layout.addLayout(weekday_layout)

        weekly_time_layout = QHBoxLayout()
        weekly_time_layout.addWidget(QLabel("Time:"))
        self.weekly_time = QTimeEdit()
        self.weekly_time.setDisplayFormat("HH:mm")
        weekly_time_layout.addWidget(self.weekly_time)
        weekly_layout.addLayout(weekly_time_layout)

        self.schedule_stack.addWidget(weekly_widget)

        # Monthly options
        monthly_widget = QWidget()
        monthly_layout = QVBoxLayout(monthly_widget)

        day_layout = QHBoxLayout()
        day_layout.addWidget(QLabel("Day of month:"))
        self.monthly_day = QSpinBox()
        self.monthly_day.setRange(1, 31)
        day_layout.addWidget(self.monthly_day)
        monthly_layout.addLayout(day_layout)

        monthly_time_layout = QHBoxLayout()
        monthly_time_layout.addWidget(QLabel("Time:"))
        self.monthly_time = QTimeEdit()
        self.monthly_time.setDisplayFormat("HH:mm")
        monthly_time_layout.addWidget(self.monthly_time)
        monthly_layout.addLayout(monthly_time_layout)

        self.schedule_stack.addWidget(monthly_widget)

        # Yearly options
        yearly_widget = QWidget()
        yearly_layout = QVBoxLayout(yearly_widget)

        date_layout = QHBoxLayout()
        date_layout.addWidget(QLabel("Date:"))
        self.yearly_date = QDateEdit()
        self.yearly_date.setDisplayFormat("MMM d")
        date_layout.addWidget(self.yearly_date)
        yearly_layout.addLayout(date_layout)

        yearly_time_layout = QHBoxLayout()
        yearly_time_layout.addWidget(QLabel("Time:"))
        self.yearly_time = QTimeEdit()
        self.yearly_time.setDisplayFormat("HH:mm")
        yearly_time_layout.addWidget(self.yearly_time)
        yearly_layout.addLayout(yearly_time_layout)

        self.schedule_stack.addWidget(yearly_widget)

        layout.addWidget(self.schedule_stack)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        # Load schedule settings if editing
        if schedule_settings:
            self.load_schedule(schedule_settings)

    def on_schedule_type_changed(self, schedule_type):
        """Handle schedule type changes"""
        type_to_index = {
            "Daily": 0,
            "Weekly": 1,
            "Monthly": 2,
            "Yearly": 3
        }
        self.schedule_stack.setCurrentIndex(type_to_index.get(schedule_type, 0))

    def load_schedule(self, settings):
        """Load existing schedule settings"""

        self.schedule_combo.setCurrentText(settings["type"])
        if settings["type"] == "Daily":
            time_parts = settings["time"].split(":")
            self.daily_time.setTime(QTime(int(time_parts[0]), int(time_parts[1])))
        elif settings["type"] == "Weekly":
            self.weekday_combo.setCurrentText(settings["day"])
            time_parts = settings["time"].split(":")
            self.weekly_time.setTime(QTime(int(time_parts[0]), int(time_parts[1])))
        elif settings["type"] == "Monthly":
            self.monthly_day.setValue(int(settings["day"]))
            time_parts = settings["time"].split(":")
            self.monthly_time.setTime(QTime(int(time_parts[0]), int(time_parts[1])))
        elif settings["type"] == "Yearly":
            date_parts = settings["date"].split("-")
            self.yearly_date.setDate(QDate(2024, int(date_parts[0]), int(date_parts[1])))
            time_parts = settings["time"].split(":")
            self.yearly_time.setTime(QTime(int(time_parts[0]), int(time_parts[1])))

    def get_schedule_settings(self):
        """Get current schedule settings as serializable data"""
        schedule_type = self.schedule_combo.currentText()
        if schedule_type == "None":
            return None

        settings = {"type": schedule_type}

        # Convert Qt time/date objects to strings
        if schedule_type == "Daily":
            settings["time"] = self.daily_time.time().toString("HH:mm")
            settings["day"] = None
            settings["date"] = None
        elif schedule_type == "Weekly":
            settings["day"] = self.weekday_combo.currentText()
            settings["time"] = self.weekly_time.time().toString("HH:mm")
            settings["date"] = None
        elif schedule_type == "Monthly":
            settings["day"] = self.monthly_day.value()
            settings["time"] = self.monthly_time.time().toString("HH:mm")
            settings["date"] = None
        elif schedule_type == "Yearly":
            date = self.yearly_date.date()
            settings["date"] = f"{date.month()}-{date.day()}"
            settings["time"] = self.yearly_time.time().toString("HH:mm")
            settings["day"] = None

        return settings

class MiniControlInterface(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent, Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setWindowTitle("")
        self.setFixedSize(300, 80)

        # Position at bottom right
        screen = QApplication.primaryScreen().geometry()
        self.move(screen.width() - 320, screen.height() - 120)

        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Task label with bold font
        self.task_label = QLabel("No task selected")
        self.task_label.setAlignment(Qt.AlignCenter)
        font = self.task_label.font()
        font.setBold(True)
        self.task_label.setFont(font)
        layout.addWidget(self.task_label)

        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        # Control buttons container
        control_container = QWidget()
        control_layout = QHBoxLayout(control_container)
        control_layout.setContentsMargins(0, 0, 0, 0)

        # Play/Pause button
        self.play_pause_button = QPushButton("▶")
        self.play_pause_button.setFixedSize(40, 30)

        # Stop button
        self.stop_button = QPushButton("◼")
        self.stop_button.setFixedSize(40, 30)
        self.stop_button.setEnabled(False)

        control_layout.addStretch()
        control_layout.addWidget(self.play_pause_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addStretch()

        layout.addWidget(control_container)

        # Make window draggable
        self.oldPos = None

        # Style
        self.setStyleSheet("""
            MiniControlInterface {
                background-color: #FFFFFF;
                border: 1px solid #CCCCCC;
                border-radius: 10px;
            }
            QLabel {
                background-color: transparent;
                font-size: 11px;
                color: #424242;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border-radius: 3px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
            QWidget#control_container {
                background-color: transparent;
            }
        """)

        # Ensure the control container has an object name for styling
        control_container.setObjectName("control_container")

    def mousePressEvent(self, event):
        self.oldPos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        if self.oldPos:
            delta = event.globalPosition().toPoint() - self.oldPos
            self.move(self.pos() + delta)
            self.oldPos = event.globalPosition().toPoint()

    def update_task(self, task_name):
        """Update the displayed task name"""
        self.task_label.setText(task_name or "No task selected")

    def update_status(self, status, is_executing, is_paused):
        """Update the status text"""
        self.status_label.setText(status)
        self.update_execution_state(is_executing, is_paused)

    def update_execution_state(self, is_executing, is_paused):
        """Update button states based on execution state"""
        self.stop_button.setEnabled(is_executing)
        if not is_executing:
            self.play_pause_button.setText("▶")
            self.play_pause_button.setStyleSheet("""
                QPushButton {
                    background-color: #2196F3;
                }
                QPushButton:hover {
                    background-color: #1976D2;
                }
            """)
        else:
            if is_paused:
                self.play_pause_button.setText("▶")
            else:
                self.play_pause_button.setText("⏸")
            self.play_pause_button.setStyleSheet("""
                QPushButton {
                    background-color: #F44336;
                }
                QPushButton:hover {
                    background-color: #D32F2F;
                }
            """)

def create_app():
    load_settings()

    # Enforce single-instance using a lock file + local server to activate existing instance
    lock_path = os.path.join(tempfile.gettempdir(), 'ComputerAutopilot.lock')
    app_lock = QLockFile(lock_path)
    app_lock.setStaleLockTime(0)
    if not app_lock.tryLock(1):
        # Another instance is running; try to activate it and exit
        try:
            sock = QLocalSocket()
            sock.connectToServer('ComputerAutopilotInstance')
            if sock.waitForConnected(200):
                sock.write(b'activate')
                sock.flush()
                sock.waitForBytesWritten(200)
            sock.disconnectFromServer()
        except Exception:
            pass
        sys.exit(0)

    app = QApplication([])

    # Set app ID for Windows taskbar icon
    if os.name == 'nt':  # Windows
        myappid = 'mycompany.computerautopilot.desktop.v1'
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        except Exception:
            pass

    # Set application-wide icon
    app_icon = QIcon("media/icon.png")
    app.setWindowIcon(app_icon)

    window = ModernChatInterface()
    window.show()

    # Local server to receive activation requests from subsequent launches
    server = QLocalServer()
    server_name = 'ComputerAutopilotInstance'
    try:
        # Clean up any stale server
        QLocalServer.removeServer(server_name)
        server.listen(server_name)
    except Exception:
        try:
            QLocalServer.removeServer(server_name)
            server.listen(server_name)
        except Exception:
            pass

    def _on_new_conn():
        try:
            s = server.nextPendingConnection()
            if s is not None:
                # best-effort read; not strictly needed
                s.waitForReadyRead(50)
                try:
                    if hasattr(window, 'restore_from_tray'):
                        window.restore_from_tray()
                    else:
                        window.showNormal()
                        window.raise_()
                        window.activateWindow()
                except Exception:
                    pass
                s.close()
        except Exception:
            pass

    server.newConnection.connect(_on_new_conn)

    # Ensure lock is released on quit
    app.aboutToQuit.connect(lambda: app_lock.unlock())

    set_app_instance(window)
    app.exec()

if __name__ == "__main__":
    create_app()