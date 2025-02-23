from deep_translator import GoogleTranslator
from PySide6.QtCore import QObject, Signal
from langdetect import detect

class ChatSignals(QObject):
    message_received = Signal(str, bool)  # message, is_user

app_instance = None
chat_signals = ChatSignals()

def translate(text, target_lang):
    """Translates text if it's not in the target language."""
    try:
        detected_lang = detect(text)
        if detected_lang != target_lang:
            translator = GoogleTranslator(source='auto', target=target_lang)
            return translator.translate(text)
    except Exception as e:
        print(f"Translation error: {str(e)}")
    return text

def print_to_chat(message, is_user=False):
    """Prints a message to the chat interface with language support."""
    global app_instance, chat_signals
    if app_instance is not None:
        # Determine the target language based on user's input language
        target_lang = getattr(app_instance, 'current_language', 'en')
        
        # Translate message if needed
        translated_message = translate(str(message), target_lang)
        
        # Emit signal to add message in the main thread
        chat_signals.message_received.emit(translated_message, is_user)

def set_app_instance(app):
    """Sets the global app instance and connects signals."""
    global app_instance
    app_instance = app
    
    # Connect the message signal to the app's add_message method
    chat_signals.message_received.connect(app.add_message)

def get_app_instance():
    """Gets the global app instance."""
    global app_instance
    return app_instance