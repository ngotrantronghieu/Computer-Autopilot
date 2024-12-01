app_instance = None  # Global variable to store the app instance

def print_to_chat(message, is_user=False):
    """Prints a message to the chat interface."""
    global app_instance
    if app_instance is not None:
        app_instance.root.after(0, lambda: app_instance.add_message(str(message), is_user=is_user))

def set_app_instance(app):
    """Sets the global app instance."""
    global app_instance
    app_instance = app