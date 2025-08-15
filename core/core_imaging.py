import pyautogui
import pygetwindow as gw
import base64
import io
import os
import litellm
from PIL import Image
from mouse_detection import get_cursor_shape

# Function to focus a window given its title
def focus_window(window_title):
    try:
        window = gw.getWindowsWithTitle(window_title)[0]  # Get the first window with the specific title
        window.activate()
        pyautogui.sleep(0.3)  # Allow some time for the window to come into focus
        return window
    except IndexError:
        print(f'No window with title "{window_title}" found.')
        return None


# Function to capture a screenshot of the specified window
def capture_screenshot(window=None, region=None):
    # Reduced code for brevity
    if region is not None:
        screenshot = pyautogui.screenshot(region=region)
    elif window is not None:
        window_box = window.box
        screenshot = pyautogui.screenshot(region=(window_box.left, window_box.top, window_box.width, window_box.height))
    else:
        screenshot = pyautogui.screenshot()
    return screenshot


# Function to encode image data to base64
def encode_image(image_data):
    return base64.b64encode(image_data).decode('utf-8')


# Function to analyze an image using LiteLLM
def analyze_image(base64_image, window_title, additional_context='What’s in this image?'):
    from core_api import current_vision_llm_model, current_vision_api_key_env_name
    
    prompt = additional_context
    image_url = f"data:image/png;base64,{base64_image}"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_url}
                }
            ]
        }
    ]

    # Provider-specific kwargs based on model
    try:
        from core_api import get_provider_kwargs_for_model, current_vision_provider_options
        provider_kwargs = get_provider_kwargs_for_model(current_vision_llm_model or "")
        merged_kwargs = {**provider_kwargs, **current_vision_provider_options}
    except Exception:
        merged_kwargs = {}

    response = litellm.completion(
        model=current_vision_llm_model,
        messages=messages,
        api_key=os.environ.get(current_vision_api_key_env_name),
        **merged_kwargs,
    )
    return response


# Improved function to both capture and analyze a specific region screenshot
def imaging(window_title=None, additional_context=None, x=None, y=None, screenshot_size=None, current_cursor_shape=None):
    window = None
    region = None

    if screenshot_size == 'Full screen':
        pass
    elif window_title:
        window = focus_window(window_title)
        if not window:
            return None
        if screenshot_size and type(screenshot_size) == tuple and x is not None and y is not None:
            offset_x, offset_y = screenshot_size[0] // 2, screenshot_size[1] // 2
            window_box = window.box
            region = (
            window_box.left + x - offset_x, window_box.top + y - offset_y, screenshot_size[0], screenshot_size[1])
        else:
            region = (window.box.left, window.box.top, window.box.width, window.box.height)

    screenshot = capture_screenshot(window, region)

    if current_cursor_shape is None:
        current_cursor_shape = get_cursor_shape()
    
    # Map cursor shapes to corresponding cursor image files and their hotspots
    cursor_images = {
        "Arrow": {
            "path": r'media\cursor.png',
            "hotspot": (4, 0)
        },
        "I-beam (Active for text input)": {
            "path": r'media\text_cursor.png',
            "hotspot": (8, 8)
        },
        "Hand": {
            "path": r'media\hand_cursor.png',
            "hotspot": (8, 0)
        },
        "Wait/Busy (Hourglass)": {
            "path": r'media\wait_cursor.png',
            "hotspot": (8, 8)
        },
        "Cross": {
            "path": r'media\cross_cursor.png',
            "hotspot": (8, 8)
        },
        "Other": {
            "path": r'media\cursor.png',
            "hotspot": (4, 0)
        }
    }
    
    # Select appropriate cursor image and hotspot
    cursor_info = cursor_images.get(current_cursor_shape, cursor_images["Other"])
    cursor_img_path = cursor_info["path"]
    hotspot_x, hotspot_y = cursor_info["hotspot"]
    
    with Image.open(cursor_img_path) as cursor:
        cursor = cursor.convert("RGBA")
        cursor = cursor.resize((20, 20), Image.Resampling.LANCZOS)
        x_cursor, y_cursor = pyautogui.position()

        if region:
            # Adjust cursor position by subtracting region offset and hotspot
            cursor_pos = (
                x_cursor - region[0] - hotspot_x, 
                y_cursor - region[1] - hotspot_y
            )
        else:
            # Only adjust for hotspot when taking full screenshot
            cursor_pos = (x_cursor - hotspot_x, y_cursor - hotspot_y)

        screenshot.paste(cursor, cursor_pos, cursor)

    # Save and preview
    # preview_path = "preview_output.png"
    # screenshot.save(preview_path)
    # screenshot.show()

    with io.BytesIO() as output_bytes:
        screenshot.save(output_bytes, 'PNG')
        bytes_data = output_bytes.getvalue()

    base64_image = encode_image(bytes_data)
    analysis_result = analyze_image(base64_image, window_title, additional_context)

    return analysis_result


if __name__ == "__main__":
    analysis_result = imaging(
        screenshot_size='Full screen'
    )
    print(analysis_result)
