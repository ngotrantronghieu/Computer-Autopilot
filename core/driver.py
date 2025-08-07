import json
import os
import psutil
import pyautogui
import pygetwindow as gw
import pyperclip
import re
import sqlite3
import subprocess
import time
import warnings
import win32gui
import win32process
from bs4 import BeautifulSoup
from core_api import api_call
from core_imaging import imaging
from last_app import last_programs_list
from mouse_detection import get_cursor_shape
from pywinauto import Application
from seleniumbase.common.exceptions import TimeoutException
from seleniumbase import SB
from tasks import load_tasks
from topmost_window import focus_topmost_window
from utils import print_to_chat, print_to_web_chat
from voice import speaker
from window_elements import analyze_app
from window_focus import activate_window_title, get_installed_apps_registry
warnings.simplefilter("ignore", UserWarning)
if os.name == 'nt':  # Windows only
    from subprocess import CREATE_NO_WINDOW

enable_semantic_router_map = True
_stop_requested = False
action_cache = {}
# Global browser and driver instances that persist between tasks
_sb_instance = None
_driver = None

def request_stop():
    global _stop_requested
    _stop_requested = True

def clear_stop():
    global _stop_requested
    _stop_requested = False

def is_stop_requested():
    return _stop_requested

def get_settings():
    from assistant import load_settings
    return load_settings()

def auto_role(message):
    assistant_call = [{
        "role": "system",
        "content": (
            "You are an AI assistant that receives a message from the user and responds with the best role related to the message.\n"
            "You can choose between the following roles and decide what fits the best:\n"
            "windows_assistant - An assistant that can perform Windows application driver testcases to achieve a goal. It can handle online data, play, pause, stream media or operate the whole computer.\n"
            "joyful_conversation - Use this role if the user isn't looking for performing anything on Windows.\n"
            "Only respond with the name of the role to use. Modify your response to match the message subject.\n"
            "If the message seems to be related to Windows, like opening an application, searching, browsing, media, or social networks, choose windows_assistant.\n"
            "If the message seems to be related with generating or writing content, choose windows_assistant.\n"
            "If the message seems that the user is trying to do something with content, choose windows_assistant.\n"
            "Otherwise, if the user is just asking question or having conversation, choose joyful_conversation."
        )
    }, {
        "role": "user",
        "content": f"Message: {message}"
    }]
    
    return api_call(assistant_call, max_tokens=50)

def app_space_map(map='', app_space_filepath="app_space_map.json"):
    """Get application space mapping for context and shortcuts."""
    try:
        with open(app_space_filepath, "r") as f:
            app_space_data = json.load(f)
    except FileNotFoundError:
        print_to_chat(f"Warning: App space file not found at {app_space_filepath}. Using default map.")
        app_space_data = {}
    except json.JSONDecodeError:
        print_to_chat(f"Error: Invalid JSON format in {app_space_filepath}. Using default map.")
        app_space_data = {}

    def convert_to_string(data):
        if isinstance(data, list):
            return "\n".join(data)
        return str(data)

    if 'app_space' in map and enable_semantic_router_map:
        element_map = []
        for app_key, element_data in app_space_data.items():
            if app_key != "keyboard_shortcuts":  # Bỏ qua phần keyboard shortcuts
                if isinstance(element_data, dict):
                    app_info = "\n".join(
                        f"{key}:\n{convert_to_string(value)}"
                        for key, value in element_data.items()
                    )
                else:
                    app_info = convert_to_string(element_data)
                element_map.append(f"{app_key}:\n{app_info}")
        
        full_map = "\n\n".join(element_map)
        # print_to_chat(f"App space map:\n{full_map}\n")
        return full_map
    else:
        if "keyboard_shortcuts" in app_space_data:
            shortcuts_map = []
            for app_key, shortcut_data in app_space_data["keyboard_shortcuts"].items():
                shortcuts_map.append(f"{app_key}:\n{convert_to_string(shortcut_data)}")
            
            full_shortcuts = "\n\n".join(shortcuts_map)
            # print_to_chat(f"App space map:\n{full_shortcuts}\n")
            return full_shortcuts

        return ""

def is_field_input_area_active():
    """Check if a text input field is currently active."""
    active_window_title = gw.getActiveWindow().title
    try:
        app = Application().connect(title=active_window_title)
        window = app[active_window_title]
        for child in window.children():
            if 'Edit' in child.class_name() or 'RichEdit' in child.class_name():
                if child.has_keyboard_focus():
                    return True
        return False
    except Exception as e:
        print_to_chat(f"Error checking input area: {e}")
        return False

def get_installed_apps_ms_store():
    """Gets a list of installed Microsoft Store apps."""
    try:
        # Modified PowerShell command to ensure proper Unicode output
        powershell_command = """
        [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
        Get-StartApps | ForEach-Object { $_.Name }
        """
        
        # Set up startupinfo to hide the console window
        startupinfo = None
        if os.name == 'nt':  # Windows
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0  # SW_HIDE
            
        # Run the command with hidden console and UTF-8 encoding
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", powershell_command], 
            capture_output=True,
            encoding='utf-8',
            errors='replace',
            startupinfo=startupinfo,
            creationflags=CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        
        # Get the app names, filter out empty lines, and wrap each name in single quotes
        app_names = [f"'{line.strip()}'" for line in result.stdout.strip().split('\n') if line.strip()]
        
        return ", ".join(app_names)
    except Exception as e:
        print_to_chat(f"Error getting installed apps from Microsoft Store: {e}")
        return ""

def get_application_title(goal="", last_step=None, actual_step=None, focus_window=False):
    """Get the most appropriate application title for the given goal."""
    if actual_step:
        print_to_chat(f"Getting the application name from the actual step: {actual_step}")
        
    installed_apps_registry = get_installed_apps_registry()
    installed_apps_ms_store = get_installed_apps_ms_store()
        
    goal_app = [{
        "role": "system",
        "content":  f"You are an AI Assistant called App Selector that receives a list of programs and responds only with the most suitable program to achieve the goal.\n"
                    f"Only respond with the window name or the program name without the ending extension.\n"
                    f"If no suitable application is found in the provided lists, respond with 'NO_APP'.\n"
                    f"Opened programs:\n{last_programs_list(focus_last_window=focus_window)}\n"
                    f"All installed programs (Registry):\n{installed_apps_registry}\n"
                    f"All installed programs (Microsoft Store):\n{installed_apps_ms_store}"
    }, {
        "role": "user",
        "content":  f"Goal: {goal}\n"
    }]

    app_name = api_call(goal_app, max_tokens=100)
    
    if "NO_APP" in app_name:
        print_to_chat("No suitable application found.")
        return None
    
    filtered_matches = re.findall(r'["\'](.*?)["\']', app_name)
    if filtered_matches and filtered_matches[0]:
        app_name = filtered_matches[0]
    
    if app_name.lower().endswith('.exe'):
        app_name = app_name[:-4]

    print_to_chat(f"Selected Application: {app_name}")

    if "command prompt" in app_name.lower():
        app_name = "cmd"
    elif "calculator" in app_name.lower():
        app_name = "calc"
    elif "sorry" in app_name.lower():
        print_to_chat("Unable to determine application.")
        return None
        
    return app_name

def get_focused_window_details():
    """Get details about the currently focused window."""
    try:
        window_handle = win32gui.GetForegroundWindow()
        window_title = win32gui.GetWindowText(window_handle)
        _, window_pid = win32process.GetWindowThreadProcessId(window_handle)
        process = psutil.Process(window_pid)
        process_name = process.name()
        rect = win32gui.GetWindowRect(window_handle)
        window_position = (rect[0], rect[1])
        window_size = (rect[2] - rect[0], rect[3] - rect[1])
        return window_title, window_handle, window_pid, process_name, window_position, window_size
    except Exception as e:
        print_to_chat(f"ERROR: {e}")
        return None

def get_available_rpa_tasks():
    """Get list of available RPA tasks with descriptions and actions"""
    tasks = load_tasks()
    task_list = []
    for name, data in tasks.items():
        if isinstance(data, dict):
            desc = data.get('description', 'No description')
            actions = data.get('actions', [])
            
            # Format actions list
            action_details = []
            for i, action in enumerate(actions, 1):
                act_type = action.get('act', '')
                act_detail = action.get('detail', '')
                action_details.append(f"  {i}. {act_type}: {act_detail}")
            
            # Build task info string
            task_info = [
                f"Task: {name}",
                f"Description: {desc}",
                "Actions:"
            ]
            task_info.extend(action_details)
            task_list.append("\n".join(task_info))
        else:
            # Handle legacy format
            actions = data
            action_details = []
            for i, action in enumerate(actions, 1):
                act_type = action.get('act', '')
                act_detail = action.get('detail', '')
                action_details.append(f"  {i}. {act_type}: {act_detail}")
                
            task_info = [
                f"Task: {name}",
                "Description: No description",
                "Actions:"
            ]
            task_info.extend(action_details)
            task_list.append("\n".join(task_info))
            
    return "\n\n".join(task_list)

def create_action_context(goal, executed_actions, app_context, keyboard_shortcuts, rpa_context, screen_info, current_cursor_shape, installed_apps_registry, installed_apps_ms_store):
    """Create action context using screenshot information and UI analysis."""
    # Format previous actions to pair assistant messages with their corresponding actions
    previous_actions_formatted = []
    i = 0
    while i < len(executed_actions):
        # Check if current item is a response message
        if i < len(executed_actions) and executed_actions[i].startswith("RESPONSE_MESSAGE:"):
            response_message = executed_actions[i][len("RESPONSE_MESSAGE:"):].strip()
            
            # Check if next item is an action
            if i + 1 < len(executed_actions) and not executed_actions[i + 1].startswith("RESPONSE_MESSAGE:"):
                action_detail = executed_actions[i + 1]
                previous_actions_formatted.append(f"{len(previous_actions_formatted) + 1}. Response Message: {response_message}\nAction: {action_detail}")
                i += 2  # Skip both the message and action
            else:
                # If there's no corresponding action, just add the message
                previous_actions_formatted.append(f"{len(previous_actions_formatted) + 1}. Response Message: {response_message}")
                i += 1
        else:
            # If it's an action without a preceding message
            previous_actions_formatted.append(f"{len(previous_actions_formatted) + 1}. Action: {executed_actions[i]}")
            i += 1
    
    previous_actions = "\n".join(previous_actions_formatted) if previous_actions_formatted else ""
    
    # Get cursor information
    cursor_x, cursor_y = pyautogui.position()
    cursor_info = f"Current Cursor Position: x={cursor_x}, y={cursor_y}\nCurrent Cursor Shape: {current_cursor_shape}"
    # input_field_status = f"Input Field Status: The input field is {'Active' if is_field_input_area_active() else 'Inactive'}"

    # Get focused window details
    focused_details = get_focused_window_details()
    if focused_details:
        window_title, _, _, process_name, window_pos, window_size = focused_details
        focused_window_info = (
            f"Focused window details:\n"
            f"Title: {window_title}\n"
            f"Process: {process_name}\n"
            f"Position: {window_pos}\n"
            f"Size: {window_size}"
        )
        ui_analysis = analyze_app(window_title)
        ui_elements = f"UIAutomation UI Elements Analysis:\n{ui_analysis}" if ui_analysis else ""
    else:
        focused_window_info = "Focused Window Details: There are no details about the focused window."

    return (
        f"You are an AI Agent called Computer Autopilot that is capable of operating freely on Windows by generating actions in sequence to accomplish the user's goal."
        f"Below is a goal that the user wants to achieve, a screenshot of the user's current Windows screen along with the previous actions you've performed. Based on these:\n"
        f"1. Determine if the goal has been achieved based on the current status being shown on the screen.\n"
        f"2. If the goal is not achieved (TASK_COMPLETED: No):\n"
        f"a. Generate a friendly response message telling the user what you're going to do in the same language that the user is using in the goal. In the response message, you can provide the UI elements state or the results of your analysis if needed."
        f" Remember that you're the one who performed all the previous actions, not the user themselves so use only the pronoun I, avoid using the pronoun you and try to respond as if you did all the previous actions using your previous response messages mentioned below along with the previous actions you've performed as contexts."
        f" In the response message:\n"
        f"- If you want the user to provide additional details related to the action or if the action requires the user to do something manually by themselves, respond with PAUSE:<reasons>\n"
        f"- If the task cannot be completed for some reasons or if you're not sure what to do next, respond with STOP:<reasons>. The sign to indicate that a task cannot be completed is you see the same action is performed too many times without achieving the goal.\n"
        f"b. If you choose to continue and not to pause or stop, provide only ONE next action to continue achieving the goal:\n"
        f"- For any action, provide an action description explaining the exact details related to that action.\n"
        f"- For any mouse action, provide the coordinates at the center of the element to interact with in x and y based on the screenshot, the screen resolution and the additional contexts. For other actions, don't include the coordinates in the JSON.\n"
        f"- Specify the number of repeats needed for action that requires multiple repeats. If not specified, the action will be performed only once.\n\n"
        f"Respond in the following format:\n"
        f"TASK_COMPLETED: <Yes/No>\n"
        f"RESPONSE_MESSAGE: <A friendly response message related to what you're doing>\n"
        f"NEXT_ACTION: <If not completed/paused/stopped, provide a JSON with only ONE next action>\n\n"
        f"JSON format for next action:\n"
        f"{{\n"
        f"    \"action\": [\n"
        f"        {{\n"
        f"            \"act\": \"<action_type>\",\n"
        f"            \"detail\": \"<action_description>\",\n"
        f"            \"coordinates\": \"x=<x_value>, y=<y_value>\" (\"x=<x_value>, y=<y_value> to x=<x_value>, y=<y_value>\" for drag action),\n"
        f"            \"repeat\": <number_of_repeats>\n"
        f"        }}\n"
        f"    ]\n"
        f"}}\n\n"
        f"Here are all available action types along with their corresponding action descriptions to provide:\n"
        f"- move_to: The element or position to move the mouse cursor to.\n"
        f"- click: The element or position to click on.\n"
        f"- double_click: The element or position to double click on.\n"
        f"- right_click: The element or position to right click on.\n"
        f"- drag: The starting position to click and drag from, and the ending position to release at (Coordinates required for both positions).\n"
        f"- press_key: The key or the combination of keys to press. (Example: \"Ctrl + T\").\n"
        f"- hold_key_and_click: The key to hold and the position to click on while holding the key.\n"
        f"- text_entry: The specific text to type or write. It can be a word, a sentence, a paragraph or an entire essay. Always use \'\\n\' to indicate a new line when writing a multi-line text. Avoid jump lines! (Example: \"Hello!\\nMy name is John.\").\n"
        f"- scroll: The direction to scroll <up, down, left, right>. (Each scroll action will scroll the screen for 850 pixels in the specified direction).\n"
        f"- open_app: The name of the application to open or focus on.\n"
        f"- time_sleep: The duration to wait for in seconds.\n"
        f"- execute_rpa_task: The name of the RPA task to execute. (Use this action to execute a saved RPA task).\n"
        f"{rpa_context}\n\n"
        f"Important Rules:\n"
        f"1. In the action description, provide ONLY the exact information related to each action type specified above without any additional text.\n"
        f"2. Generate action based primarily on the current status of the task completion progress being shown within the screenshot and only relate to the previous actions for additional contexts."
        f" Pay close attention to the cursor position and shape. Ignore the subtitle if it's currently showing up on the screen.\n"
        f"3. If you see that the last action didn't perform correctly based on the current status being shown on the screen, you can try again using a better alternative action that you think to be more effective.\n"
        f"4. If the last action is a mouse action and you see that it didn't perform correctly, you can also try again with the same mouse action but try to modify the previous coordinates to improve the accuracy.\n"
        f"5. If the goal requires interacting with an application, always provide an open_app action to open and focus on that application before performing any other actions on that application.\n"
        f"6. Before providing any text_entry action on an input area, make sure a click action or a press_key action that leads to focus on the required input area is performed beforehand.\n"
        f"7. Always prioritize using a press_key action if it can replace a mouse action that results in the same outcome.\n"
        f"8. Prioritize generating execute_rpa_task action to achieve the goal more efficiently if available.\n\n"
        f"Here is the goal the user wants to achieve: {goal}\n"
        f"Previous actions you've performed:{f'\n{previous_actions}' if previous_actions else ' There are no previous actions performed.'}\n\n"
        f"Additional contexts:\n"
        f"{screen_info}\n\n"
        f"{focused_window_info}\n\n"
        f"{f'{ui_elements}\n\n' if ui_elements else ''}"
        f"{cursor_info}\n\n"
        f"Here are lists of all programs on the user's Windows:\n"
        f"All currently opened programs:\n{last_programs_list}\n\n"
        f"All installed programs (Registry):\n{installed_apps_registry}\n\n"
        f"All installed programs (Microsoft Store):\n{installed_apps_ms_store}\n\n"
        f"Additional guides:\n{app_context}\n\nKeyboard Shortcuts:\n{keyboard_shortcuts}"
    )

def parse_assistant_result(result):
    """Parse assistant result with better error handling."""
    lines = result.strip().split('\n')
    task_completed = False
    next_action = None
    response_message = None
    
    try:
        # First, check if the last action was a desktop-focusing command
        is_desktop_focus = False
        try:
            remaining_text = result[result.index('NEXT_ACTION:') + len('NEXT_ACTION:'):].strip()
            start_idx = remaining_text.find('{')
            end_idx = remaining_text.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                action_json = remaining_text[start_idx:end_idx]
                action_data = json.loads(action_json)
                action = action_data['action'][0]
                if (action['act'] == 'press_key' and 
                    ('windows + d' in action['detail'].lower() or 
                     'win + d' in action['detail'].lower())):
                    is_desktop_focus = True
        except:
            pass
            
        for line in lines:
            if line.startswith('TASK_COMPLETED:'):
                task_completed = 'yes' in line.lower()
            elif line.startswith('RESPONSE_MESSAGE:'):
                response_message = line[len('RESPONSE_MESSAGE:'):].strip()
                if response_message.startswith('PAUSE:'):
                    from utils import get_app_instance
                    app = get_app_instance()
                    if app:
                        app.paused_execution = True
                    request_stop()
                elif response_message.startswith('STOP:'):
                    request_stop()
            elif line.startswith('NEXT_ACTION:'):
                remaining_text = result[result.index('NEXT_ACTION:') + len('NEXT_ACTION:'):].strip()
                start_idx = remaining_text.find('{')
                end_idx = remaining_text.rfind('}') + 1
                if start_idx != -1 and end_idx != -1:
                    next_action = remaining_text[start_idx:end_idx]
                break
                
        if response_message:
            print_to_chat(response_message)
            # Pass the is_desktop_focus flag to the speaker function
            speaker(response_message.replace('PAUSE:', '').replace('STOP:', '').strip(), 
                   skip_focus=is_desktop_focus)
            
    except Exception as e:
        print_to_chat(f"Error parsing assistant result: {e}")
        
    return task_completed, next_action

def assistant(goal="", executed_actions=None, additional_context=None, resumed=False, called_from=None):
    """Main assistant function for processing and executing user goals."""
    clear_stop()
    
    if not goal:
        speaker("ERROR: No prompt provided. Please provide a prompt to the assistant.")
        time.sleep(10)
        raise ValueError("ERROR: No step provided.")
    else:
        original_goal = goal
        if additional_context:
            original_goal = f"{original_goal}. {additional_context}"
        print_to_chat(f"Prompt: {original_goal}")

        if called_from == "assistant":
            print_to_chat(f"Called from: {called_from}")
        elif not resumed:
            speaker(f"I'm analyzing your request:", additional_text=f"\"{original_goal}\"")
        else:
            speaker("Resuming task execution.")

    app_context = app_space_map(map='app_space')
    keyboard_shortcuts = app_space_map()

    # Add available RPA tasks to context
    rpa_tasks = get_available_rpa_tasks()
    rpa_context = f"Available RPA Tasks:\n{rpa_tasks}" if rpa_tasks else "Available RPA Tasks: There are currently no available RPA tasks."

    # Get screen resolution
    screen_width, screen_height = pyautogui.size()
    screen_info = f"Screen Resolution: {screen_width}x{screen_height}"

    # Get installed applications
    installed_apps_registry = get_installed_apps_registry()
    installed_apps_ms_store = get_installed_apps_ms_store()
    
    # Get settings with defaults
    settings = get_settings()
    max_attempts = settings.get("max_attempts", 20)
    action_delay = settings.get("action_delay", 1.5)

    attempt = 0
    executed_actions = executed_actions or []
    
    while attempt < max_attempts:
        # Check for stop request before generating next action
        if is_stop_requested():
            # Check if it's a pause request
            from utils import get_app_instance
            app = get_app_instance()
            if app and app.paused_execution:
                app.paused_actions = executed_actions
                return "Task paused: Waiting for resume"
            return "Task incomplete: Execution stopped by user"
            
        current_cursor_shape = get_cursor_shape()
        
        action_context = create_action_context(
            original_goal,
            executed_actions,
            app_context,
            keyboard_shortcuts,
            rpa_context,
            screen_info,
            current_cursor_shape,
            installed_apps_registry,
            installed_apps_ms_store
        )
        
        action_key = f"{original_goal}_{attempt}"
        if action_key not in action_cache:
            result = imaging(
                additional_context=action_context,
                screenshot_size='Full screen',
                current_cursor_shape=current_cursor_shape
            )['choices'][0]['message']['content']
            action_cache[action_key] = result
        else:
            result = action_cache[action_key]
            
        task_completed, next_action = parse_assistant_result(result)
        
        # Store the response message in executed_actions
        try:
            lines = result.strip().split('\n')
            for line in lines:
                if line.startswith('RESPONSE_MESSAGE:'):
                    executed_actions.append(line.strip())
                    break
        except Exception as e:
            print_to_chat(f"Error storing response message: {e}")
        
        if task_completed:
            return "Task completed! Can I help you with something else?"
        
        print_to_chat(f"Next action: {next_action}")

        if next_action:
            success = execute_action(next_action)
            
            if not success:
                print_to_chat("Action execution failed!")
                speaker("Action execution failed")
                try:
                    action_data = json.loads(next_action)
                    action = action_data['action'][0]
                    executed_actions.append(f"FAILED - {action['act']}: {action['detail']}")
                except Exception as e:
                    executed_actions.append(f"FAILED - {str(next_action)}")
                return "Task incomplete: Action execution failed"
            
            try:
                action_data = json.loads(next_action)
                action = action_data['action'][0]
                repeat_str = f", repeat: {action.get('repeat', 1)}" if action.get('repeat', 1) and action.get('repeat', 1) > 1 else ", repeat: 1"
                executed_actions.append(f"{action['act']}: {action['detail']} at {action.get('coordinates', 'N/A')}{repeat_str}")
            except Exception as e:
                print_to_chat(f"Error parsing action JSON: {str(e)}")
                executed_actions.append(str(next_action))
            
            time.sleep(action_delay)
            
        attempt += 1
    
    return "Task incomplete! Task execution aborted!"

def identify_element_coordinates(element_description):
    """Use LLM to identify element coordinates from description and current screen."""
    try:
        # Get screen resolution
        screen_width, screen_height = pyautogui.size()
        
        # Prepare prompt for element identification
        prompt = f"""
        You are an AI assistant that helps identify UI elements on screen. 
        I need to find the coordinates of a specific element.

        Screen Resolution: {screen_width}x{screen_height}
        Element description: {element_description}

        Look at the current screen and identify the exact pixel coordinates (x, y) of the center of the described element.

        IMPORTANT: You must return ONLY the coordinates in the exact format: x=123, y=456
        Do not include any other text, explanations, or formatting.

        If the element is not found or not clearly visible, return: NOT_FOUND
        """
        
        # Call the assistant to analyze the screenshot and find the element
        response = imaging(additional_context=prompt, screenshot_size='Full screen')['choices'][0]['message']['content']
        
        # Parse the response to extract coordinates
        if "NOT_FOUND" in response:
            return None
            
        # Extract coordinates from response
        import re
        coord_match = re.search(r'x=(\d+),\s*y=(\d+)', response)
        if coord_match:
            x = int(coord_match.group(1))
            y = int(coord_match.group(2))
            return (x, y)
        else:
            print_to_chat(f"Could not parse coordinates from response: {response}")
            return None
            
    except Exception as e:
        print_to_chat(f"Error identifying element coordinates: {e}")
        return None

def scroll_until_element_visible(element_description, scroll_direction="down", max_attempts=10):
    """Scroll until the specified element becomes visible."""
    try:
        for attempt in range(max_attempts):
            print_to_chat(f"Searching for element (attempt {attempt + 1}/{max_attempts}): {element_description}")
            
            # Try to find the element
            coords = identify_element_coordinates(element_description)
            if coords:
                print_to_chat(f"Element found at coordinates: {coords}")
                return coords
            
            # Element not found, scroll and try again
            print_to_chat(f"Element not found, scrolling {scroll_direction}...")
            scroll(scroll_direction)
            time.sleep(1)  # Wait for scroll to complete
        
        print_to_chat(f"Element '{element_description}' not found after {max_attempts} scroll attempts")
        return None
        
    except Exception as e:
        print_to_chat(f"Error in scroll until element visible: {e}")
        return None

def handle_assistant_task(task_description, repeat=1):
    """Handle assistant task action by calling the main assistant function."""
    try:
        for i in range(repeat):
            print_to_chat(f"Executing assistant task (attempt {i+1}/{repeat}): {task_description}")
            
            # Call the main assistant function with the task description
            assistant(goal=task_description, called_from="rpa_action")
            
            if repeat > 1 and i < repeat - 1:
                time.sleep(1)  # Small delay between repetitions
                
        return True
    except Exception as e:
        print_to_chat(f"Error executing assistant task: {e}")
        return False

def execute_action(action_json):
    """Execute action using coordinates from the action JSON."""
    try:
        if isinstance(action_json, str):
            action_json = action_json.replace('```json', '').replace('```', '').strip()
            start_idx = action_json.find('{')
            end_idx = action_json.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                action_json = action_json[start_idx:end_idx]
        
        instructions = json.loads(action_json)
        action = instructions.get('action', [{}])[0]
        
        if 'act' not in action or 'detail' not in action:
            raise ValueError("Invalid action format: missing 'act' or 'detail'")
        
        # Handle null/None repeat values
        try:
            repeat = max(1, int(action.get('repeat', 1)))
        except (ValueError, TypeError):
            repeat = 1
        
        # Handle coordinates or element identification
        x = y = None
        
        # Check if this is an element-based action
        if action.get('method') == 'element' and 'element_description' in action:
            # Use LLM to identify element coordinates
            coords = identify_element_coordinates(action['element_description'])
            if coords:
                x, y = coords
                print_to_chat(f"Element found at: x={x}, y={y}")
            else:
                print_to_chat(f"Could not find element: {action['element_description']}")
                return False
        elif action.get('method') == 'elements' and action['act'] == 'drag':
            # Handle element-based drag
            start_coords = identify_element_coordinates(action['start_element_description'])
            end_coords = identify_element_coordinates(action['end_element_description'])
            
            if start_coords and end_coords:
                x, y = start_coords
                action['end_pos'] = end_coords
                print_to_chat(f"Start element found at: x={x}, y={y}")
                print_to_chat(f"End element found at: x={end_coords[0]}, y={end_coords[1]}")
            else:
                if not start_coords:
                    print_to_chat(f"Could not find start element: {action['start_element_description']}")
                if not end_coords:
                    print_to_chat(f"Could not find end element: {action['end_element_description']}")
                return False
        else:
            # Handle traditional coordinate-based actions
            coordinates_str = action.get('coordinates', '')
            
            if coordinates_str and action['act'] in {"move_to", "click", "double_click", "right_click", "hold_key_and_click", "drag"}:
                try:
                    # For drag action, parse both start and end coordinates
                    if action['act'] == "drag" and " to " in coordinates_str:
                        start_coords, end_coords = coordinates_str.split(" to ")
                        start_x, start_y = map(float, re.findall(r'x=(\d+\.?\d*), y=(\d+\.?\d*)', start_coords)[0])
                        end_x, end_y = map(float, re.findall(r'x=(\d+\.?\d*), y=(\d+\.?\d*)', end_coords)[0])
                        x, y = start_x, start_y  # Store start position in x,y
                        action['end_pos'] = (end_x, end_y)  # Store end position
                    else:
                        coordinates = {k.strip(): float(v.strip()) for k, v in 
                                   (item.split('=') for item in coordinates_str.split(','))}
                        x, y = coordinates['x'], coordinates['y']
                except Exception as e:
                    print_to_chat(f"Error parsing coordinates: {e}")
                    return False

        def repeat_action(func):
            """Helper function to repeat an action multiple times."""
            try:
                for _ in range(repeat):
                    if not func():
                        return False
                    if repeat > 1:
                        time.sleep(0.5)
                return True
            except Exception as e:
                print_to_chat(f"Error in repeat_action: {str(e)}")
                return False

        def handle_text_entry():
            text_to_write = action['detail']
            def write_once():
                try:
                    original_clipboard = pyperclip.paste()
                    try:
                        pyperclip.copy(text_to_write)
                        pyautogui.hotkey('ctrl', 'v')
                    finally:
                        # Restore original clipboard content
                        pyperclip.copy(original_clipboard)
                    return True
                except Exception as e:
                    print_to_chat(f"Error writing text: {str(e)}")
                    return False

            return repeat_action(write_once)
        
        def handle_open_app():
            app_title = get_application_title(action['detail'])
            if app_title is None:
                print_to_chat(f"Could not find application: {action['detail']}")
                return False
            return activate_window_title(app_title)

        def handle_scroll_action():
            """Handle scroll action - either direction-based or element-based."""
            if action.get('scroll_method') == 'element':
                # Element-based scrolling
                element_desc = action.get('element_description', '')
                scroll_dir = action.get('scroll_direction', 'down')
                coords = scroll_until_element_visible(element_desc, scroll_dir)
                return coords is not None
            else:
                # Traditional direction-based scrolling
                return repeat_action(lambda: scroll(action['detail']))

        def perform_drag_action():
            """Perform drag action from start to end coordinates."""
            try:
                # Move to start position
                move_mouse(x, y)
                time.sleep(0.2)  # Small delay before clicking
                
                # Press and hold mouse button
                pyautogui.mouseDown()
                time.sleep(0.2)  # Small delay before dragging
                
                # Drag to end position
                end_x, end_y = action['end_pos']
                move_mouse(end_x, end_y, duration=1.0)  # Slower movement for drag
                time.sleep(0.2)  # Small delay before release
                
                # Release mouse button
                pyautogui.mouseUp()
                return True
            except Exception as e:
                print_to_chat(f"Error performing drag action: {e}")
                pyautogui.mouseUp()  # Ensure mouse is released
                return False

        action_map = {
            "move_to": lambda: move_mouse(x, y) if x is not None and y is not None else False,
            "click": lambda: perform_mouse_action(x, y, "single", repeat) if x is not None and y is not None else False,
            "double_click": lambda: perform_mouse_action(x, y, "double", repeat) if x is not None and y is not None else False,
            "right_click": lambda: perform_mouse_action(x, y, "right", repeat) if x is not None and y is not None else False,
            "drag": lambda: repeat_action(perform_drag_action) if x is not None and y is not None and 'end_pos' in action else False,
            "press_key": lambda: repeat_action(lambda: perform_simulated_keypress(action['detail'])),
            "hold_key_and_click": lambda: perform_mouse_action(x, y, "hold", repeat, hold_key=action['detail'].split(" and click ")[0]) if x is not None and y is not None else False,
            "text_entry": handle_text_entry,
            "scroll": lambda: handle_scroll_action(),
            "open_app": handle_open_app,
            "time_sleep": lambda: repeat_action(lambda: time.sleep(float(action['detail']))) if action['detail'].isdigit() else 1,
            "execute_rpa_task": lambda: execute_rpa_task(action['detail'], repeat),
            "assistant_task": lambda: handle_assistant_task(action['detail'], repeat)
        }

        if action['act'] in action_map:
            success = action_map[action['act']]()
            # Only check coordinates for non-text_entry actions
            if success is False:
                if coordinates_str and action['act'] in {"move_to", "click", "double_click", "right_click", "hold_key_and_click", "drag"}:
                    print_to_chat("Failed to execute action with provided coordinates")
                    return False
                elif action['act'] == "open_app":
                    return False
            return True
        else:
            print_to_chat(f"WARNING: Unrecognized action '{action['act']}'.")
            return False
        
    except Exception as e:
        print_to_chat(f"Error executing action: {str(e)}")
        return False

def move_mouse(x, y, duration=0.5, ease_function=pyautogui.easeOutQuad):
    """Move mouse to specific coordinates with customizable duration and easing."""
    pyautogui.moveTo(x, y, duration, ease_function)

def click_mouse(click_type="single", repeat=1, interval=0.1):
    """Perform mouse clicks with specified type and repeat count."""
    current_x, current_y = pyautogui.position()
    
    for _ in range(repeat):
        if click_type == "single":
            pyautogui.click(current_x, current_y)
        elif click_type == "double":
            pyautogui.doubleClick(current_x, current_y)
        elif click_type == "right":
            pyautogui.rightClick(current_x, current_y)
            
        if repeat > 1:
            time.sleep(interval)

def perform_mouse_action(x, y, action_type="single", repeat=1, interval=0.1, hold_key=None):
    """Combined function to move mouse and perform click actions."""
    move_mouse(x, y)
    
    if hold_key and action_type == "hold":
        # Map the key if needed
        key_mapping = {
            'win': 'winleft',
            'windows': 'winleft',
            'escape': 'esc',
            'space bar': 'space',
            'spacebar': 'space',
        }
        key = key_mapping.get(hold_key.lower(), hold_key.lower())
        
        # Hold the key, perform click, then release
        try:
            pyautogui.keyDown(key)
            time.sleep(0.1)  # Small delay to ensure key is registered
            click_mouse("single", repeat, interval)
            time.sleep(0.1)  # Small delay before releasing
        finally:
            pyautogui.keyUp(key)
    else:
        click_mouse(action_type, repeat, interval)

def perform_simulated_keypress(press_key):
    """Simulate keyboard key press."""
    try:
        # Remove the 'Press ' prefix
        press_key = press_key.replace('Press ', '').strip()
        
        # Extract the key combination before any descriptive text after "to"
        if " to " in press_key:
            press_key = press_key.split(" to ")[0].strip()
            
        # Define common key mappings
        key_mapping = {
            'win': 'winleft',
            'windows': 'winleft',
            'escape': 'esc',
            'space bar': 'space',
            'spacebar': 'space',
            'arrowup': 'up',
            'arrowdown': 'down',
            'arrowleft': 'left',
            'arrowright': 'right',
        }
        
        # Split the key combination
        keys = [k.strip().lower() for k in press_key.split('+')]
        
        # Map the keys to their pyautogui equivalents
        pyautogui_keys = [key_mapping.get(k.strip(), k.strip()) for k in keys]
        
        # Press all keys in sequence
        for key in pyautogui_keys:
            pyautogui.keyDown(key)
        
        time.sleep(0.2)  # Increased delay to ensure keys are registered
        
        # Release keys in reverse order
        for key in reversed(pyautogui_keys):
            pyautogui.keyUp(key)
            
        return True
        
    except Exception as e:
        print_to_chat(f"Error performing key press: {str(e)}")
        return False

def scroll(target):
    # Parse scroll direction from target element description
    scroll_amount = 850  # Base scroll amount
    
    if any(term in target.lower() for term in ['up', 'top']):
        scroll_direction = 1  # Positive for up
    elif any(term in target.lower() for term in ['down', 'bottom']):
        scroll_direction = -1  # Negative for down
    elif any(term in target.lower() for term in ['right']):
        scroll_direction = -1  # Horizontal scroll right
        pyautogui.hscroll = True
    elif any(term in target.lower() for term in ['left']):
        scroll_direction = 1  # Horizontal scroll left
        pyautogui.hscroll = True
    else:
        scroll_direction = -1  # Default to scroll down
            
    # Perform scroll action
    if hasattr(pyautogui, 'hscroll') and pyautogui.hscroll:
        pyautogui.hscroll(scroll_amount * scroll_direction)
    else:
        pyautogui.scroll(scroll_amount * scroll_direction)

    time.sleep(0.3)  # Wait for scroll to complete

def execute_rpa_task(task_name, repeat=1):
    """Execute a saved RPA task by finding task names in the action description"""
    try:
        # Load all available tasks
        tasks = load_tasks()
        
        # Look for any task names in the action description that match our task keys
        found_tasks = []
        for task_key in tasks.keys():
            if task_key in task_name:
                found_tasks.append(task_key)
                
        if not found_tasks:
            print_to_chat(f"No known task names found in: {task_name}")
            return False
            
        # Execute each found task
        for task_key in found_tasks:
            print_to_chat(f"Executing task: {task_key}")
            task_data = tasks[task_key]
            
            if isinstance(task_data, dict):
                actions = task_data.get('actions', [])
            else:
                # Handle legacy format
                actions = task_data
                
            if not actions:
                print_to_chat(f"No actions found for task: {task_key}")
                continue
                
            for _ in range(repeat):
                for action in actions:
                    if is_stop_requested():
                        return False
                        
                    success = execute_action(json.dumps({"action": [action]}))
                    if not success:
                        print_to_chat(f"Failed to execute action in task {task_key}")
                        return False
                        
                    # Get delay from settings
                    settings = get_settings()
                    action_delay = settings.get("action_delay", 1.5)
                    time.sleep(action_delay)
                    
        return True
        
    except Exception as e:
        print_to_chat(f"Error executing RPA task: {str(e)}")
        return False

def fast_act(single_step, app_name="", original_goal="", action_type="single", repeat=1):
    """Optimized fast action execution using only screenshot context."""
    if not app_name:
        app_name = activate_window_title(focus_topmost_window())
    else:
        app_name = activate_window_title(app_name)

    # Create minimal context for coordinate finding
    coordinate_context = (
        f"You are an AI Windows Mouse Agent. Based on the screenshot of {app_name}, find the coordinates of: {single_step}\n"
        f"Only respond with the coordinates in this exact format: \"x=<value>, y=<value>\""
    )
    
    # Get coordinates from screenshot
    result = imaging(
        window_title=app_name,
        additional_context=coordinate_context,
        screenshot_size='Full screen'
    )['choices'][0]['message']['content']
    
    try:
        # Parse coordinates
        coordinates = {k.strip(): float(v.strip()) for k, v in 
                      (item.split('=') for item in result.split(','))}
        x, y = coordinates['x'], coordinates['y']
        
        # Perform mouse action
        if action_type == "move":
            move_mouse(x, y)
        else:
            perform_mouse_action(x, y, action_type, repeat)
            
        return result
        
    except Exception as e:
        print_to_chat(f"Error performing fast action: {e}")
        return None

def write_action(goal=None, press_enter=False, app_name="", original_goal=None, last_step=""):
    # Generate text content
    message_writer_agent = [{
        "role": "system",
        "content":  f"You're an AI Agent called Writer that processes the goal and only returns the final text goal.\n"
                    f"Process the goal with your own response as you are actually writing into a text box. Avoid jump lines."
                    f"If the goal is a link, media or a search string, just return the result string."
    }, {
        "role": "user",
        "content": f"Goal: {goal}"
    }]
    
    message_to_write = api_call(message_writer_agent, max_tokens=1000)
    
    # Handle click actions if needed
    if any(click_term in goal.lower() for click_term in ["click on", "click the", "click"]):
        print_to_chat("Found to click on the goal.")
        if not is_field_input_area_active():
            print_to_chat("A text box is not active. Clicking on the target element.")
            fast_act(goal, app_name=app_name, original_goal=original_goal)
    elif last_step is None or "text_entry" not in last_step:
        print_to_chat(f"Focusing on the text input area: {goal}")
        if not is_field_input_area_active():
            fast_act(goal, app_name=app_name, original_goal=original_goal)

    # Type the text
    try:
        original_clipboard = pyperclip.paste()  # Save current clipboard
        try:
            pyperclip.copy(message_to_write)
            pyautogui.hotkey('ctrl', 'v')
        finally:
            # Restore original clipboard content
            pyperclip.copy(original_clipboard)
    except Exception as e:
        print_to_chat(f"Error writing text: {str(e)}")
    
    # Handle enter key press if needed
    if any(enter_term in goal.lower() for enter_term in ["press enter", "press the enter", "'enter'", '"enter"']) or press_enter:
        print_to_chat("Found to press the enter key in the goal.")
        pyautogui.press('enter')
    else:
        print_to_chat("AI no \"enter\" key press being made.")

def create_web_action_context(executed_actions, browser_info, webpage_info):
    """Create context for web agent actions."""
    # Format previous actions
    previous_actions_formatted = []
    i = 0
    while i < len(executed_actions):
        # Check if current item is a response message
        if i < len(executed_actions) and executed_actions[i].startswith("RESPONSE_MESSAGE:"):
            response_message = executed_actions[i][len("RESPONSE_MESSAGE:"):].strip()
            
            # Check if next item is an action
            if i + 1 < len(executed_actions) and not executed_actions[i + 1].startswith("RESPONSE_MESSAGE:"):
                action_detail = executed_actions[i + 1]
                previous_actions_formatted.append(f"{len(previous_actions_formatted) + 1}. Response Message: {response_message}\nAction: {action_detail}")
                i += 2  # Skip both the message and action
            else:
                # If there's no corresponding action, just add the message
                previous_actions_formatted.append(f"{len(previous_actions_formatted) + 1}. Response Message: {response_message}")
                i += 1
        else:
            # If it's an action without a preceding message
            previous_actions_formatted.append(f"{len(previous_actions_formatted) + 1}. Action: {executed_actions[i]}")
            i += 1
    
    previous_actions = "\n".join(previous_actions_formatted) if previous_actions_formatted else ""
    
    return (
        f"You are an AI Agent that is capable of operating freely on browsers by generating web actions in sequence to accomplish the user's goal."
        f"\nBased on the user's goal and the current webpage state:"
        f"\n1. Determine if the goal has been achieved."
        f"\n2. If the goal is not achieved (TASK_COMPLETED: No):"
        f"\na. Generate a friendly response message telling the user what you're going to do in the same language that the user is using in the goal."
        f"\n- Use only the pronoun 'I' to refer to yourself, and respond as if you performed all previous actions."
        f"\n- If you want the user to provide additional details related to the action or if the action requires the user to do something manually by themselves, respond with PAUSE:<reasons>"
        f"\n- If the task cannot be completed for some reasons or if you're not sure what to do next, respond with STOP:<reasons>. The sign to indicate that a task cannot be completed is you see the same action is performed too many times without achieving the goal.\n"
        f"\nb. If you choose to continue and not to pause or stop, provide only ONE next action to continue achieving the goal."
        f"\n\nRespond in the following format:"
        f"\nTASK_COMPLETED: <Yes/No>"
        f"\nRESPONSE_MESSAGE: <Your response>"
        f"\nNEXT_ACTION: <If not completed/paused/stopped, provide a JSON with only ONE next action>"
        f"\n\nJSON format for next action:"
        f"\n```json"
        f"\n{{"
        f"\n    \"action\": ["
        f"\n        {{"
        f"\n            \"act\": \"<action_type>\","
        f"\n            \"detail\": \"<action_description>\","
        f"\n            \"selector\": \"<CSS/XPath_selector>\","
        f"\n            \"wait\": <seconds_to_wait_after_the_action>"
        f"\n        }}"
        f"\n    ]"
        f"\n}}"
        f"\n```"
        f"\n\nAvailable web action types:"
        f"\n- navigate_to: Navigate to a URL specified in the detail field."
        f"\n- click: Click on an element identified by the selector."
        f"\n- input_text: Type text into an element identified by the selector. The text is provided in the detail field."
        f"\n- select_option: Select an option from a dropdown list. The selector identifies the dropdown, and detail contains the option text."
        f"\n- extract_text: Extract text from an element identified by the selector."
        f"\n- scroll: Scroll the page. Detail should specify 'up', 'down', or a specific value like 'pixel:500'."
        f"\n- open_tab: Open a new browser tab. If detail contains a URL, navigate to that URL in the new tab."
        f"\n- switch_tab: Switch to a different browser tab. Detail should specify tab index or title fragment."
        f"\n- submit_form: Submit a form. The selector should identify the form or submit button."
        f"\n- execute_javascript: Execute a custom JavaScript code. The detail field should contain the JavaScript code to be executed."
        f"\n- wait: Wait for specified seconds. The detail field should contain the number of seconds."
        f"\n- wait_for_element: Wait for an element to be visible/available. The selector identifies the element."
        f"\n\nImportant Rules:"
        f"\n1. Always generate actions based on the current webpage state."
        f"\n2. When specifying selectors in your actions, use the exact CSS selectors from the 'Available page elements' list in the webpage information."
        f"\n3. Choose the most specific and reliable selector for each element (prefer IDs over classes when available)."
        f"\n4. For input_text actions, ensure you're using a selector for an input element or textarea."
        f"\n5. For click actions, ensure you're using a selector for a clickable element like a button, link, or element with a role of 'button'."
        f"\n6. Wait for pages to load after navigation before performing actions."
        f"\n7. Use JavaScript execution only when simpler actions won't work."
        f"\nPrevious Actions:{f'\n{previous_actions}' if previous_actions else ' There are no previous actions performed.'}"
        f"\n\nCurrent Browser Information:"
        f"\n{browser_info}"
        f"\n\nCurrent Webpage Information:"
        f"\n{webpage_info}"
    )

def extract_page_selectors(browser_instance, max_elements=50):
    """Extract robust selector information from a webpage for automation."""
    try:
        # Use JavaScript to extract element details and robust selectors
        script = """
        function getElementInfo() {
            const elements = [];
            const allElements = document.querySelectorAll('button, a, input, select, textarea, [role], [data-testid], [data-cy], [data-qa], .btn, form');
            const maxElements = arguments[0] || 50;
            const elementsToProcess = Array.from(allElements).slice(0, maxElements);
            
            function getXPath(el) {
                if (!el || el.nodeType !== 1) return '';
                if (el.id) return '//*[@id="' + el.id + '"]';
                let parts = [];
                while (el && el.nodeType === 1) {
                    let nb = 0, idx = 0;
                    let siblings = el.parentNode ? el.parentNode.childNodes : [];
                    for (let i = 0; i < siblings.length; i++) {
                        let sib = siblings[i];
                        if (sib.nodeType === 1 && sib.tagName === el.tagName) {
                            nb++;
                            if (sib === el) idx = nb;
                        }
                    }
                    let tagName = el.tagName.toLowerCase();
                    let part = tagName + (nb > 1 ? '[' + idx + ']' : '');
                    parts.unshift(part);
                    el = el.parentNode;
                }
                return '/' + parts.join('/');
            }

            elementsToProcess.forEach((el, index) => {
                const tagName = el.tagName.toLowerCase();
                const id = el.id || '';
                const classes = Array.from(el.classList).filter(Boolean);
                const name = el.getAttribute('name') || '';
                const type = el.getAttribute('type') || '';
                const placeholder = el.getAttribute('placeholder') || '';
                const value = el.getAttribute('value') || '';
                const role = el.getAttribute('role') || '';
                const dataTestId = el.getAttribute('data-testid') || '';
                const dataCy = el.getAttribute('data-cy') || '';
                const dataQa = el.getAttribute('data-qa') || '';
                const ariaLabel = el.getAttribute('aria-label') || '';
                const text = el.textContent ? el.textContent.trim().substring(0, 80) : '';
                // Build selector options
                let selectors = [];
                if (id) selectors.push(`#${id}`);
                if (dataTestId) selectors.push(`[data-testid="${dataTestId}"]`);
                if (dataCy) selectors.push(`[data-cy="${dataCy}"]`);
                if (dataQa) selectors.push(`[data-qa="${dataQa}"]`);
                if (ariaLabel) selectors.push(`[aria-label="${ariaLabel}"]`);
                if (name) selectors.push(`${tagName}[name="${name}"]`);
                if (role) selectors.push(`${tagName}[role="${role}"]`);
                if (classes.length) selectors.push(`${tagName}.${classes.join('.')}`);
                if (placeholder) selectors.push(`${tagName}[placeholder="${placeholder}"]`);
                // XPath with text if text is short and unique enough
                if (text && text.length < 80) selectors.push(`XPath: //${tagName}[contains(text(), "${text.split('"').join('\\"')}")]`);
                // Fallback: XPath by structure
                selectors.push(`XPath: ${getXPath(el)}`);
                // Remove duplicates
                selectors = Array.from(new Set(selectors));
                // Check visibility
                const rect = el.getBoundingClientRect();
                const isVisible = rect.width > 0 && rect.height > 0 &&
                    window.getComputedStyle(el).display !== 'none' &&
                    window.getComputedStyle(el).visibility !== 'hidden';
                // Collect all attributes
                let attributes = {};
                for (let attr of el.attributes) {
                    attributes[attr.name] = attr.value;
                }
                elements.push({
                    tagName,
                    selectors,
                    text,
                    type,
                    placeholder,
                    isVisible,
                    attributes
                });
            });
            return elements;
        }
        return getElementInfo(arguments[0]);
        """
        elements = browser_instance.execute_script(script, max_elements)
        # Format the results for AI prompt
        formatted_elements = []
        for i, el in enumerate(elements):
            element_info = f"{i+1}. {el['tagName']}"
            if el['text']:
                element_info += f" | Text: \"{el['text']}\""
            if el['type']:
                element_info += f" | Type: {el['type']}"
            if el['placeholder']:
                element_info += f" | Placeholder: \"{el['placeholder']}\""
            if not el['isVisible']:
                element_info += " (Not visible)"
            # List selectors in order of reliability
            element_info += "\n    Selectors: " + ", ".join(el['selectors'])
            # Show all attributes for reference
            if el['attributes']:
                attr_str = ", ".join(f"{k}='{v}'" for k, v in el['attributes'].items())
                element_info += f"\n    Attributes: {attr_str}"
            formatted_elements.append(element_info)
        return "\n".join(formatted_elements)
    except Exception as e:
        return f"Error extracting page selectors: {str(e)}"

def execute_web_action(action_json, sb_driver, delay=1.0):
    """Execute a web action using SeleniumBase driver."""
    try:
        action_data = json.loads(action_json)
        action = action_data['action'][0]
        
        act_type = action.get('act', '')
        detail = action.get('detail', '')
        selector = action.get('selector', '')
        wait = float(action.get('wait', delay))        
        
        result = None
        
        # Execute action based on action type using SeleniumBase methods
        if act_type == 'navigate_to':
            sb_driver.open(detail)
            
        elif act_type == 'click':
            sb_driver.click(selector)
            
        elif act_type == 'input_text':
            sb_driver.type(selector, detail)
            
        elif act_type == 'select_option':
            sb_driver.select_option_by_text(selector, detail)
            
        elif act_type == 'wait_for_element':
            sb_driver.wait_for_element_visible(selector)
            
        elif act_type == 'extract_text':
            text = sb_driver.get_text(selector)
            result = text
            
        elif act_type == 'scroll':
            if detail.lower() == 'down':
                sb_driver.execute_script("window.scrollBy(0, 500);")
            elif detail.lower() == 'up':
                sb_driver.execute_script("window.scrollBy(0, -500);")
            elif detail.lower().startswith('pixel:'):
                pixels = int(detail[6:])
                sb_driver.execute_script(f"window.scrollBy(0, {pixels});")
            else:
                # Scroll to element if detail is a selector
                try:
                    sb_driver.scroll_to(detail)
                except:
                    pass
            
        elif act_type == 'open_tab':
            try:
                # If detail contains a URL, open tab with that URL, otherwise just open a new tab
                if detail and detail.startswith(('http://', 'https://')):
                    sb_driver.open_new_tab(detail)
                else:
                    sb_driver.open_new_tab("about:blank")
                
            except Exception as e:
                print_to_web_chat(f"Error opening new tab: {str(e)}")
                return False, None
            
        elif act_type == 'switch_tab':
            try:
                if detail.isdigit():
                    tab_index = int(detail)
                    sb_driver.switch_to_window(tab_index)
                else:
                    # Try to match by title
                    sb_driver.switch_to_window(detail)
            except Exception as e:
                print_to_web_chat(f"Error switching tab: {str(e)}")
                pass
            
        elif act_type == 'submit_form':
            # If selector points to a form
            try:
                sb_driver.submit(selector)
            except:
                # If selector points to a button
                try:
                    sb_driver.click(selector)
                except Exception as e:
                    print_to_web_chat(f"Error submitting form: {str(e)}")
                    pass
        
        elif act_type == 'wait':
            try:
                seconds = float(detail)
                sb_driver.sleep(seconds)
            except Exception as e:
                print_to_web_chat(f"Error in wait action: {str(e)}")
                pass
        
        elif act_type == 'execute_javascript':
            script_result = sb_driver.execute_script(detail)
            if script_result:
                result = str(script_result)
        
        else:
            print_to_web_chat(f"WARNING: Unrecognized action '{action['act']}'.")
            return False, None
        
        # Wait after action execution
        sb_driver.sleep(wait)
        return True, result
        
    except TimeoutException:
        print_to_web_chat("Timeout waiting for element")
        return False, None
    except Exception as e:
        print_to_web_chat(f"Error executing web action: {str(e)}")
        return False, None

def web_assistant(goal="", executed_actions=None, headless=False, use_vision=False):
    """Main web assistant function for processing and executing web agent tasks using SeleniumBase."""
    clear_stop()
    
    if not goal:
        return "ERROR: No prompt provided. Please provide a prompt to the web assistant."
    
    print_to_web_chat(f"Web Task: {goal}")
    
    # Initialize
    if executed_actions is None:
        executed_actions = []
    
    # Get settings with defaults from the same source as main assistant
    settings = get_settings()
    max_attempts = settings.get("web_max_attempts", 20)
    action_delay = settings.get("web_action_delay", 1.5)
    
    # Initialize or reuse browser instance
    global _sb_instance, _driver
    try:
        # Check if the existing driver is still responsive
        if _driver is not None:
            try:
                # A lightweight operation to check if the browser is still there
                _driver.title # Accessing a property like title will fail if the browser is closed
            except Exception as e: # Catches WebDriverException, NoSuchWindowException, etc.
                print_to_web_chat("Browser session lost, re-initializing browser...")
                if _sb_instance:
                    try:
                        _sb_instance.__exit__(None, None, None) # Attempt to clean up the old SB instance
                    except Exception as e_exit:
                        print_to_web_chat(f"Error during old SB instance cleanup: {e_exit}")
                _sb_instance = None
                _driver = None
        
        # Initialize browser if it's not already or if it was just reset
        if _sb_instance is None:
            print_to_web_chat("Initializing new browser session...")
            # Create new SeleniumBase instance
            _sb_instance = SB(browser="chrome", headless=headless, uc=True)
            # Enter the context manager and store the driver
            _driver = _sb_instance.__enter__()
            # Maximize the browser window
            _driver.maximize_window()
        sb = _driver # Use the (potentially new) driver for the current task
        
        attempt = 0
        
        while attempt < max_attempts and not is_stop_requested():
            # Get browser and webpage information using SeleniumBase driver
            try:
                current_url = sb.get_current_url()
                page_title = sb.get_page_title()
                
                # Get basic page structure (using SeleniumBase's driver access)
                page_source = sb.get_page_source()
                
                # Extract important elements (simplified for prompt context)
                soup = BeautifulSoup(page_source, 'html.parser')
                
                # Gather forms, inputs, buttons, links
                forms = len(soup.find_all('form'))
                inputs = len(soup.find_all('input'))
                buttons = len(soup.find_all('button'))
                links = len(soup.find_all('a'))
                
                # Extract visible text (simplified)
                body_text = soup.body.get_text(strip=True) if soup.body else ""
                visible_text = body_text[:1000] + "..." if len(body_text) > 1000 else body_text
                
                # Extract detailed selector information using SeleniumBase driver
                page_selectors = extract_page_selectors(sb.driver) # Pass the underlying driver
                
                # Prepare browser information
                browser_info = f"Headless Mode: {headless}\nCurrent URL: {current_url}\nPage Title: {page_title}"
                
                # Prepare webpage information
                webpage_info = (
                    f"Page has {forms} forms, {inputs} input fields, {buttons} buttons, and {links} links.\n"
                    f"Visible text excerpt:\n{visible_text}\n\n"
                    f"Available page elements (selectors):\n{page_selectors}"
                )
                
            except Exception as e:
                browser_info = f"Headless Mode: {headless}\nError getting page info: {str(e)}"
                webpage_info = "Unable to extract webpage information"
            
            # Create context for the web action
            web_action_context = create_web_action_context(
                executed_actions,
                browser_info,
                webpage_info
            )
            
            # Format messages for api_call
            messages = [
                {"role": "system", "content": web_action_context},
                {"role": "user", "content": goal}
            ]
            
            if use_vision:
                # Use imaging function to get visual analysis with the web_action_context
                print_to_web_chat("Using vision AI to analyze the page...")
                
                # Create a complete context that includes the user's goal
                vision_context = f"{web_action_context}\n\nUser Goal: {goal}"
                
                result = imaging(
                    additional_context=vision_context,
                    screenshot_size='Full screen',
                    current_cursor_shape=get_cursor_shape()
                )['choices'][0]['message']['content']
            else:
                # Use regular API call without visual input
                result = api_call(
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1500
                )
            
            # Parse results
            lines = result.strip().split('\n')
            task_completed = False
            next_action = None
            response_message = None
            
            for line in lines:
                if line.startswith('TASK_COMPLETED:'):
                    task_completed = 'yes' in line.lower()
                elif line.startswith('RESPONSE_MESSAGE:'):
                    response_message = line[len('RESPONSE_MESSAGE:'):].strip()
                    # Store in executed actions
                    executed_actions.append(line.strip())
                    
                    if response_message.startswith('PAUSE:'):
                        from utils import get_app_instance
                        app = get_app_instance()
                        if app:
                            app.paused_execution = True
                        request_stop()
                    elif response_message.startswith('STOP:'):
                        request_stop()
                        
                elif line.startswith('NEXT_ACTION:'):
                    # Find JSON block
                    json_start = result.find('```json', result.index('NEXT_ACTION:'))
                    if json_start != -1:
                        json_end = result.find('```', json_start + 7)
                        if json_end != -1:
                            next_action = result[json_start + 7:json_end].strip()
                    else:
                        # Alternative format without code block
                        remaining_text = result[result.index('NEXT_ACTION:') + len('NEXT_ACTION:'):].strip()
                        start_idx = remaining_text.find('{')
                        end_idx = remaining_text.rfind('}') + 1
                        if start_idx != -1 and end_idx != -1:
                            next_action = remaining_text[start_idx:end_idx]
            
            if response_message:
                # Display and speak the message
                clean_message = response_message.replace('PAUSE:', '').replace('STOP:', '').strip()
                print_to_web_chat(clean_message)
                speaker(clean_message)
            
            if task_completed:
                return "Task completed! Can I help you with something else?"
            
            if next_action and not is_stop_requested():
                # Display the raw JSON action in the chat
                print_to_web_chat(f"```json\n{next_action}\n```", False)
                
                success, result_data = execute_web_action(next_action, sb, action_delay) # Pass sb driver
                
                if not success:
                    print_to_web_chat("Web action execution failed!")
                    speaker("Web action execution failed")
                    executed_actions.append("FAILED - Web action execution")
                else:
                    try:
                        action_data = json.loads(next_action)
                        action = action_data['action'][0]
                        
                        # Handle results from extract_text and execute_javascript actions
                        if action['act'] in ['extract_text', 'execute_javascript'] and result_data:
                            print_to_web_chat(f"Result: {result_data}")
                            executed_actions.append(f"{action['act']}: {action.get('detail', '')} {action.get('selector', '')} -> Result: {result_data}")
                        else:
                            executed_actions.append(f"{action['act']}: {action.get('detail', '')} {action.get('selector', '')}")
                            
                    except Exception as e:
                        print_to_web_chat(f"Error parsing action JSON: {str(e)}")
                        executed_actions.append(str(next_action))
            
            attempt += 1
        
        if is_stop_requested():
            return "Task execution stopped."
        return "Task incomplete! Maximum number of actions reached."
    except Exception as e:
        print_to_web_chat(f"Error in web assistant: {str(e)}")
        return f"Error in web assistant: {str(e)}"
    finally:
        # If there was an error creating the browser instance, ensure it's cleaned up
        if _sb_instance is None:
            close_browser()

def close_browser():
    """Close the browser instance when shutting down the application."""
    global _sb_instance, _driver
    if _sb_instance is not None:
        try:
            if _driver is not None:
                # Don't call quit() as it will be handled by __exit__
                pass
            # Exit the context manager properly
            _sb_instance.__exit__(None, None, None)
        except Exception as e:
            print_to_web_chat(f"Error closing browser: {str(e)}")
        finally:
            _sb_instance = None
            _driver = None

def create_database(database_file):
    """Create SQLite database for storing task instructions."""
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS app_cases (
        id INTEGER PRIMARY KEY,
        app_name TEXT NOT NULL,
        title TEXT NOT NULL,
        instructions TEXT NOT NULL,
        UNIQUE(app_name, title, instructions)
    )
    ''')
    conn.commit()
    conn.close()

def database_add_case(database_file, app_name, goal, instructions):
    """Add a case to the database."""
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()
    try:
        cursor.execute('''
        INSERT INTO app_cases (app_name, title, instructions)
        VALUES (?, ?, ?)
        ''', (app_name, goal, json.dumps(instructions)))
        conn.commit()
    except sqlite3.IntegrityError:
        print_to_chat("AI skipping element insertion to program map database.")
    finally:
        conn.close()

def print_database(database_file):
    """Print all cases in the database."""
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM app_cases')
    rows = cursor.fetchall()
    for row in rows:
        print_to_chat(row)
    conn.close()

# Initialize database
database_file = r'history.db'
create_database(database_file)

# Example Usage
if __name__ == "__main__":
    assistant(goal="Open Reddit, Youtube, TikTok, and Netflix on new windows by using the keyboard on each corner of the screen", app_name="Microsoft Edge")
