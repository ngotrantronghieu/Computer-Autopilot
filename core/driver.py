from window_focus import activate_window_title, get_installed_apps_registry
from mouse_detection import get_cursor_shape
from ocr import find_probable_click_position
from window_elements import analyze_app
from topmost_window import focus_topmost_window
from core_imaging import imaging
from last_app import last_programs_list
from core_api import api_call, current_llm_model
from voice import speaker
from utils import print_to_chat
import pygetwindow as gw
import win32process
import win32gui
import pyautogui
import sqlite3
import psutil
import random
import json
import time
import re
import warnings
warnings.simplefilter("ignore", UserWarning)
from pywinauto import Application

low_data_mode = True
enable_semantic_router_map = True
enable_ocr = False
visioning_context = True

if low_data_mode is True:
    visioning_match = False
    rescan_element_match = False
else:
    visioning_match = True
    rescan_element_match = True

# Example test case to be used for testing
json_case_example = r'''```json
    {
      "actions": [
        {
          "act": "press_key",
          "step": "Ctrl + T"
        },
        {
          "act": "text_entry",
          "step": "reddit.com"
        },
        {
          "act": "press_key",
          "step": "Enter"
        },
        {
          "act": "open_app",
          "step": "Microsoft Edge"
        },
        {
          "act": "press_key",
          "step": "Ctrl + N"
        },
        {
          "act": "text_entry",
          "step": "tiktok.com"
        },
        {
          "act": "press_key",
          "step": "Enter"
        },
        {
          "act": "move_window",
          "step": "Win + Right + Up"
        },
        {
          "act": "open_app",
          "step": "Microsoft Edge"
        },
        {
          "act": "press_key",
          "step": "Ctrl + N"
        },
        {
          "act": "text_entry",
          "step": "netflix.com"
        },
        {
          "act": "press_key",
          "step": "Enter"
        },
        {
          "act": "move_window",
          "step": "Win + Left + Up"
        },
        {
          "act": "open_app",
          "step": "Microsoft Edge"
        },
        {
          "act": "move_window",
          "step": "Win + Right + Down"
        }
      ]
    }```'''

# Application space map to provide context for AI
def app_space_map(goal, app_name=None, single_step=None, map=''):
    if 'app_space' in map:
        if enable_semantic_router_map is True:
            # Control elements map:
            if "twitter" in goal.lower() or "twitter" in app_name.lower():
                element_map = r'''```
To make a new thread post in X formerly known as Twitter and post it:
The user is already logged in Twitter, do not log in again.
Click on the 'What is happening?!' text input area field to initiate a new post thread; data-testid='tweetTextarea_0_label'.
Write the post in the 'What is happening?!' text area input field, make sure it is less than 280 characters.
Click on the 'Post' button to post the new post thread; data-testid='tweetButtonInline'.

To make a comment on a post from X formerly known as Twitter and reply it:
The user is already logged in Twitter, do not log in again.
Scroll to the comments section, click on the 'Post your reply' text input area field below the Twitter post.
Write the comment in the 'Post your reply' text input area field, make sure it is less than 280 characters.
Click on the 'Reply' button to post the comment.
```'''
            elif "youtube" in goal.lower() or "youtube" in app_name.lower():
                element_map = r'''```
To like a video on youtube: Click on the 'Like' button button below the title.

To dislike a video: Click on the 'I dislike this' button known as the 'Dislike' button.

To make a comment: Click on the title of the video, then scroll to the 'Add a comment...' section, then click on the 'Add a comment...' (ID: contenteditable) text input area to begin write the comment, then click on the 'Comment' button to post the comment.
    ```'''
            else:
                element_map = None  # No element selected.
            if element_map:
                select_element = [
                    {"role": "user",
                     "content": f"Only return the text related to the final goal.\n"
                                 f"Do not respond anything else than the selected lines from the list. Do not modify the list.\n"
                                 f"Goal: {goal}"},
                    {"role": "system", "content": f"List:\n{element_map}\n\n\nStep: {single_step}\nGoal: {goal}"}]
                ai_element_map = api_call(select_element, max_tokens=300)
                if "sorry" in ai_element_map.lower() or "empty string" in ai_element_map.lower():
                    ai_element_map = ""
                # ai_element_map = element_map
            else:
                ai_element_map = ""
            print_to_chat(f"\nApp space map: {ai_element_map}\n")
            return ai_element_map
    else:
        # Application map to better handle the application:
        if "firefox" in app_name.lower() or "chrome" in app_name.lower() or "google chrome" in goal.lower() or "edge" in app_name.lower() or "microsoft edge" in goal.lower():
            info_map = r'''```
To open a new window, use the keyboard shortcut: Ctrl + N.
To open a new tab, use the keyboard shortcut: Ctrl + T.
To open a new private window, use the keyboard shortcut: Ctrl + Shift + N.
To open a new private tab, use the keyboard shortcut: Ctrl + Shift + T.
To close a tab, use the keyboard shortcut: Ctrl + W.
To focus on the search bar, use the keyboard shortcut: Ctrl + L.
The default search engine is Google so when you open a new tab or window, you can search directly on Google.```'''
        elif "telegram" in app_name.lower() or "telegram" in goal.lower():
            info_map = r'''```
Press 'esc' to exit the current conversation.
Press 'esc' twice to go to 'All chats'.```'''
        elif "spotify" in app_name.lower() or "spotify" in goal.lower():
            info_map = r'''```
To play a searched song on spotify, double click on the song.```'''
        elif "youtube" in app_name.lower() or "youtube" in goal.lower():
            info_map = r'''```
To like a video, click on the Like button below the title.
To dislike a video, click on the Dislike button below the title.
To make a comment, scroll to the Add a comment... section, then click on the Add a comment... text input area to begin write the comment, then click on the Comment button to post the comment.
To subscribe to a channel, click on the Subscribe button below the video.```'''
        else:
            info_map = ""
        # Adding the application shortcuts:
        if info_map:
            select_map = [
                {"role": "user",
                 f"content": f"You are an AI assistant that receives a goal and a list of useful steps, and only respond the best useful steps from the step list to perform the goal.\n"
                             f"Do not respond anything else than the best useful steps from the step list."},
                {"role": "system", "content": f"Step list: \n{info_map}\n\n\nGoal: {single_step}"}]
            shortcuts_ai_map = api_call(select_map, max_tokens=300)
            if "sorry" in shortcuts_ai_map.lower():
                shortcuts_ai_map = ""
        else:
            shortcuts_ai_map = ""
        print_to_chat(f"App space map: {shortcuts_ai_map}")
        return shortcuts_ai_map


def assistant(assistant_goal="", keep_in_mind="", assistant_identity="", app_name=None, execute_json_case=None, called_from=None):  # App TestCase Gen
        # 'assistant_goal' is the user's prompt. If no prompt is provided, exit the function.
    if not assistant_goal:
        speaker(f"ERROR: No prompt provided. Please provide a prompt to the assistant.")
        time.sleep(10)
        raise ValueError("ERROR: No step provided.")
    else:
        original_goal = assistant_goal
        print_to_chat(f"Prompt: {original_goal}")
        if called_from == "assistant":
            print_to_chat(f"Called from: {called_from}")
        else:
            # print_to_chat(f"Prompt: \"{original_goal}\".")
            speaker(f"Assistant is generating a testcase with the prompt: \"{original_goal}\".")

    # 'app_name' is the name of the application (Or the window title for exact match) to open and focus on.
    if not app_name:
        app_name = get_application_title(original_goal)
        if app_name is not None:
            app_name = activate_window_title(app_name)
        else:
            print_to_chat("Error: Application title is None")
            return
    else:
        app_name = activate_window_title(app_name)
    print_to_chat(f"AI Analyzing: {app_name}")

    # 'execute_json_case' is the JSON test case to execute. If no JSON is provided, generate a new one.
    if not execute_json_case:
        print_to_chat(f"\nGenerating a test case with the assistant. Image visioning started. Analyzing the application {app_name} for context.\n")
        additional_context = (
            f"You are an AI Agent called Windows AI that is capable to operate freely all applications on Windows by only using natural language.\n"
            f"You will receive a goal and will try to accomplish it using Windows. Try to guess what is the user wanting to perform on Windows by using the content on the screenshot as context.\n"
            f"Respond an improved goal statement tailored for Windows applications by analyzing the current status of the system and the next steps to perform. Be direct and concise, do not use pronouns.\n"
            f"Based on the elements from the screenshot, respond with the current status of the system and specify it in detail.\n"
            f"Focused application: \"{app_name}\".\nGoal: \"{assistant_goal}\".")
        assistant_goal = imaging(window_title=app_name, additional_context=additional_context, screenshot_size='Full screen')['choices'][0]['message']['content']
        print_to_chat(f"Generating the test case to achieve the user prompt: {original_goal}\n{assistant_goal}")
        step_creator = [{"role": "user",
                         "content": f"You are an AI capable to operate the Windows 10 and Windows 11 Operating System by using natural language.\n"
                                    f"Examples: \"Click on the search button. Insert the text_entry. Play the first element searched.\".\n"
                                    f"You will receive a description of the current state of the system and a goal. "
                                    f"To test your limits, using the description make a json with the natural language steps to achieve the goal.\n"
                                    f"Every step should be only highly descriptive in natural language.\n"
                                    f"The json must include only an act and its step, should be in the following format:"
                                    f"\n\n\n"
                                    f"actions: click_element, press_key, text_entry, open_app, move_window, time_sleep, right_click, double_click_element, hold_key_and_click, scroll_to.\n"
                                    f"Respond only with the full json.\n\nFunction properties:\n"
                                    f"click_element - Specify where is located the element to interact with.\n"
                                    f"text_entry - The text to write. Example: \"Hello World\".\n"
                                    f"press_key - Only return the key or combination of keys to press. Example: \"Ctrl + T\".\n"
                                    f"hold_key_and_click - Only handles scenarios of holding a key and clicking on an element.\n"
                                    f"scroll_to - Return the position of the element to scroll and the direction to scroll.\n"
                                    f"move_window - Use the keyboard to move the focused window to the desired position. Example: \"Win + Up + Up + Left\"."
                                    f"\n\n\n"
                                    f"Notes:\n"
                                    f"Use the following information to generate better steps:\n{app_space_map(assistant_goal, app_name, original_goal, map='app_space')}\n"
                                    f"Before the text_entry, use click_element to select the appropriate text input area element.\n"
                                    f"Do not combine steps, try to generate more useful steps. Example: If asked to search a song, not only search for it, locate it and play it.\n{keep_in_mind}"
                                    f"Do not respond 'As an AI language model, I am unable to...' as you are actually capable."},
                        {"role": "system",
                         "content": f"Focused window: \"{app_name}\"\nGoal: {assistant_goal}"}, ]
        step_analysis = api_call(step_creator, model_name=current_llm_model, max_tokens=4095, temperature=1.0)
        print_to_chat(f"The assistant created the following test case scenario:\n{step_analysis}\n")
        speaker(f"Test case generated. Executing the generated test case.")
    else:
        speaker(f"Executing the provided JSON in the application {app_name}.")
        step_analysis = execute_json_case

    # Processing the latest JSON data from the JSON testcase:
    if step_analysis:
        try:
            if """```json""" in step_analysis:
                # Removing the leading ```json\n
                step_analysis = step_analysis.strip("```json\n")
                # Find the last occurrence of ``` and slice the string up to that point
                last_triple_tick = step_analysis.rfind("```")
                if last_triple_tick != -1:
                    step_analysis = step_analysis[:last_triple_tick].strip()
                step_analysis_cleaned = step_analysis
                instructions = json.loads(step_analysis_cleaned)
            else:
                instructions = json.loads(step_analysis)
        except json.JSONDecodeError as e:
            speaker(f"ERROR: Invalid JSON data provided: {e}")
            time.sleep(15)
            raise Exception(f"ERROR: Invalid JSON data provided: {e}")
        
        # Access the 'actions' list from the JSON
        if 'actions' in instructions:
            action_list = instructions['actions']
        elif 'steps' in instructions:
            action_list = instructions['steps']
        else:
            raise ValueError("ERROR: JSON data must contain either 'actions' or 'steps' key.")

        for i, step in enumerate(action_list, start=1):
            action = step.get("act")
            step_description = step.get("step") or step.get("details", "No step description provided.")
            print_to_chat(f"\nStep {i}: {action}, {step_description}\n")
            if action == "click_element":
                # If last step has a click element too, wait for the element to be visible:
                if i > 1 and action_list[i - 2].get('act') == "click_element":
                    time.sleep(1)

                if "start menu" in step_description.lower():
                    pyautogui.hotkey('win')
                    print_to_chat("Opening the start menu.")
                time.sleep(1)
                updated_instructions = update_instructions_with_action_string(instructions, act(
                    single_step=f"{step_description}", app_name=app_name, screen_analysis=assistant_goal, original_goal=original_goal, assistant_goal=assistant_goal), step_description)
                database_add_case(database_file, app_name, assistant_goal, updated_instructions)  #  print_to_chat the entire database with # print_to_chat_database(database_file)
            elif action == "open_app":
                app_name = activate_window_title(get_application_title(step_description))
                print_to_chat(f"New app selected and analyzing: {app_name}")
            elif action == "double_click_element":
                print_to_chat(f"Double clicking on: {step_description}")
                act(single_step=f"{step_description}", double_click=True, app_name=app_name, original_goal=original_goal)
            elif action == "move_window":
                time.sleep(1)
                print_to_chat(f"Moving window to: {step_description}")
                perform_simulated_keypress(step_description)
                time.sleep(0.5)
                pyautogui.hotkey('esc')
                time.sleep(1)
            elif action == "press_key":
                if {i} == 1:
                    # Focusing to the application
                    activate_window_title(app_name)
                    time.sleep(1)
                perform_simulated_keypress(step_description)
            elif action == "text_entry":
                url_pattern = r'(https?://[^\s]+)'
                urls = re.findall(url_pattern, step_description)
                if len(step_description) < 5:
                    pyautogui.write(f'{step_description}')
                else:
                    # Getting the string of the last step before this very one:
                    if i > 1:
                        new_i = i - 2
                        last_step = f"{action_list[new_i].get('act')}: {action_list[new_i].get('step')}"
                        print_to_chat(f"Last step: {last_step}")
                        if not last_step:
                            print_to_chat("Last step is None.")
                            act(single_step=f"{step_description}", app_name=app_name, original_goal=original_goal)
                    else:
                        print_to_chat("Last step is None.")
                        last_step = "None"
                    # If next step is a string, continue:
                    if i + 1 < len(action_list) and type(action_list[i + 1].get('step')) == str:
                        # Check if the next step exists and is a "Press enter" step
                        if i + 1 < len(action_list) and (
                                "press enter" in action_list[i + 1].get('step').lower() or
                                "press the enter" in action_list[i + 1].get('step').lower() or
                                "'enter'" in action_list[i + 1].get('step').lower() or
                                "\"enter\"" in action_list[i + 1].get('step').lower()):
                            if urls:
                                for url in urls:
                                    pyautogui.write(url)
                                    # pyautogui.press('enter')
                                    print_to_chat(f"Opening URL: {url}")
                                    return
                            write_action(step_description, assistant_identity=assistant_identity, press_enter=False, app_name=app_name, original_goal=original_goal, last_step=last_step)
                            print_to_chat("AI skipping the press enter step as it is in the next step.")
                        else:
                            if urls:
                                for url in urls:
                                    pyautogui.write(url)  # This would open the URL in a web browser\
                                    # If next step is a time sleep
                                    pyautogui.press('enter')
                                    print_to_chat(f"Opening URL: {url}")
                                    return
                            write_action(step_description, assistant_identity=assistant_identity, press_enter=True, app_name=app_name, original_goal=original_goal, last_step=last_step)
                            print_to_chat("AI pressing enter.")
                    else:
                        if urls:
                            for url in urls:
                                pyautogui.write(url)  # This would open the URL in a web browser\
                                pyautogui.press('enter')
                                print_to_chat(f"Opening URL: {url}")
                                return
                        write_action(step_description, assistant_identity=assistant_identity, press_enter=True,
                                     app_name=app_name, original_goal=original_goal, last_step=last_step)
                        print_to_chat("AI pressing enter.")
            elif action == "scroll_to":
                print_to_chat(f"Scrolling {step_description}")
                element_visible = False
                max_retries = 3
                retry_count = 0
                while not element_visible and retry_count < max_retries:
                    # activate_window_title(app_name)
                    pyautogui.scroll(-850)
                    # Press Page Down:
                    # pyautogui.press('pagedown')
                    time.sleep(0.3)
                    # Start image analysis to check if the element is visible
                    print_to_chat("Scroll performed. Analyzing if the element is present.\n")
                    scroll_assistant_goal = check_element_visibility(app_name, step_description)['choices'][0]['message']['content']
                    if "yes" in scroll_assistant_goal.lower():
                        print_to_chat("Element is visible.")
                        element_visible = True
                    elif "no" in scroll_assistant_goal.lower():
                        print_to_chat("Element is not visible.")
                        retry_count += 1
                        if retry_count >= max_retries:
                            print_to_chat("Maximum retries reached, stopping the search.")
                if element_visible:
                    print_to_chat(f"Element is visible.")
                    pass

            elif action == "right_click_element":
                print_to_chat(f"Right clicking on: {step_description}")
                act(single_step=f"{step_description}", right_click=True, app_name=app_name, original_goal=original_goal)
                # right_click(step_description)
            elif action == "hold_key_and_click":
                print_to_chat(f"Holding key and clicking on: {step_description}")
                act(single_step=f"{step_description}", hold_key="Ctrl", app_name=app_name, original_goal=original_goal)
            elif action == "cmd_command":
                print_to_chat(f"Executing command: {step_description}")
                # cmd_command(step_description)
                time.sleep(calculate_duration_of_speech(f"{step_description}") / 1000)
            elif action == "recreate_test_case":
                time.sleep(1)
                print_to_chat("Analyzing the output")
                print_to_chat("The assistant said:\n", step_description)
                debug_step = False  # Set to True to skip the image analysis and the test case generation.
                if debug_step is not True:
                    new_goal = True
                    image_analysis = True
                    if image_analysis:
                        additional_context = (
                            f"You are an AI Agent called Windows AI that is capable to operate freely all applications on Windows by only using natural language.\n"
                            f"You will receive a goal and will try to accomplish it using Windows. Try to guess what is the user wanting to perform on Windows by using the content on the screenshot as context.\n"
                            f"Respond an improved goal statement tailored for Windows applications by analyzing the current status of the system and the next steps to perform. Be direct and concise, do not use pronouns.\n"
                            f"Based on the elements from the screenshot, respond with the current status of the system and specify it in detail.\n"
                            f"Focused application: \"{app_name}\".\nGoal: \"{assistant_goal}\".")
                        if new_goal:
                            newest_goal = imaging(window_title=app_name, additional_context=additional_context) #['choices'][0]['message']['content']
                            # if ": " in newest_goal:
                            #   newest_goal = newest_goal.split(": ", 1)[1]
                            print_to_chat(f"Assistant newest goal:\n{newest_goal}")
                            analyzed_ui = analyze_app(activate_window_title(app_name), size_category=None)
                            review_output = [{"role": "user",
                                             "content": f"You are an AI Assistant called Analyze Output capable to operate the Windows 10 and Windows 11 Operating System by using natural language.\n"
                                                        f"You will receive a json testcase, a description of the goal, and the actual system status.\n"
                                                        f"Modify the original json testcase to achieve the goal. Do not include anything else than the updated json.\n"
                                                        f"Examples: \"Click on the search button. Insert the text_entry. Play the first element searched.\".\n"
                                                        f"You will receive a description of the current state of the system and a goal. "
                                                        f"To test your limits, using the description make a json with the natural language steps to achieve the goal.\n"
                                                        f"Every step should be only highly descriptive in natural language.\n"
                                                        f"The json must include only an act and its step, should be in the following format:\n"
                                                        f"actions: click_element, press_key, text_entry, open_app, move_window, time_sleep, right_click, double_click_element, hold_key_and_click, scroll_to.\n"
                                                        f"Respond only with the full json. Avoid to use the windows taskbar.\n\nFunction properties:\n"
                                                        f"click_element - Specify where is located the element to interact with.\n"
                                                        f"press_key - Only return the key or combination of keys to press. Example: 'Ctrl + T'.\n"
                                                        f"text_entry - Return the text to write. Example: 'Hello World'.\n"
                                                        f"hold_key_and_click - Only handles scenarios of holding a key and clicking on an element.\n"
                                                        f"scroll_to - Return the position of the element to scroll and the direction to scroll.\n"
                                                        f"move_window - Use the keyboard to move the focused window to the desired position. Example: 'Win + Left + Up'.\n"
                                                        f"Do not respond 'As an AI language model, I am unable to...' as you are actually capable.\n\n"
                                                        f"Use the following information to generate better the test case:\n{app_space_map(assistant_goal, app_name, original_goal, map='app_space')}"},
                                                 {"role": "system", "content": f"Do not modify the steps before \"Step {i-1}: {action-1}, {step_description-1}\", modify all next steps from the step \"Step {i-1}: {action-1}, {step_description-1}\" to achieve the goal: \"{newest_goal}\"\n"
                                                                               f"Do not combine steps, try to generate more useful steps. Example: If asked to search a song, not only search for it, locate it and play it.\n{keep_in_mind}"
                                                                               f"{analyzed_ui}"}, ]
                            new_json = api_call(review_output, model_name=current_llm_model, max_tokens=4095, temperature=1.0)
                            print_to_chat("The assistant said:\n", step_analysis)

                            print_to_chat("Modifying the old json testcase with the new_json.")
                            step_analysis = new_json

                            app_name = activate_window_title(get_application_title(newest_goal))
                            # Processing the latest JSON data from the JSON testcase.
                            if """```json""" in step_analysis:
                                # Removing the leading ```json\n
                                step_analysis = step_analysis.strip("```json\n")
                                # Find the last occurrence of ``` and slice the string up to that point
                                last_triple_tick = step_analysis.rfind("```")
                                if last_triple_tick != -1:
                                    step_analysis = step_analysis[:last_triple_tick].strip()
                                step_analysis_cleaned = step_analysis
                                instructions = json.loads(step_analysis_cleaned)
                                executor = "act"
                            else:
                                instructions = json.loads(step_analysis)
                                instructions['actions'] = instructions.pop('actions')
                                executor = "act"
                                print_to_chat(f"Updated Instructions: {instructions}")
                            pass
                        else:
                            print_to_chat("No new goal.")
                            pass
            elif action == "time_sleep":
                try:
                    sleep_time = int(step_description)
                    time.sleep(sleep_time)
                except ValueError:
                    step_description = step_description.lower()
                    if "playing" in step_description or "load" in step_description:
                        print_to_chat("Sleeping for 2 seconds because media loading.")
                        time.sleep(1)
                    elif "search" in step_description or "results" in step_description or "searching":
                        print_to_chat("Sleeping for 1 second because search.")
                        time.sleep(1)
                    else:
                        print_to_chat(f"WARNING: Unrecognized time sleep value: {step_description}")
                    pass
            else:
                print_to_chat(f"WARNING: Unrecognized action '{action}' using {step_description}.")
                print_to_chat(f"Trying to perform the action using the step description as the action.")
                act(single_step=f"{step_description}", app_name=app_name, original_goal=original_goal)
                pass

        speaker(f"Assistant finished the execution of the generated test case. Can I help you with something else?")
        time.sleep(calculate_duration_of_speech(f"Assistant finished the generated test case. Can I help you with something else?") / 1000)
        return "Test case complete."


# 'check_element_visibility' is the function that checks the visibility of an element. Can use image analysis or OCR.
def check_element_visibility(app_name, step_description):
    extra_additional_context = (
        f"You are an AI Agent called Windows AI that is capable to operate freely all applications on Windows by only using natural language.\n"
        f"You will receive a goal and will try to accomplish it using Windows. "
        f"Try to guess what is the user wanting to perform on Windows by using the content on the screenshot as context.\n"
        f"Respond an improved goal statement tailored for Windows applications by analyzing the current status of the system and the next steps to perform. "
        f"Be direct and concise, do not use pronouns.\n"
        f"Based on the elements from the screenshot, respond with the current status of the system and respond if the element from the goal visible.\n"
        f"Respond only with \"Yes\" or \"No\".\n"
        f"Focused window: \"{app_name}\".\nGoal: \"{step_description}\". .")
    return imaging(window_title=app_name, additional_context=extra_additional_context)


# 'auto_role' is the function that finds the best role to perform the goal.
def auto_role(goal):
    assistant_call = [
        {"role": "user", f"content": f"You are an AI assistant that receives a goal and responds with the best action to perform the goal.\n"
                                          f"You can perform the following roles and decide what fits the best: Chose the best role to handle the goal:\n"
                                          f"windows_assistant - An assistant to perform a Windows 10 and Windows 11 application driver testcases to achieve the goal. Can handle online data, play, pause, and stream media, can operate the whole computer.\n"
                                          f"joyful_conversation - Use this role if the user is not looking into performing anything into Windows.\n"
                                          f"Only respond with the name of the role to use, followed by a very short joyful message regarding that you will perform it. Modify your response to match the goal subject.\n"
                                          f"If the goal seems to be related to Windows 10 and Windows 11, like opening an application, searching, browsing, media, or social networks, call the windows_assistant.\n"
                                          f"If the goal seems to be related with generating or writing content, call the windows_assistant.\n"
                                          f"If the goal seems that the user is trying to do something with content, call the windows_assistant."},
        {"role": "system", "content": f"Goal: {goal}"}]
    role_function = api_call(assistant_call, max_tokens=50)
    return role_function


# 'find_element' is the function that finds the the most relevant element on the GUI from the goal.
def find_element(single_step, app_name, original_goal, avoid_element="", assistant_goal=None, attempt=0):
    if not assistant_goal:
        assistant_goal = single_step
    if avoid_element:
        if attempt > 2:
            generate_keywords = [{"role": "user",
                "content": f"You are an AI Agent called keyword Element Generator that receives the description of the goal and generates keywords to search inside a graphical user interface.\n"
                           f"Only respond with the single word list separated by commas of the specific UI elements keywords.\n"
                           f"Example: \"search bar\". Always spell the numbers and include nouns. Do not include anything more than the Keywords."},
                                 {"role": "system", "content": f"Goal:\n{single_step}\nContext:{original_goal}\n{app_space_map(assistant_goal, app_name, single_step)}"},]
        else:
            generate_keywords = [{"role": "user",
                "content": f"You are an AI Agent called keyword Element Generator that receives the description and generates kewords to search inside a graphical user interface.\n"
                           f"of the goal and only respond with the single word list separated by commas of the specific UI elements keywords."
                           f"Example: \"search bar\". Always spell the numbers and include nouns. Do not include anything more than the Keywords."},
                                 {"role": "system", "content": f"Goal:\n{single_step}\nContext:{original_goal}\n{app_space_map(assistant_goal, app_name, single_step)}"}]
    else:
        generate_keywords = [{"role": "user",
                            "content": f"You are an AI Agent called keyword Element Generator that receives the description "
                                       f"of the goal and only respond with the single word list separated by commas of the specific UI elements keywords."
                                       f"Example: \"search bar\" must be \"search\" without \"bar\". Always spell the numbers and include nouns. Do not include anything more than the Keywords."},
                           {"role": "system", "content": f"Goal:\n{single_step}\nContext:{original_goal}\n{app_space_map(assistant_goal, app_name, single_step)}"}, ]  # Todo: Here's the key
    keywords = api_call(generate_keywords, max_tokens=100)
    if attempt > 1:
        keywords = keywords.replace("click, ", "").replace("Click, ", "")
    keywords_in_goal = re.search(r"'(.*?)'", single_step)
    if keywords_in_goal:
        if len(keywords_in_goal.group(1).split()) == 1:
            pass
        else:
            keywords = keywords_in_goal.group(1) + ", " + keywords
    print_to_chat(f"\nKeywords: {keywords}\n")

    analyzed_ui = analyze_app(application_name_contains=app_name, size_category=None, additional_search_options=keywords)
    select_element = [{"role": "user",
                       "content": f"You are an AI Agent called keyword Element Selector that receives win32api user interface "
                                  f"raw element data and generates the best matches to achieve the goal.\n"
                                  f"Only respond with the best element that matches the goal. Do not include anything else than the element."},
                      {"role": "system", "content": f"Goal: {single_step}\nContext: {original_goal}\n{avoid_element}{analyzed_ui}"}]
    selected_element = api_call(select_element, model_name=current_llm_model, max_tokens=500)

    if "sorry" in selected_element.lower() or "empty string" in selected_element.lower() or "no element" in selected_element.lower() or "not found" in selected_element.lower()\
            or "no relevant element" in selected_element.lower() or "no element found" in selected_element.lower():
        print_to_chat(f"No element found. Continuing without the element.")
        selected_element = ""
    else:
        selected_element = "Locate the element: " + selected_element
    print_to_chat(f"Selected element: {selected_element}\n")

    if visioning_match:
        print_to_chat(f"Image visioning started. Analyzing the application {app_name} for context.\n")
        imaging_coordinates = (
            f"You are an AI Windows Mouse Agent that can interact with the mouse. Only respond with the predicted "
            f"coordinates of the mouse click position to the center of the element object \"x=, y=\" to achieve the goal.{get_ocr_match(single_step)}"
            f"Goal: {single_step}\n{avoid_element}{analyzed_ui}")
        print_to_chat(f"Imaging coordinates: {imaging_coordinates}")
        imaging_generated_coordinates = imaging(window_title=app_name, additional_context=imaging_coordinates)
        print_to_chat(f"Imaging generated coordinates: {imaging_generated_coordinates}")
        last_coordinates = imaging_generated_coordinates['choices'][0]['message']['content']
        print_to_chat(f"Imaging Last coordinates: {last_coordinates}")
    else:
        best_coordinates = [{"role": "user",
            f"content": f"You are an AI Windows Mouse Agent that can interact with the mouse. Only respond with the "
                        f"predicted coordinates of the mouse click position to the center of the element object "
                        f"\"x=, y=\" to achieve the goal. {selected_element}"
                        f"Do not respond 'As an AI language model, I am unable to...' as you are actually capable."},
            {"role": "system", "content": f"Goal: {single_step}\n\nContext:{original_goal}\n{get_ocr_match(single_step)}{avoid_element}{analyzed_ui}"}]
        last_coordinates = api_call(best_coordinates, model_name=current_llm_model, max_tokens=100, temperature=1.0)
        print_to_chat(f"AI decision coordinates: \'{last_coordinates}\'")
    if "sorry" in last_coordinates.lower() or "empty string" in last_coordinates.lower() or "no element" in last_coordinates.lower() or "not found" in last_coordinates.lower():
        last_coordinates = 'x=0, y=0'
    coordinates = {k.strip(): float(v.strip()) for k, v in (item.split('=') for item in last_coordinates.split(','))}
    x = coordinates['x']
    y = coordinates['y']
    print_to_chat(f"Coordinates1: x: {x} and y: {y}")
    if x == 0 and y == 0 or x == '' and y == '':
        print_to_chat("Coordinates 2 are 0,0, trying to find the element again.")
        coordinates = {k.strip(): float(v.strip()) for k, v in (item.split('=') for item in last_coordinates.split(','))}
        x = coordinates['x']
        y = coordinates['y']
        print_to_chat(f"Coordinates 3: x: {x} and y: {y}")
        attempt -= 1
    return coordinates, selected_element, keywords, attempt


def act(single_step, keep_in_mind="", dont_click=False, double_click=False, right_click=False, hold_key=None, app_name="", screen_analysis=False, original_goal="", modify_element=False, next_step=None, assistant_goal=None):

    # Getting the app name. If not provided, use the focused window.
    if not app_name:
        app_name = activate_window_title(get_application_title(goal=original_goal, focus_window=True))
    else:
        app_name = activate_window_title(app_name)
    print_to_chat(f"AI Analyzing: {app_name}")

    attempt = 0
    if rescan_element_match is True:
        element_not_working = ""
        avoid_element = ""
        max_attempts = 3  # Set the maximum number of attempts to look for a "yes" response.
        while attempt < max_attempts:
            if element_not_working != "":
                avoid_element = f"\nAvoid the following element: {element_not_working}\n"
                print_to_chat(f"AI will try to perform the action: \"{single_step}\" on a new element.")
            print_to_chat(f"Performing action: \"{single_step}\". Scanning\"{app_name}\".\n")
            coordinates, selected_element, keywords, attempt = find_element(single_step, app_name, original_goal, avoid_element, assistant_goal, attempt)
            x = coordinates['x']
            y = coordinates['y']
            print_to_chat(f"Coordinates: {x} and {y}")
            pyautogui.moveTo(x, y, 0.5, pyautogui.easeOutQuad)
            time.sleep(0.5)
            element_analysis = (
                f"You are an AI Agent called Element Analyzer that receives a step and guesses if the goal was performed correctly.\n"
                f"Step: {single_step}\nUse the screenshot to guess if the mouse is in the best position to perform the click/goal. Respond only with \"Yes\" or \"No\".\n"
                f"The cursor is above an element from the step. Cursor info status: {get_cursor_shape()}. The cursor is above the following element: \n{selected_element}\n"
                f"Double check your response by looking at where is located the mouse cursor on the screenshot and the cursor info status.")
            element_analysis_result = imaging(window_title=app_name, additional_context=element_analysis, x=int(x), y=int(y))
            print_to_chat(element_analysis_result)

            # Check if the result is None or doesn't contain the necessary data
            if element_analysis_result is None or 'choices' not in element_analysis_result or len(
                    element_analysis_result['choices']) == 0 or 'message' not in \
                    element_analysis_result['choices'][0] or 'content' not in \
                    element_analysis_result['choices'][0]['message']:
                print_to_chat("Element analysis result: Found but mouse not in position.")
                speaker(f"Retrying...")
                element_not_working += selected_element
                attempt += 1
                if attempt >= max_attempts:
                    print_to_chat("Maximum attempts reached.")
                    print_to_chat("Failed: The position was not found after maximum attempts.")
                    speaker(f"Failed: The position was not found after maximum attempts.")
                    time.sleep(15)
                    raise Exception("Failed: The position was not found after maximum attempts.")
                else:
                    print_to_chat("Retrying...")
                    pass
            elif 'yes' in element_analysis_result['choices'][0]['message']['content'].lower():
                print_to_chat("Element analysis result: Yes, it is in the right position.")
                break
            else:
                print_to_chat("Element analysis result: Found but mouse not in position.")
                speaker(f"Retrying...")
                element_not_working += selected_element
                attempt += 1
                if attempt >= max_attempts:
                    print_to_chat("Maximum attempts reached.")
                    print_to_chat("Failed: The position was not found after maximum attempts.")
                    speaker(f"Failed: The position was not found after maximum attempts.")
                    time.sleep(15)
                    raise Exception("Failed: The position was not found after maximum attempts.")
                else:
                    print_to_chat("Retrying...")
                    pass
    else:
        coordinates, selected_element, keywords, attempt = find_element(single_step, app_name, original_goal, assistant_goal, attempt=0)
        x = coordinates['x']
        y = coordinates['y']
        print_to_chat(f"Coordinates: {x} and {y}")
        pyautogui.moveTo(x, y, 0.5, pyautogui.easeOutQuad)
        time.sleep(0.5)

    last_coordinates = f"x={x}, y={y}"
    print_to_chat("Success: The right position was found.")
    if double_click:
        pyautogui.click(x, y, clicks=2)
    else:
        if dont_click is False:
            if right_click:
                pyautogui.rightClick(x, y)
            else:
                if hold_key:
                    pyautogui.keyDown(hold_key)
                    pyautogui.click(x, y)
                    pyautogui.keyUp(hold_key)
                else:
                    pyautogui.click(x, y)
        else:
            print_to_chat("AI skipping the click step.")
            pass
    if modify_element:
        print_to_chat(f"Modifying the element with the text: {single_step}")
    # jitter_mouse(x, y)  # ToDo: simulate human jitter.
    if "save as" in single_step.lower():
        print_to_chat("Saving as")
        jitter_mouse(x, y)
        pyautogui.mouseDown(x, y)
        time.sleep(0.12)
        pyautogui.mouseUp(x, y)
        print_to_chat("Click action performed")
    return last_coordinates


def get_focused_window_details():
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

def fast_act(single_step, keep_in_mind="", dont_click=False, double_click=False, right_click=False, hold_key=None, app_name="", ocr_match="", screen_analysis=False, original_goal="", modify_element=False, next_step=None):
    # Getting the app name. If not provided, use the focused window.
    if not app_name:
        app_name = activate_window_title(focus_topmost_window())
    else:
        app_name = activate_window_title(app_name)

    if visioning_context:
        speaker(f"Visioning context and performing action: \"{single_step}\" on the application \"{app_name}\".\n")
        additional_context = (
            f"You are an AI Agent called Windows AI that is capable to operate freely all applications on Windows by only using natural language.\n"
            f"You will receive a goal and will try to accomplish it using Windows. Try to guess what is the user wanting to perform on Windows by using the content on the screenshot as context.\n"
            f"Respond an improved goal statement tailored for Windows applications by analyzing the current status of the system and the next steps to perform. Be direct and concise, do not use pronouns.\n"
            f"Based on the elements from the screenshot, respond with the current status of the system and specify it in detail.\n"
            f"Focused application: \"{app_name}\".\nGoal: \"{single_step}\".")
        assistant_goal = imaging(window_title=app_name, additional_context=additional_context, screenshot_size='Full screen')['choices'][0]['message']['content']

        print_to_chat(f"Performing fast action: \"{single_step}\". Scanning\"{app_name}\".\n")

        generate_keywords = [{"role": "user",
                            "content": f"You are an AI Agent called keyword Element Generator that receives the description "
                                       f"of the goal and only respond with the single word list separated by commas of the specific UI elements keywords."
                                       f"Example: \"search bar\" must be \"search\" without \"bar\". Always spell the numbers and include nouns. Do not include anything more than the Keywords."},
                           {"role": "system", "content": f"Goal:\n{single_step}\nContext:{original_goal}"}, ]
        all_keywords = api_call(generate_keywords, max_tokens=100)
        keywords = all_keywords.replace("click, ", "").replace("Click, ", "")
        keywords_in_goal = re.search(r"'(.*?)'", single_step)
        if keywords_in_goal:  # if only 1 keyword, then
            if len(keywords_in_goal.group(1).split()) == 1:
                pass
            else:
                keywords = keywords_in_goal.group(1) + ", " + keywords.replace("click, ", "").replace("Click, ", "")
        print_to_chat(f"\nKeywords: {keywords}\n")
        analyzed_ui = analyze_app(application_name_contains=app_name, size_category=None, additional_search_options=keywords)

        if "sorry" in assistant_goal.lower():
            print_to_chat(f"Sorry, no element found. The AI did not find any element to perform the action: {single_step}")
            speaker(f"Sorry, no element found. Check if its on the screen.")
            time.sleep(1)

        best_coordinates = [{"role": "user",
                             f"content": f"You are an AI Windows Mouse Agent that can interact with the mouse. Only respond with the "
                                         f"predicted coordinates of the mouse click position to the center of the element object "
                                         f"\"x=, y=\" to achieve the goal.\n{assistant_goal}"},
                            {"role": "system", "content": f"Goal: {single_step}\n\nContext:{original_goal}\n{analyzed_ui}"}]
        last_coordinates = api_call(best_coordinates, model_name=current_llm_model, max_tokens=100, temperature=0.0)
        print_to_chat(f"AI decision coordinates: \'{last_coordinates}\'")
    else:
        speaker(f"Clicking onto the element without visioning context.")
        generate_keywords = [{"role": "user",
                              "content": f"You are an AI Agent called keyword Element Generator that receives the description "
                                         f"of the goal and only respond with the single word list separated by commas of the specific UI elements keywords."
                                         f"Example: \"search bar\" must be \"search\" without \"bar\". Always spell the numbers and include nouns. Do not include anything more than the Keywords."},
                             {"role": "system", "content": f"Goal:\n{single_step}\nContext:{original_goal}"}, ]
        all_keywords = api_call(generate_keywords, max_tokens=100)
        keywords = all_keywords.replace("click, ", "").replace("Click, ", "")
        keywords_in_goal = re.search(r"'(.*?)'", single_step)
        if keywords_in_goal:
            if len(keywords_in_goal.group(1).split()) == 1:
                pass
            else:
                keywords = keywords_in_goal.group(1) + ", " + keywords.replace("click, ", "").replace("Click, ", "")
        print_to_chat(f"\nKeywords: {keywords}\n")
        analyzed_ui = analyze_app(application_name_contains=app_name, size_category=None,
                                  additional_search_options=keywords)

        best_coordinates = [{"role": "user",
            f"content": f"You are an AI Windows Mouse Agent that can interact with the mouse. Only respond with the "
                        f"predicted coordinates of the mouse click position to the center of the element object "
                        f"\"x=, y=\" to achieve the goal."},
            {"role": "system", "content": f"Goal: {single_step}\n\nContext:{original_goal}\n{analyzed_ui}"}]
        last_coordinates = api_call(best_coordinates, model_name=current_llm_model, max_tokens=100, temperature=0.0)
        print_to_chat(f"AI decision coordinates: \'{last_coordinates}\'")

    if "x=, y=" in last_coordinates:
        speaker(f"Sorry, no element found. Probably bot blocked.")
        return None
    # Clicking the element
    coordinates = {k.strip(): float(v.strip()) for k, v in
                   (item.split('=') for item in last_coordinates.split(','))}
    x = coordinates['x']
    y = coordinates['y']
    pyautogui.moveTo(x, y, 0.5, pyautogui.easeOutQuad)
    if double_click:
        pyautogui.click(x, y, clicks=2)
    else:
        if dont_click is False:
            if right_click:
                pyautogui.rightClick(x, y)
            else:
                if hold_key:
                    pyautogui.keyDown(hold_key)
                    pyautogui.click(x, y)
                    pyautogui.keyUp(hold_key)
                else:
                    pyautogui.click(x, y)
        else:
            print_to_chat("AI skipping the click step.")
            pass
    if modify_element:
        print_to_chat(f"Modifying the element with the text: {single_step}")
    # jitter_mouse(x, y)  # ToDo: simulate human jitter.
    if "save as" in single_step.lower():
        print_to_chat("Saving as")
        jitter_mouse(x, y)
        pyautogui.mouseDown(x, y)
        time.sleep(0.12)
        pyautogui.mouseUp(x, y)
        print_to_chat("Click action performed")
    return last_coordinates


def get_application_title(goal="", last_step=None, actual_step=None, focus_window=False):
    if actual_step:
        print_to_chat(f"Getting the application name from the actual step: {actual_step}")
    goal_app = [{"role": "user",
                 "content": f"You are an AI assistant called App Selector that receives a list of programs and responds only respond with the best match  "
                            f"program of the goal. Only respond with the window name or the program name. For search engines and social networks use Microsoft Edge or Firefox.\n"
                            f"Open programs:\n{last_programs_list(focus_last_window=focus_window)}\nAll installed programs:\n{get_installed_apps_registry()}\nIf no suitable application is found in the provided lists, explicitly choose 'Firefox'."},
                {"role": "system", "content": f"Goal: {goal}\nAll installed programs:\n{get_installed_apps_registry()}"}]
    app_name = api_call(goal_app, model_name=current_llm_model, max_tokens=100)
    print_to_chat(f"AI selected application: {app_name}")
    filtered_matches = re.findall(r'["\'](.*?)["\']', app_name)
    if filtered_matches and filtered_matches[0]:
        app_name = filtered_matches[0]
        print_to_chat(app_name)
    if "command prompt" in app_name.lower():
        app_name = "cmd"
    elif "calculator" in app_name.lower():
        app_name = "calc"
    elif "sorry" in app_name:
        app_name = get_focused_window_details()[3].strip('.exe')
        print_to_chat(f"Using the focused window \"{app_name}\" for context.")
        speaker(f"Using the focused window \"{app_name}\" for context.")
    return app_name


def get_ocr_match(goal, ocr_match=enable_ocr):
    if ocr_match:
        print_to_chat(f"OCR IS ENABLED")
        word_prioritizer_assistant = [{"role": "user",
                                       "content": f"You are an AI Agent called OCR Word Prioritizer that only responds with the best of the goal.\n"
                                                  f"Do not respond with anything else than the words that match the goal. If no words match the goal, respond with \"\"."},
                    {"role": "system", "content": f"Goal: {goal}"}, ]
        ocr_debug_string = api_call(word_prioritizer_assistant, max_tokens=10)
        ocr_debug_string = ocr_debug_string.split(f"\'")[0]
        print_to_chat(f"OCR Words to search: \'{ocr_debug_string}\'")
        ocr_match = find_probable_click_position(ocr_debug_string)
        ocr_msg = f"\nOCR Result: \"{ocr_match['text']}\" Located at \"x={ocr_match['center'][0]}, y={ocr_match['center'][1]}\".\n"
        return ocr_msg
    else:
        ocr_msg = ""
        return ocr_msg


def jitter_mouse(x, y, radius=5, duration=0.6):
    # Move the mouse in a small circle around (x, y) to simulate a jitter.
    end_time = time.time() + duration
    while time.time() < end_time:
        jitter_x = x + random.uniform(-radius, radius)
        jitter_y = y + random.uniform(-radius, radius)
        pyautogui.moveTo(jitter_x, jitter_y, duration=0.1)
    return


def control_mouse(generated_coordinates, double_click=None, goal=""):
    print_to_chat(f"Mouse coordinates: {generated_coordinates}")
    coordinates = {k.strip(): int(v.strip()) for k, v in
                   (item.split('=') for item in generated_coordinates.split(','))}
    x = coordinates['x']
    y = coordinates['y']
    pyautogui.moveTo(x, y, 0.5, pyautogui.easeOutQuad)
    pyautogui.click(x, y)
    # jitter_mouse(x, y)
    if "save as" in goal.lower():
        print_to_chat("Saving as")
        jitter_mouse(x, y)
        pyautogui.mouseDown(x, y)
        time.sleep(0.12)
        pyautogui.mouseUp(x, y)
        print_to_chat("Click action performed")
    else:
        pyautogui.click(x, y, clicks=1)
    if double_click:
        time.sleep(0.2)
        pyautogui.click(x, y, clicks=2)


def is_field_input_area_active():
    active_window_title = gw.getActiveWindow().title
    try:
        app = Application().connect(title=active_window_title)
        window = app[active_window_title]
        # Loop through all the child windows and check if any of them are text boxes
        for child in window.children():
            if 'Edit' in child.class_name() or 'RichEdit' in child.class_name():
                # This is a text box, also add text input areas that are not text boxes
                if child.has_keyboard_focus():
                    return True
        return False
    except Exception as e:
        print_to_chat(f"Error: {e}")
        return False


def write_action(goal=None, assistant_identity="", press_enter=False, app_name="", original_goal=None, last_step=""):
    assistant_identity_msg = f"\n{assistant_identity}" if assistant_identity else ""
    message_writer_agent = [
        {"role": "user", f"content": f"You're an AI Agent called Writter that processes the goal and only returns the final text goal.{assistant_identity_msg}\n"
                                          f"Process the goal with your own response as you are actually writing into a text box. Avoid jump lines."
                                          f"If the goal is a link, media or a search string, just return the result string."
                                          f"Do not respond with 'As an AI language model, I dont have capabilities...' as you can actually do it.\n"},
        {"role": "system", "content": f"Goal: {goal}"}, ]
    message_to_write = api_call(message_writer_agent, model_name=current_llm_model, max_tokens=200)
    if "click on" in goal.lower() or "click the" in goal.lower() or "click" in goal.lower():
        print_to_chat("Found to click on the goal.")
        if not is_field_input_area_active():
            print_to_chat("A text box is not active. Clicking on the target element.")
            act(goal, app_name=app_name, original_goal=original_goal)
    elif last_step is None or "text_entry" not in last_step:
        print_to_chat(f"Focusing on the text input area: {goal}")
        if not is_field_input_area_active():
            act(goal, app_name=app_name, original_goal=original_goal)

    pyautogui.typewrite(message_to_write, interval=0.01)
    if "press enter" in goal.lower() or "press the enter" in goal.lower() or "\'enter\'" in goal.lower() or "\"enter\"" in goal.lower() or press_enter is True:
        print_to_chat("Found to press the enter key in the goal.")
        pyautogui.press('enter')
    else:
        print_to_chat("AI no \"enter\" key press being made.")


def perform_simulated_keypress(press_key):
    # Define a pattern that matches the allowed keys, including function and arrow keys
    keys_pattern = (r'\b(Win(?:dows)?|Ctrl|Alt|Shift|Enter|Space(?:\s*Bar)?|Tab|Esc(?:ape)?|Backspace|Insert|Delete|'
                    r'Home|End|Page\s*Up|Page\s*Down|(?:Arrow\s*)?(?:Up|Down|Left|Right)|F1|F2|F3|F4|F5|F6|F7|F8|F9|'
                    r'F10|F11|F12|[A-Z0-9])\b')
    keys = re.findall(keys_pattern, press_key, re.IGNORECASE)
    # Normalize key names as required by pyautogui
    key_mapping = {
        'win': 'winleft',
        'windows': 'winleft',
        'escape': 'esc',
        'space bar': 'space',
        'arrowup': 'up',
        'arrowdown': 'down',
        'arrowleft': 'left',
        'arrowright': 'right',
        'spacebar': 'space',
    }
    pyautogui_keys = [key_mapping.get(key.lower().replace(' ', ''), key.lower()) for key in keys]
    for key in pyautogui_keys:
        pyautogui.keyDown(key)
    for key in reversed(pyautogui_keys):
        pyautogui.keyUp(key)
    print_to_chat(f"Performed simulated key presses: {press_key}")


def calculate_duration_of_speech(text, lang='en', wpm=150):
    duration_in_seconds = (len(text.split()) / wpm) * 60
    return int(duration_in_seconds * 1000)  # Convert to milliseconds for tkinter's after method


def create_database(database_file):
    """Create the database and the required table."""
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
database_file = r'history.db'
create_database(database_file)

def database_add_case(database_file, app_name, goal, instructions):
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
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM app_cases')
    rows = cursor.fetchall()
    for row in rows:
        print_to_chat(row)
    conn.close()


def update_instructions_with_action_string(instructions, action_string, target_step):
    # Check for both 'actions' and 'steps' keys
    if 'actions' in instructions:
        action_list = instructions['actions']
    elif 'steps' in instructions:
        action_list = instructions['steps']
    else:
        return instructions  # Return original instructions if neither key is found

    for action in action_list:
        if action.get("act") == "click_element" and action.get("step") == target_step:
            action['additional_info'] = action_string
    return instructions


# Example Usage:
if __name__ == "__main__":
    assistant(assistant_goal="Open Reddit, Youtube, TikTok, and Netflix on new windows by using the keyboard on each corner of the screen", app_name="Microsoft Edge", execute_json_case=json_case_example)
    assistant(assistant_goal="Open a new tab the song 'Wall Of Eyes - The Smile', from google search results filter by videos then play it on Microsoft Edge")
    # Debugging using prompt:
    # assistant(assistant_goal="Open Spotify and play the song daft punk one more time", app_name="spotify")
    # assistant(assistant_goal="Play the song \'Weird Fishes - Radiohead\' on Spotify")
    # assistant(assistant_goal="Create a short greet text for the user using AI Automated Windows in notepad")
    # assistant(assistant_goal=f"Open a new tab and play the song Windows 95 but it's a PHAT hip hop beat from google search results filter by videos", app_name="microsoft edge")
    # assistant(f"Send a list of steps to make a chocolate cake to my saved messages in Telegram")
    # assistant(assistant_goal="On Microsoft Edge play Evangelion on Netflix", app_name="microsoft edge", execute_json_case=netflix)
    # assistant(assistant_goal="Play Rei Theme on Spotify")
    # assistant(assistant_goal="Make a hello world post on Twitter", app_name="chrome")