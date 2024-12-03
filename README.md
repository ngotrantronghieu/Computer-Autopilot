# Computer Autopilot
Computer Autopilot is an AI Agent capable of automating windows using natural languages. It can automate any task with windows user interface using test case generation and visual context analysis with the help of large language models. It has built-in assistance options to improve human utilization of a computer, with a new technical approach to User Interface and User Experience assistance and testing, generalizes correctly any natural language prompt, and plans to perform correct actions into the OS with security in mind.

<img src="https://github.com/user-attachments/assets/0e64a0f5-6a82-4c17-b518-6f0bf18eb915" alt="Main Chat Interface" width="700"/>

![Mini Chat Interface](https://github.com/user-attachments/assets/7a321b3b-bb93-4f71-b8cd-ff6be39ab2fd)

# Overview
Chat and talk to your computer friendly and naturally to perform any User Interface activity.
Use natural language to operate freely your Windows Operating System.
Generates and plans test cases of your User Interface applications for continuous testing on any Win32api supported application by simply using natural language.
Your own open and secure personal assistant that responds as you want, control the way you want your computer to assist you.
It's engineered to be modular, understand and execute a wide range of tasks, automating interactions with any desktop applications.
Currently supporting all generalized win32api apps, meaning: Microsoft Edge, Chrome, Firefox, OperaGX, Discord, Telegram, Spotify...

# Key Features
1. Dynamic Case Generator: The assistant() function accepts a goal parameter, which is a natural language command, and intelligently maps it to a series of executable actions. This allows for a seamless translation of user intentions into effective actions on the computer.
2. Single Action Execution:
The act() function is a streamlined method for executing single, straightforward actions, enhancing the tool's efficiency and responsiveness.
3. Advanced Context Handling: The framework is adept at understanding context through analyzing the screen and the application, ensuring that actions are carried out with an awareness of the necessary prerequisites or steps.
4. Semantic router map: The framework has a database of a semantic router map to successfully execute generated test cases. This semantic maps can be created by other AI.
5. Wide Application Range: From multimedia control (like playing songs or pausing playback on Spotify and YouTube) to complex actions (like creating AI-generated text, sending emails, or managing applications like Telegram or Firefox), the framework covers a broad spectrum of tasks.
6. Customizable AI Identity: The write_action() function allows for a customizable assistant identity, enabling personalized interactions and responses that align with the user's preferences or the nature of the task.

# Technical Innovations
1. Natural Language Processing (NLP): Employs advanced NLP techniques to parse and understand user commands in a natural, conversational manner.
2. Task Automation Algorithms: Utilizes sophisticated algorithms to break down complex tasks into executable steps.
3. Context-Aware Execution: Integrates contextual awareness for more nuanced and effective task execution.
4. Cross-Application Functionality: Seamlessly interfaces with various applications and web services, demonstrating extensive compatibility and integration capabilities.

# Use Cases
1. Automating repetitive tasks in a Windows environment.
2. Streamlining workflows for professionals and casual users alike.
3. Enhancing accessibility for users with different needs, enabling voice or simple text commands to control complex actions.
4. Assisting in learning and exploration by providing AI-driven guidance and execution of tasks.

# Conclusion
This Large Language Model Based User Interface Automation System is a pioneering tool in the realm of desktop automation. Its ability to understand and execute a wide range of commands in a natural, intuitive manner makes it an invaluable asset for anyone looking to enhance their productivity and interaction with their Windows environment. It's not just a tool; it's a step towards a future where AI seamlessly integrates into our daily computing tasks, making technology more accessible and user-friendly.

# Installation
```
# Install requirements:
cd computerAutopilot
pip install -r .\requirements.txt

# Execute the assistant:
cd .\core
python ./assistant.py

# Go to https://docs.litellm.ai/docs/providers to get the LLM Model Name.
# Add your LLM Model, LLM Vision Model and API Keys in the app settings and you're good to go.
# If you don't want to enter your API Keys in the Settings, you can set the API Keys as Environment Variables and enter the Environment Variables name in the API Key Env Name fields.
```

# Usage
Run "assistant.py", type the task you want to perform in the text field or click the voice button to toggle voice input. Scroll the wheel to change the assistant voice volume, you can also turn off the Subtitle, or mute to mute the assistant voice. Open Settings to change the LLM Model, LLM Vision Model and API Keys of the LLM of your choice.

For debugging mode, execute "driver.py". Inside it, you can debug and try easily the functions of "act" which is used alongside the assistant, "fast_act" and "assistant" by using the examples.
To run a JSON test case, modify the JSON path from the "assistant" function.
