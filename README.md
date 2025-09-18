# Computer Autopilot
Computer Autopilot is an AI Agent capable of automating Windows using natural language. It can automate any task with Windows user interface using visual context analysis with the help of large language models. It features a modern graphical interface with both chat and RPA capabilities, making computer automation accessible and user-friendly.

<img width="1252" height="914" alt="image" src="https://github.com/user-attachments/assets/2dc34790-86d0-445f-aec9-cf92fc4d7bb2" />

<img width="1252" height="914" alt="image" src="https://github.com/user-attachments/assets/532c3632-c689-4549-acc6-c266033e0ead" />

<img width="1252" height="914" alt="image" src="https://github.com/user-attachments/assets/8ae3e9c2-9dac-4a3a-ae00-ef50c941664b" />


# Overview
- Natural language control of your Windows Operating System
- Modern GUI with chat interface and voice input capabilities
- RPA (Robotic Process Automation) tab for managing automated tasks
- Actions generation for UI applications using natural language
- Advanced web assistant for browser automation and webpage interaction
- Supports all Win32api applications including: Microsoft Edge, Chrome, Firefox, OperaGX, Discord, Telegram, Spotify
- Configurable performance settings for both desktop and web automation
- Minimizable to mini chat or mini control interface for better workflow


# Key Features
1. **Dynamic Case Generator**: Translates natural language commands into executable actions through the `assistant()` function.

2. **Modern User Interface**:
   - Chat interface with voice input support
   - Volume control and mute options
   - Subtitle toggle functionality
   - Settings configuration window
   - Minimizable to mini chat/control interfaces
   - RPA task management tab

3. **Advanced Context Handling**: 
   - Screen analysis capabilities
   - Application context awareness
   - Cursor shape detection
   - Window focus management

4. **Semantic Router Map**: 
   - Database of semantic mappings for test case execution
   - AI-extensible mapping system
   - Application-specific action patterns

5. **Wide Application Support**:
   - Multimedia control (Spotify, YouTube)
   - Browser automation (Edge, Chrome, Firefox)
   - Communication apps (Telegram, Discord)
   - General Windows applications

6. **Web Assistant Functionality**:
   - Advanced browser automation
   - Webpage element detection and interaction
   - Form filling and submission
   - JavaScript execution
   - Multi-tab management
   - Robust selector extraction (CSS, XPath)

7. **Customizable Settings**:
   - LLM Model selection
   - Vision LLM Model configuration
   - API key management
   - Environment variable support
   - Action delay customization
   - Web action delay configuration
   - Performance tuning options
   - Windows startup option


# Technical Innovations
1. **Natural Language Processing**: Advanced NLP for command interpretation
2. **Task Automation**: Smart algorithms for task breakdown
3. **Context-Aware Execution**: Integrated contextual analysis
4. **Cross-Application Support**: Seamless multi-app integration
5. **Visual Analysis**: Screen content interpretation
6. **Voice Integration**: Speech input and output capabilities
7. **Web Automation**: Advanced browser interaction with element detection
8. **Selector Extraction**: Robust CSS and XPath selector generation
9. **Performance Optimization**: Configurable action delays and execution parameters


# Installation
## Option 1: Download Release (Recommended)
1. Visit the [Releases](https://github.com/ngotrantronghieu/Computer-Autopilot/releases) section
2. Download the latest release
3. Extract the downloaded file
4. Run `Computer Autopilot.exe`

## Option 2: From Source
```batch
# Clone the repository
git clone https://github.com/ngotrantronghieu/Computer-Autopilot.git

# Change directory
cd .\Computer-Autopilot\

# Create and activate a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

# Install requirements
pip install -r .\requirements.txt

# Execute the assistant
python .\core\assistant.py
```


## Configuration
1. Configure in app settings:
   - LLM Model
   - LLM Vision Model
   - API Keys
   - Performance Settings:
     - Action delay
     - Maximum attempts
     - Web action delay
     - Web maximum attempts
2. Optional: Use environment variables for API keys


# Usage
1. **Main Interface**:
   - Launch `assistant.py`
   - Use text input or voice button for commands
   - Adjust assistant volume with mouse wheel
   - Toggle subtitles or mute as needed
   - Access settings via gear icon

2. **Chat Features**:
   - Natural language input
   - Voice commands
   - Real-time responses
   - Chat history
   - Clear chat option

3. **RPA Tab**:
   - Create and manage automated tasks
   - Save frequently used automations
   - Execute saved tasks
   - Monitor task status

4. **Web Assistant**:
   - Automate browser tasks with natural language
   - Interact with webpage elements (buttons, forms, links)
   - Navigate between websites and tabs
   - Extract information from webpages
   - Execute custom JavaScript
   - Perform complex web workflows

5. **Mini Interfaces**:
   - Minimize to compact chat or control interface
   - Continue interactions while saving screen space
   - Quick access to essential controls

6. **Debug Mode**:
   - Run `driver.py` for debugging
   - Test individual functions: `fast_act()`, `assistant()`, `web_assistant()`
   - Monitor execution flow


# Security and Privacy
- Local execution of automation tasks
- Secure API key management
- Environment variable support for sensitive data
- Open-source codebase for transparency


# System Requirements
- Windows Operating System
- Python 3.x+
- Internet connection for LLM API access
