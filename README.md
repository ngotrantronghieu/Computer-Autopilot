# Computer Autopilot
Computer Autopilot is an AI Agent capable of automating Windows using natural language. It can automate any task with Windows user interface using visual context analysis with the help of large language models. It features a modern graphical interface with both chat and RPA capabilities, making computer automation accessible and user-friendly.

![image](https://github.com/user-attachments/assets/83c0265d-6ca6-4fa5-a0dc-98caebf9ccd0)

![image](https://github.com/user-attachments/assets/94e11852-a8c6-4985-9202-5a5819f8b36b)


# Overview
- Natural language control of your Windows Operating System
- Modern GUI with chat interface and voice input capabilities
- RPA (Robotic Process Automation) tab for managing automated tasks
- Actions generation for UI applications using natural language
- Supports all Win32api applications including: Microsoft Edge, Chrome, Firefox, OperaGX, Discord, Telegram, Spotify
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

6. **Customizable Settings**:
   - LLM Model selection
   - Vision LLM Model configuration
   - API key management
   - Environment variable support
   - Action delay customization
   - Windows startup option


# Technical Innovations
1. **Natural Language Processing**: Advanced NLP for command interpretation
2. **Task Automation**: Smart algorithms for task breakdown
3. **Context-Aware Execution**: Integrated contextual analysis
4. **Cross-Application Support**: Seamless multi-app integration
5. **Visual Analysis**: Screen content interpretation
6. **Voice Integration**: Speech input and output capabilities


# Installation

## Option 1: Download Release (Recommended)
1. Visit the [Releases](https://github.com/ngotrantronghieu/Computer-Autopilot/releases) section
2. Download the latest release
3. Extract the downloaded file
4. Run `assistant.exe`

## Option 2: From Source
```bash
# Install requirements:
cd Computer-Autopilot
pip install -r .\requirements.txt

# Execute the assistant:
cd .\core
python ./assistant.py
```


## Configuration:
1. Visit https://docs.litellm.ai/docs/providers for LLM Model options
2. Configure in app settings:
   - LLM Model
   - LLM Vision Model
   - API Keys
3. Optional: Use environment variables for API keys


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

4. **Mini Interfaces**:
   - Minimize to compact chat or control interface
   - Continue interactions while saving screen space
   - Quick access to essential controls

5. **Debug Mode**:
   - Run `driver.py` for debugging
   - Test individual functions: `fast_act()`, `assistant()`
   - Monitor execution flow


# Security and Privacy
- Local execution of automation tasks
- Secure API key management
- Environment variable support for sensitive data
- Open-source codebase for transparency


# System Requirements
- Windows Operating System
- Python 3.x
- Internet connection for LLM API access
