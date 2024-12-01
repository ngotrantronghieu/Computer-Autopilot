import os
import litellm

current_llm_model = "gemini/gemini-1.5-flash-002"
current_api_key_env_name = "GEMINI_API_KEY"
current_vision_llm_model = "gemini/gemini-1.5-flash-002"
current_vision_api_key_env_name = "GEMINI_API_KEY"

def api_call(messages, model_name=current_llm_model, temperature=0.5, max_tokens=150):
    try:
        # Execute the chat completion using the chosen model
        response = litellm.completion(
            model=model_name,
            messages=messages,
            api_key=os.environ.get(current_api_key_env_name),
            temperature=temperature,  # Values can range from 0.0 to 1.0
            max_tokens=max_tokens,  # This specifies the maximum length of the response
        )

        if response.choices and hasattr(response.choices[0], 'message'):
            decision_message = response.choices[0].message

            # Make sure we have 'content' in the message
            if hasattr(decision_message, 'content'):
                decision = decision_message.content.strip()
            else:
                decision = None
        else:
            decision = None

        return decision
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

def set_llm_model(model_name):
    global current_llm_model
    current_llm_model = model_name

def set_api_key(api_key, api_key_env_name):
    global current_api_key_env_name
    current_api_key_env_name = api_key_env_name
    if api_key:
        os.environ[current_api_key_env_name] = api_key

def set_vision_llm_model(model_name):
    global current_vision_llm_model
    current_vision_llm_model = model_name

def set_vision_api_key(api_key, api_key_env_name):
    global current_vision_api_key_env_name
    current_vision_api_key_env_name = api_key_env_name
    if api_key:
        os.environ[current_vision_api_key_env_name] = api_key