import os
from typing import Any, Dict, Tuple
import litellm

current_llm_model = None
current_api_key_env_name = None
current_vision_llm_model = None
current_vision_api_key_env_name = None
current_llm_provider_options: Dict[str, Any] = {}
current_vision_provider_options: Dict[str, Any] = {}

# --- Provider helpers ---

def _split_provider_model(model_name: str) -> Tuple[str, str]:
    if not model_name:
        return ("openai", "gpt-4o")
    if "/" in model_name:
        p, m = model_name.split("/", 1)
        return (p or "openai", m)
    return ("openai", model_name)

# Map provider -> env var names for extras we can pass to LiteLLM
# We rely on env vars since advanced UI is not enabled
PROVIDER_ENV_HINTS: Dict[str, Dict[str, str]] = {
    # key: provider, value: mapping of litellm kwarg -> env var name
    "openai": {},
    "openai_compatible": {"api_base": "OPENAI_API_BASE"},
    "azure": {"api_base": "AZURE_API_BASE", "api_version": "AZURE_API_VERSION"},
    "azure_ai": {"api_base": "AZURE_API_BASE", "api_version": "AZURE_API_VERSION"},
    "vertex": {"vertex_project": "VERTEXAI_PROJECT", "vertex_location": "VERTEXAI_LOCATION"},
    "gemini": {},
    "anthropic": {},
    "bedrock": {"aws_region_name": "AWS_REGION"},
    "aws_sagemaker": {"aws_region_name": "AWS_REGION"},
    "mistral": {},
    "cohere": {},
    "huggingface": {"api_base": "HUGGINGFACE_API_BASE"},
    "groq": {},
    "deepseek": {},
    "openrouter": {"api_base": "OPENROUTER_API_BASE"},
    "togetherai": {"api_base": "TOGETHERAI_API_BASE"},
    "replicate": {},
    "fireworks_ai": {"api_base": "FIREWORKS_API_BASE"},
    "perplexity": {},
    "ollama": {"api_base": "OLLAMA_API_BASE"},
    "github": {"api_base": "GITHUB_MODELS_API_BASE"},
    "github_copilot": {},
    "clarifai": {},
    "vllm": {"api_base": "VLLM_API_BASE"},
    "llamafile": {"api_base": "LLAMAFILE_API_BASE"},
    "cloudflare_workers": {"api_base": "CLOUDFLARE_WORKERS_API_BASE"},
}

def _provider_kwargs(provider: str) -> Dict[str, Any]:
    mapping = PROVIDER_ENV_HINTS.get(provider, {})
    kwargs: Dict[str, Any] = {}
    for litellm_kwarg, env_var in mapping.items():
        val = os.environ.get(env_var)
        if val:
            kwargs[litellm_kwarg] = val
    # Common fallbacks
    if "api_base" not in kwargs:
        base = os.environ.get("API_BASE")
        if base:
            kwargs["api_base"] = base
    return kwargs

def get_provider_kwargs_for_model(model_name: str) -> Dict[str, Any]:
    provider, _ = _split_provider_model(model_name or "")
    return _provider_kwargs(provider)

# --- Response normalization ---

def _extract_text_from_response(resp: Any) -> str:
    """Return best-effort text from a LiteLLM response across providers/models."""
    try:
        # LiteLLM ModelResponse shape
        choices = getattr(resp, "choices", None)
        if choices:
            msg = getattr(choices[0], "message", None)
            if msg is not None:
                content = getattr(msg, "content", None)
                # content may be string, list, or None
                if isinstance(content, str):
                    return content.strip()
                if isinstance(content, list):
                    parts = []
                    for item in content:
                        if isinstance(item, dict):
                            t = item.get("text") or item.get("content") or ""
                            if t:
                                parts.append(str(t))
                        elif isinstance(item, str):
                            parts.append(item)
                    if parts:
                        return "\n".join(p.strip() for p in parts if str(p).strip())
                rc = getattr(msg, "reasoning_content", None)
                if isinstance(rc, str) and rc.strip():
                    return rc.strip()
        # Dict-like fallbacks
        if isinstance(resp, dict):
            choices = resp.get("choices") or []
            if choices:
                msg = choices[0].get("message") if isinstance(choices[0], dict) else None
                if msg:
                    content = msg.get("content")
                    if isinstance(content, str):
                        return content.strip()
                    if isinstance(content, list):
                        parts = []
                        for item in content:
                            if isinstance(item, dict):
                                t = item.get("text") or item.get("content") or ""
                                if t:
                                    parts.append(str(t))
                            elif isinstance(item, str):
                                parts.append(item)
                        if parts:
                            return "\n".join(p.strip() for p in parts if str(p).strip())
                    rc = msg.get("reasoning_content")
                    if isinstance(rc, str) and rc.strip():
                        return rc.strip()
        # Final fallback: try str()
        s = str(resp)
        return s.strip()
    except Exception:
        try:
            return str(resp).strip()
        except Exception:
            return ""
def api_call(messages, model_name=None, temperature=0.5, max_tokens=150, **extra_params):
    try:
        # Use the global variable's current value if model_name is not provided
        if model_name is None:
            model_name = current_llm_model
        provider_kwargs = get_provider_kwargs_for_model(model_name or "")
        # Merge precedence: extra_params > user provider options > env-derived
        merged_kwargs = {**provider_kwargs, **current_llm_provider_options, **extra_params}
        # Execute the chat completion using the chosen model
        response = litellm.completion(
            model=model_name,
            messages=messages,
            api_key=os.environ.get(current_api_key_env_name),
            temperature=temperature,  # Values can range from 0.0 to 1.0
            max_tokens=max_tokens,  # This specifies the maximum length of the response
            **merged_kwargs,
        )

        decision = _extract_text_from_response(response)
        return decision if decision is not None else ""
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

def set_llm_model(model_name, provider_options: Dict[str, Any] | None = None):
    global current_llm_model, current_llm_provider_options
    current_llm_model = model_name
    current_llm_provider_options = provider_options or {}

def set_api_key(api_key, api_key_env_name):
    global current_api_key_env_name
    current_api_key_env_name = api_key_env_name
    if api_key:
        os.environ[current_api_key_env_name] = api_key

def set_vision_llm_model(model_name, provider_options: Dict[str, Any] | None = None):
    global current_vision_llm_model, current_vision_provider_options
    current_vision_llm_model = model_name
    current_vision_provider_options = provider_options or {}

def set_vision_api_key(api_key, api_key_env_name):
    global current_vision_api_key_env_name
    current_vision_api_key_env_name = api_key_env_name
    if api_key:
        os.environ[current_vision_api_key_env_name] = api_key