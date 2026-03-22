# ============================================================
# OLLAMA CLIENT — ResumeIQ
# Handles all communication with local Ollama server
# Make sure Ollama is running: ollama serve
# ============================================================

import requests

OLLAMA_GENERATE = "http://localhost:11434/api/generate"
OLLAMA_TAGS     = "http://localhost:11434/api/tags"
OLLAMA_BASE     = "http://localhost:11434"


def check_ollama_running():
    """Check if Ollama server is running. Returns True/False."""
    try:
        requests.get(OLLAMA_BASE, timeout=3)
        return True
    except:
        return False


def get_available_models():
    """Get list of models installed in Ollama."""
    try:
        response = requests.get(OLLAMA_TAGS, timeout=5)
        models   = response.json().get("models", [])
        names    = [m["name"] for m in models]
        return names if names else ["qwen3.5:0.8b"]
    except:
        return ["qwen3.5:0.8b  "]


def generate_answer(prompt, model="qwen2.5:0.8b", temperature=0.3):
    """
    Send prompt to Ollama and return response.

    temperature 0.3 = focused factual answers
    Perfect for resume Q&A where accuracy matters.

    temperature 0.7 = more creative
    Better for generating varied interview questions.
    """
    payload = {
        "model" : model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature"   : temperature,
            "num_predict"   : 1024,
            "top_k"         : 40,
            "top_p"         : 0.9,
            "repeat_penalty": 1.1,
            "num_ctx"       : 2048
        }
    }

    try:
        response = requests.post(
            OLLAMA_GENERATE,
            json=payload,
            timeout=180
        )
        result = response.json()
        return result.get("response", "No response received.")

    except requests.exceptions.ConnectionError:
        return (
            "ERROR: Ollama is not running.\n"
            "Open terminal and run: ollama serve"
        )

    except requests.exceptions.Timeout:
        return "ERROR: Model timed out. Try again or use a smaller model."

    except Exception as e:
        return f"ERROR: {str(e)}"