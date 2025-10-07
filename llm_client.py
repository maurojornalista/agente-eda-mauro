# llm_client.py — cliente LLM multi-provider (Gemini, Groq, OpenAI)
import os
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

# Status
_loaded = False
_has_openai = False
_has_gemini = False
_has_groq = False

_openai_client = None
_gemini_model = None
_groq_client = None

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")


def _lazy_load():
    global _loaded, _has_openai, _has_gemini, _has_groq
    global _openai_client, _gemini_model, _groq_client
    if _loaded:
        print("Status final do lazy_load: Groq =", _has_groq)
        return

    # OpenAI
    try:
        from openai import OpenAI
        key = os.getenv("OPENAI_API_KEY")
        if key:
            _openai_client = OpenAI(api_key=key)
            _has_openai = True
    except Exception:
        _has_openai = False

    # Gemini
    # Gemini
try:
    import google.generativeai as genai
    key = os.getenv("GOOGLE_API_KEY")
    if key:
        genai.configure(api_key=key)
        # tenta modelo do .env e faz fallback para aliases válidos
        prefer = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")
        candidates = [prefer, "gemini-1.5-flash-latest",
                      "gemini-1.5-pro-latest"]
        model_obj = None
        last_err = None
        for name in candidates:
            try:
                model_obj = genai.GenerativeModel(name)
                # smoke call leve para validar
                _ = model_obj.generate_content("ok")
                _gemini_model = model_obj
                _has_gemini = True
                break
            except Exception as e:
                last_err = e
        if not _has_gemini:
            print(f"[GEMINI] Falhou ao inicializar. Último erro: {last_err}")
except Exception as e:
    _has_gemini = False
    print(f"[GEMINI] Erro de import/config: {e}")

    # Groq
    # try:
    #   from groq import Groq
    #  key = os.getenv("GROQ_API_KEY")
    # if key:
    #    _groq_client = Groq(api_key=key)
    #   _has_groq = True
    # except Exception:
    #   _has_groq = False

    _loaded = True


def available() -> bool:
    _lazy_load()
    return _has_openai or _has_gemini or _has_groq


def ask_llm(prompt: str, system: Optional[str] = None, max_tokens: int = 220) -> str:
    """
    Envia o prompt para o primeiro provedor LLM disponível.
    Prioridade: GEMINI > OPENAI > GROQ
    """
    _lazy_load()
    sys_msg = system or "Você é um assistente de EDA. Responda em português, de forma útil e direta."

    # 1️⃣ Gemini primeiro
    if _has_gemini:
        try:
            content = f"{sys_msg}\n\nUsuário:\n{prompt}"
            resp = _gemini_model.generate_content(content)
            return (getattr(resp, "text", "") or "").strip()
        except Exception as e:
            print(f"[ERRO Gemini] {e}")

    # 2️⃣ Depois OpenAI
    if _has_openai:
        try:
            resp = _openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"[ERRO OpenAI] {e}")

    # 3️⃣ Por último Groq
    if _has_groq:
        try:
            resp = _groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"[ERRO Groq] {e}")

    # Nenhum provedor funcionou
    return "⚠️ Nenhum provedor LLM pôde responder no momento."


def debug_status() -> str:
    _lazy_load()
    return (
        f"LLM disponível? {available()}\n"
        f"OpenAI: {_has_openai} (modelo={OPENAI_MODEL})\n"
        f"Gemini: {_has_gemini} (modelo={GEMINI_MODEL})\n"
        f"Groq:   {_has_groq} (modelo={GROQ_MODEL})"
    )
