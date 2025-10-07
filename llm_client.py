# llm_client.py — cliente LLM (Gemini em 1º), com OpenAI opcional
import os
from typing import Optional

# Em dev/local, .env ajuda; no Render, as env vars já vêm do painel
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- Flags / estado ---
_loaded = False
_has_openai = False
_has_gemini = False

_openai_client = None
_gemini_client = None
_gemini_model_name = None  # guarda o nome realmente usado

# Modelos (podem ser sobrescritos via env)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
# Use um modelo válido da API recente:
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-pro-preview-03-25")


def _lazy_load():
    """Carrega clientes uma única vez, de forma segura em produção."""
    global _loaded, _has_openai, _has_gemini
    global _openai_client, _gemini_client, _gemini_model_name

    if _loaded:
        return

    # ---------- OpenAI (opcional) ----------
    try:
        from openai import OpenAI
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        if openai_key:
            _openai_client = OpenAI(api_key=openai_key)
            _has_openai = True
    except Exception as e:
        _has_openai = False
        print(f"[OpenAI] desativado: {e}")

    # ---------- Gemini (principal) ----------
    try:
        import google.generativeai as genai
        gkey = os.getenv("GOOGLE_API_KEY", "").strip()
        if gkey:
            genai.configure(api_key=gkey)
            # Tenta na ordem: env > fallback(s)
            candidates = [
                GEMINI_MODEL,  # do env se existir
                "models/gemini-2.5-pro-preview-03-25",
                "models/gemini-1.5-flash",
            ]
            last_err = None
            for name in candidates:
                try:
                    _gemini_client = genai.GenerativeModel(name)
                    _gemini_model_name = name
                    _has_gemini = True
                    break
                except Exception as e:
                    last_err = e
            if not _has_gemini:
                print(
                    f"[GEMINI] Falhou ao criar modelo. Último erro: {last_err}")
        else:
            print("[GEMINI] GOOGLE_API_KEY vazia/ausente.")
    except Exception as e:
        _has_gemini = False
        print(f"[GEMINI] Erro import/config: {e}")

    _loaded = True


def available() -> bool:
    _lazy_load()
    return _has_gemini or _has_openai


def ask_llm(prompt: str, system: Optional[str] = None, max_tokens: int = 220) -> str:
    """
    Envia o prompt para o primeiro provedor disponível.
    Prioridade: GEMINI > OPENAI
    """
    _lazy_load()
    sys_msg = system or "Você é um assistente de EDA. Responda em português, de forma útil e direta."

    # 1) Gemini
    if _has_gemini and _gemini_client is not None:
        try:
            content = f"{sys_msg}\n\nUsuário:\n{prompt}"
            resp = _gemini_client.generate_content(content)
            return (getattr(resp, "text", "") or "").strip()
        except Exception as e:
            print(f"[ERRO Gemini] {e}")

    # 2) OpenAI
    if _has_openai and _openai_client is not None:
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

    # nenhum provedor
    return "⚠️ Nenhum provedor LLM pôde responder no momento."


def debug_status() -> str:
    _lazy_load()
    gem_model = _gemini_model_name or GEMINI_MODEL
    return (
        f"LLM disponível? {available()}\n"
        f"OpenAI: {_has_openai} (modelo={OPENAI_MODEL})\n"
        f"Gemini: {_has_gemini} (modelo={gem_model})"
    )
