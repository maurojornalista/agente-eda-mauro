import json, os, time
from typing import List, Dict, Any

MEMO_PATH = os.environ.get("AGENT_MEMORY_PATH", "agent_memory.json")

def _load() -> Dict[str, Any]:
    if not os.path.exists(MEMO_PATH):
        return {"conclusions": [], "qna": []}
    try:
        with open(MEMO_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"conclusions": [], "qna": []}

def _save(mem: Dict[str, Any]) -> None:
    with open(MEMO_PATH, "w", encoding="utf-8") as f:
        json.dump(mem, f, ensure_ascii=False, indent=2)

def add_qna(question: str, answer: str):
    mem = _load()
    mem["qna"].append({"ts": int(time.time()), "q": question, "a": answer})
    _save(mem)

def add_conclusion(text: str):
    mem = _load()
    mem["conclusions"].append({"ts": int(time.time()), "text": text})
    _save(mem)

def get_conclusions() -> List[str]:
    mem = _load()
    return [c["text"] for c in mem.get("conclusions", [])]

def dump_memory() -> dict:
    return _load()
