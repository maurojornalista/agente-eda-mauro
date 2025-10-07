# agente_core.py (Versão FINAL com Lógica Aprimorada)
from llm_client import debug_status  # se criou a função
import re
import itertools
import math  # Adicionado para cálculos de layout
from llm_client import ask_llm, available as llm_available

from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Assegure que memory.py existe com estas funções
from memory import add_qna, add_conclusion, get_conclusions

# ============================== Helpers ==============================


def _normalize_text(text: str) -> str:
    return (text or "").lower().replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")


def _is_greeting(t: str) -> bool:
    patterns = [r"\bolá\b", r"\bola\b", r"\boi\b", r"\bhello\b", r"\bhi\b", r"\bhey\b",
                r"\bbom dia\b", r"\bboa tarde\b", r"\bboa noite\b"]
    return any(re.search(p, t) for p in patterns)


def _match_columns_in_text(text: str, columns: List[str]) -> List[str]:
    t = _normalize_text(text)
    found = [c for c in columns if c.lower() in t]
    if len(found) < 2:
        quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', t)
        quoted = [q[0] or q[1] for q in quoted]
        for q in quoted:
            for c in columns:
                if c.lower() == _normalize_text(q) and c not in found:
                    found.append(c)
    if len(found) < 2:
        parts = re.split(r"(?:\s+e\s+| vs | x |,|;|:|/|\\|\s+)", t)
        for p in parts:
            p = p.strip()
            for c in columns:
                if c.lower() == p and c not in found:
                    found.append(c)
    return found


def _extract_bins(question: str, default: int = 30) -> int:
    m = re.search(r"\b(\d{1,3})\b", question or "")
    if m:
        try:
            val = int(m.group(1))
            if 3 <= val <= 200:
                return val
        except Exception:
            pass
    return default


def _pretty_hist(df: pd.DataFrame, cols: List[str], bins: int = 30):
    # Função de histograma: Agora plota MÚLTIPLAS colunas no mesmo gráfico
    plt.style.use("ggplot")
    palette = itertools.cycle(
        ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6", "#f39c12", "#16a085", "#34495e"])
    n = len(cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows))
    axes = np.array(axes).reshape(-1)
    last_i = -1
    for i, col in enumerate(cols):
        last_i = i
        ax = axes[i]
        ax.hist(df[col].dropna().values, bins=bins, alpha=0.95,
                edgecolor="white", linewidth=1.0, color=next(palette))
        ax.set_title(col, weight="bold", fontsize=12)
        ax.grid(True, alpha=0.25)
    for j in range(last_i + 1, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    return fig


def sa_boxplot_outliers(df: pd.DataFrame, cols: Optional[List[str]] = None):
    # Mantendo a função boxplot
    num = df.select_dtypes(include=[np.number])
    if cols:
        num = num[[c for c in cols if c in num.columns]]
    if num.empty:
        raise ValueError("Não há colunas numéricas para boxplots/outliers.")
    out_counts = {}
    for c in num.columns:
        x = num[c].dropna().values
        if x.size == 0:
            out_counts[c] = 0
            continue
        q1, q3 = np.percentile(x, 25), np.percentile(x, 75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out_counts[c] = int(((x < low) | (x > high)).sum())
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(num.columns)), 4.5))
    ax.boxplot([num[c].dropna().values for c in num.columns],
               labels=list(num.columns), vert=True)
    ax.set_title("Boxplots (IQR)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return fig, out_counts

# ============================== Sub-agentes ==============================


class GreetingsAgent:
    def handle(self, df: pd.DataFrame, q: str):
        if _is_greeting(_normalize_text(q)):
            return {"result": "Olá! Pronto para começar.", "kind": "greeting"}, None
        return None


class StatsAgent:
    def handle(self, df: pd.DataFrame, q: str):
        t = _normalize_text(q)

        # EXPANDIDO: Nulos
        if re.search(r"\b(nulo|nulos|null|nan|n/a|ausent[eo]s?)\b", t):
            df_nulos = df.isna().sum().to_frame("Nulos")
            total = int(df_nulos["Nulos"].sum())
            return {"result": f"Total de **{total}** nulos. Veja a distribuição por coluna abaixo.",
                    "kind": "nulls", "table": df_nulos}, None

        # EXPANDIDO: Dtypes
        if any(k in t for k in ["dtypes", "dtype", "tipo de dado", "tipos"]):
            df_dtypes = df.dtypes.astype(str).to_frame("Tipo de Dado")
            return {"result": "Tipos de dados por coluna:", "kind": "dtypes", "table": df_dtypes}, None

        # EXPANDIDO: Describe (Agora aceita 'estatística', 'desvio', 'variância', etc.)
        if any(k in t for k in ["estatística", "estatistica", "describe", "resumo", "mediana", "variancia", "desvio"]):
            df_desc = df.describe(include="all").T
            return {"result": "Estatística descritiva (describe) gerada:", "kind": "profile", "table": df_desc}, None

        # EXPANDIDO: Shape/Colunas
        if any(k in t for k in ["shape", "dimens", "linhas", "colunas", "quantas"]):
            msg = f"O dataset possui **{df.shape[0]} linhas** e **{df.shape[1]} colunas**.\n\n**Colunas:** {', '.join(df.columns)}"
            return {"result": msg, "kind": "shape_cols"}, None

        # EXPANDIDO: Média
        if any(k in t for k in ["media", "média", "mean"]):
            cols = _match_columns_in_text(q, list(df.columns))
            if cols:
                col = cols[0]
                if np.issubdtype(df[col].dtype, np.number):
                    val = float(df[col].mean())
                    return {"result": f"A média da coluna **{col}** é **{val:.4f}**.", "kind": "mean_scalar"}, None
            return {"result": "Para média, especifique uma coluna numérica (ex.: 'Qual a média de \"Amount\"?').", "kind": "text"}, None

        return None


class CorrelationAgent:
    def handle(self, df: pd.DataFrame, q: str):
        t = _normalize_text(q)
        if "correla" not in t:
            return None

        if "matriz" in t or "heatmap" in t:
            corrm = df.corr(numeric_only=True).round(4)
            if corrm.shape[0] > 1:
                ac = corrm.abs().copy()
                np.fill_diagonal(ac.values, 0.0)
                i, j = np.unravel_index(np.nanargmax(ac.values), ac.shape)
                add_conclusion(
                    f"A maior correlação absoluta é entre {corrm.index[i]} e {corrm.columns[j]} ({ac.iloc[i, j]:.4f}).")
            return {"result": "Matriz de correlação entre colunas numéricas:", "kind": "corr_matrix", "table": corrm}, None

        cols = _match_columns_in_text(q, list(df.columns))
        if len(cols) >= 2:
            c1, c2 = cols[:2]
            try:
                val = df[c1].corr(df[c2])
                return {"result": f"A correlação entre **{c1}** e **{c2}** é {val:.4f}", "kind": "corr_scalar"}, None
            except Exception as e:
                return {"error": f"Falha ao calcular correlação entre {c1} e {c2} (verifique se são numéricas): {e}"}, None

        corrm = df.corr(numeric_only=True).round(4)
        return {"result": "Não foram especificadas colunas. Matriz de correlação completa:", "kind": "corr_matrix", "table": corrm}, None


class VizAgent:
    def handle(self, df: pd.DataFrame, q: str):
        t = _normalize_text(q)
        # Adicionei 'plot'
        if any(k in t for k in ["histogram", "distribui", "histo", "plot"]):
            bins = _extract_bins(q, default=30)
            num_cols = list(df.select_dtypes("number").columns)
            cols_parsed = _match_columns_in_text(q, list(df.columns))

            # CORREÇÃO CRÍTICA: Plota as 3-4 primeiras colunas para melhor visualização
            cols = [c for c in cols_parsed if c in num_cols] or num_cols[:4]

            if not cols:
                return {"result": "Não há colunas numéricas para histogramas.", "kind": "text"}, None

            # Plota
            fig = _pretty_hist(df, cols, bins=bins)
            add_conclusion(
                f"Histogramas gerados para {len(cols)} colunas com {bins} intervalos.")
            return {"result": f"Histogramas gerados para: {', '.join(cols)}", "kind": "plot"}, fig

        if any(k in t for k in ["outlier", "boxplot", "iqr", "atipico"]):
            cols_parsed = _match_columns_in_text(q, list(df.columns))
            num_cols = list(df.select_dtypes("number").columns)
            cols = [c for c in cols_parsed if c in num_cols] or num_cols[:4]
            if not cols:
                return {"result": "Não há colunas numéricas para boxplots/outliers.", "kind": "text"}, None

            fig, out_counts = sa_boxplot_outliers(df, cols=cols)
            total_out = sum(out_counts.values())
            df_outliers = pd.Series(out_counts).to_frame("Outliers (IQR)")
            add_conclusion(
                f"Detecção de outliers (método IQR). Total de outliers: {total_out}.")
            return {"result": f"Boxplots gerados. Total de **{total_out}** outliers nas colunas analisadas.",
                    "kind": "plot_table", "table": df_outliers}, fig
        return None


class InsightsAgent:
    def handle(self, df: pd.DataFrame, q: str):
        t = _normalize_text(q)
        # Remove a checagem de 'conclus' aqui para evitar duplicar a lógica com ConclusionAgent
        if not any(k in t for k in ["insight", "tendencia", "padrao", "tendenc"]):
            return None

        if 'Class' in df.columns:
            counts = df['Class'].value_counts()
            if len(counts) > 1:
                ratio = counts.min() / counts.max()
                if ratio < 0.1:
                    add_conclusion(
                        f"ATENÇÃO: Desbalanceamento de classes severo (proporção menor/maior = {ratio:.4f}).")

        num = df.select_dtypes(include=[np.number])
        if not num.empty and num.shape[1] > 0:
            top_std = num.std().sort_values(ascending=False)
            add_conclusion(
                f"A coluna **{top_std.index[0]}** tem a maior variabilidade (desvio padrão).")

        return {"result": "Novos insights gerados e salvos na memória. Pergunte 'Quais são as conclusões?' para um resumo.", "kind": "text"}, None


class ConclusionAgent:
    def handle(self, df: pd.DataFrame, q: str):
        t = _normalize_text(q)
        # Mantém apenas a checagem de 'conclus' para ser o agente RESPONSÁVEL
        if "conclus" in t:
            conc = get_conclusions()
            if not conc:
                return {"result": "Ainda não há conclusões salvas. Rode análises (nulos, correlação, histogramas) primeiro.", "kind": "text"}, None

            # Melhora a apresentação para ser mais amigável, como você pediu
            msg = "✅ **Resumo das Conclusões do Agente:**\n\n" + \
                "\n".join([f"- {c}" for c in conc])
            return {"result": msg, "kind": "summary"}, None
        return None


class DebugAgent:
    def handle(self, df: pd.DataFrame, q: str):
        if "debug llm" in (q or "").lower():
            return {"result": f"```\n{debug_status()}\n```", "kind": "text"}, None
        return None


class FallbackAgent:
    def handle(self, df: pd.DataFrame, q: str):
        # Se houver LLM configurada, usa; senão, responde com sugestão guiada
        if llm_available():
            cols = ", ".join(df.columns)
            prompt = (
                f"Contexto do dataset: {df.shape[0]} linhas, {df.shape[1]} colunas: {cols}.\n"
                f"Pergunta do usuário: {q}\n"
                "Responda em português, de forma útil e prática. "
                "Se for necessário calcular ou plotar algo específico, sugira a pergunta exata que devo fazer ao agente (ex.: "
                "\"Mostre a matriz de correlação\", \"Histogramas de Amount\", \"Quais são os nulos?\")."
            )
            text = ask_llm(prompt)
            if text:
                return {"result": f"🧠 *(LLM)* {text}", "kind": "llm_response"}, None

        # Sem LLM → ajuda guiada (não quebra nada)
        return {
            "result": (
                "Não entendi sua pergunta em linguagem natural. Tive um problema ao falar com a LLM agora.\n"
                "Tente, por exemplo:\n"
                "- Existem nulos?\n"
                "- Quais os dtypes?\n"
                "- Matriz de correlação\n"
                "- Histogramas de \"Amount\"\n"
                "- Quais são as conclusões?"
            ),
            "kind": "help"
        }, None

# ============================== Orquestrador ==============================


class Orchestrator:
    def __init__(self):
        # Ordem de prioridade (ConclusionAgent é crucial)
        self.agents = [
            DebugAgent(),
            GreetingsAgent(),
            ConclusionAgent(),
            CorrelationAgent(),
            VizAgent(),
            InsightsAgent(),
            StatsAgent()
        ]
        self.fallback = FallbackAgent()

    def ask(self, df: pd.DataFrame, question: str) -> Tuple[Dict[str, Any], Optional[plt.Figure]]:
        for agent in self.agents:
            out = agent.handle(df, question)
            if out is not None:
                return out
        return self.fallback.handle(df, question)

# ============================== API pública ==============================


def run_agent(question: str, data: Optional[pd.DataFrame]):
    if data is None:
        return {"error": "Envie um CSV primeiro."}, None
    import os
    print(
        f"Executando agente: Chave={os.getenv('GROQ_API_KEY')}, Pergunta={question}")
    orch = Orchestrator()
    print(f"Orchestrator criado, chamando ask com pergunta: {question}")
    out, fig = orch.ask(data, question)
    print(f"Resposta do Orchestrator: {out}")
    try:
        add_qna(question, str(out.get("result", "Resposta do agente.")))
    except Exception:
        pass
    return out, fig
