
# app.py (Versão FINAL e Definitiva: Fluxo UX Restaurado e Estável)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Tuple
import re
import io
import os
from agente_core import run_agent
from memory import get_conclusions
from dotenv import load_dotenv
load_dotenv()


load_dotenv()


# Importa o agente principal e a função de memória

# ==============================
# Config e Header
# ==============================
st.set_page_config(
    page_title="Agentes Autônomos de Inteligência Artificial — I2A2",
    page_icon="🤖",
    layout="wide",
)

TITLE = "Agentes Autônomos de Inteligência Artificial — I2A2"
AUTHOR = "Mauro de Oliveira"

# ------------------ Estado ------------------
if "df" not in st.session_state:
    st.session_state.df: Optional[pd.DataFrame] = None
if "log" not in st.session_state:
    st.session_state.log: List[Dict[str, Any]] = []
if "user_name" not in st.session_state:
    st.session_state.user_name: Optional[str] = None
if "ask_name_submitted" not in st.session_state:
    st.session_state.ask_name_submitted = False
if "initial_greeting_sent" not in st.session_state:
    st.session_state.initial_greeting_sent = False
if "csv_uploaded_once" not in st.session_state:
    # Novo flag para controlar a saudação pós-CSV
    st.session_state.csv_uploaded_once = False

# ------------------ Helpers ------------------


def polite_greeting() -> str:
    nome = st.session_state.user_name or "usuário"
    return f"Olá, {nome}! Você pode me perguntar sobre nulos, correlação, histogramas ou conclusões."


def polite_farewell(question: str) -> str:
    nome = st.session_state.user_name
    t = (question or "").lower()
    if any(p in t for p in ["obrigado", "obrigada", "valeu"]):
        return (f"De nada, **{nome}**! Fico feliz em ajudar. 👋") if nome else "De nada! Fico feliz em ajudar. 👋"
    return (f"Até mais, **{nome}**! Qualquer coisa, é só chamar. 👋") if nome else "Até mais! Qualquer coisa, é só chamar. 👋"


def get_example_questions() -> str:
    # A lista de perguntas que o outro bot removeu (e que você quer de volta!)
    return (
        "Aqui estão exemplos de algumas perguntas que você pode fazer:\n"
        "• Existem nulos?\n"
        "• Quais os dtypes?\n"
        "• Quantas linhas e colunas tem?\n"
        "• Mostre a matriz de correlação\n"
        "• Gere histogramas com 15 intervalos\n"
        "• Qual a média da coluna 'V1'?\n"
        "• **Quais são as conclusões?** (Pergunta obrigatória)"
    )


def build_markdown_report() -> str:
    df = st.session_state.df
    lines = [f"# {TITLE}", f"**Autor:** {AUTHOR}", "", "## 1. Stack",
             "- Python, Streamlit (UI), Pandas/Numpy (EDA), Matplotlib (gráficos).", "", "## 2. Dataset",]
    if df is not None:
        lines.append(
            f"- Linhas: **{df.shape[0]}** | Colunas: **{df.shape[1]}**")
        lines.append(f"- Colunas: {', '.join(map(str, df.columns))}")
    else:
        lines.append("- (Nenhum CSV carregado.)")

    lines += ["",
              "## 3. Perguntas-Exemplo e Respostas (trechos do histórico)",]
    shown = 0
    for i, item in enumerate(st.session_state.log):
        if item.get("role") == "user" and shown < 4:
            if i + 1 < len(st.session_state.log):
                nxt = st.session_state.log[i + 1]
                if nxt.get("role") == "assistant" and nxt.get("out") and "error" not in nxt["out"]:
                    lines.append(f"- **Pergunta**: {item.get('q', '')}")
                    ans = str(nxt['out'].get(
                        'result', 'Resposta complexa (ver gráfico/tabela).'))
                    lines.append(
                        f"  **Resposta**: {ans[:120]}{'...' if len(ans) > 120 else ''}")
                    shown += 1

    lines += ["", "## 4. Conclusões do agente (Resposta Obrigatória)"]
    try:
        conclusions = get_conclusions()
    except Exception:
        conclusions = []

    if conclusions:
        for c in conclusions:
            lines.append(f"- {c}")
    else:
        lines.append(
            "- Sem conclusões salvas. Rode análises (correlação/insights).")

    lines += ["", "---", "_Relatório gerado automaticamente pelo agente._"]
    return "\n".join(lines)


def render_assistant_message(out: Dict[str, Any], fig_bytes: Optional[bytes]):
    """Renderiza a saída do agente."""
    if not out:
        return

    if "error" in out:
        st.error(out["error"])
    elif "table" in out:
        st.markdown(out.get("result", ""))
        st.dataframe(out["table"], width="stretch")
    elif "result" in out:
        st.markdown(out["result"])

    if fig_bytes:
        st.image(fig_bytes, width="stretch")

# ------------------ Layout Principal ------------------


st.subheader(TITLE)
st.caption(f"Autor: **{AUTHOR}**")
st.markdown("---")


# --- Sidebar ---
with st.sidebar:
    st.header("Entrada de Dados")
    csv = st.file_uploader("Envie um CSV", type=["csv"])

    # Carregar CSV (Lógica de upload aprimorada)
    if csv is not None:
        try:
            df = pd.read_csv(csv)
            # Verifica se é um NOVO CSV
            if st.session_state.df is None or st.session_state.df.shape != df.shape or not st.session_state.df.head(1).equals(df.head(1)):
                st.session_state.df = df
                st.session_state.log.clear()  # Limpa o log se um novo arquivo for carregado
                st.session_state.initial_greeting_sent = False  # Reseta a saudação
                st.session_state.csv_uploaded_once = True  # Marca que o upload foi feito

            st.success(
                f"CSV carregado: **{df.shape[0]} linhas** × **{df.shape[1]} colunas**.")
            st.dataframe(df.head(), use_container_width=True,
                         height=180)  # Prévia na Sidebar
        except Exception as e:
            st.error(f"Erro ao ler CSV: {e}")
            st.session_state.df = None
    else:
        st.session_state.df = st.session_state.get("df")

    st.markdown("---")
    # LISTA DE PERGUNTAS NA SIDEBAR (Para referência rápida)
    st.markdown("**Exemplos de perguntas:**")
    st.markdown(get_example_questions())
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 Limpar Conversa", use_container_width="stretch"):
            st.session_state.log.clear()
            st.session_state.initial_greeting_sent = False
            st.rerun()
    with col2:
        if st.button("🔄 Recarregar App", use_container_width="stretch"):
            st.rerun()

# --- Tratamento de Nome (Formulário no Início) ---
if st.session_state.user_name is None and not st.session_state.ask_name_submitted:
    with st.container(border=True):
        st.markdown("👋 **Bem-vindo! Antes de começar, como devo te chamar?**")
        with st.form("name_form"):
            name_input = st.text_input(
                "Seu nome", placeholder="Ex.: Mauro de Oliveira")
            submitted = st.form_submit_button("Confirmar")
            if submitted:
                if name_input:
                    st.session_state.user_name = name_input.strip()
                    st.session_state.ask_name_submitted = True
                    st.rerun()
                else:
                    st.error("Por favor, digite seu nome.")

    if st.session_state.user_name is None:
        st.stop()


# --- Saudações e Mensagens Iniciais no Chat Log ---

# 1. Saudação Inicial (Após o nome)
if len(st.session_state.log) == 0 and st.session_state.user_name and not st.session_state.csv_uploaded_once:
    # Mensagem Única com instrução para subir o CSV
    initial_msg = (
        f"Olá, **{st.session_state.user_name}**! 👋 Sou seu Agente de EDA.\n\n"
        f"**Para começar**, por favor, **suba um arquivo CSV na barra lateral**."
    )
    st.session_state.log.append(
        {"role": "assistant", "out": {"result": initial_msg}})
    st.session_state.initial_greeting_sent = True
# Se for a primeira saudação, manda também exemplos
if len(st.session_state.log) == 0 and not st.session_state.initial_greeting_sent:
    st.session_state.log.append({"role": "assistant", "out": {
        "result": f"Olá, **{st.session_state.user_name}**! 👋 "
        f"Sou seu Agente de EDA. Por favor, envie um CSV na barra lateral e comece a perguntar."
    }})

    exemplos = (
        "Aqui estão algumas perguntas que você pode fazer:\n"
        "- Existem nulos?\n"
        "- Quais os dtypes?\n"
        "- Quantas linhas e colunas tem?\n"
        "- Mostre a matriz de correlação\n"
        "- Gere histogramas com 15 intervalos\n"
        "- Quais são as conclusões?"
    )
    st.session_state.log.append(
        {"role": "assistant", "out": {"result": exemplos}})
    st.session_state.initial_greeting_sent = True

# 2. Mensagem Pós-Upload (Com a lista de perguntas)
if st.session_state.csv_uploaded_once and len(st.session_state.log) == 1:
    # Limpa o log e coloca a mensagem de pronto para perguntar
    st.session_state.log.clear()

    post_upload_msg = (
        f"✅ **CSV carregado e pronto para análise, {st.session_state.user_name}!**\n\n"
        f"Agora você pode me fazer perguntas.\n\n"
        f"{get_example_questions()}"
    )
    st.session_state.log.append(
        {"role": "assistant", "out": {"result": post_upload_msg}})
    st.session_state.csv_uploaded_once = False  # Reseta para não entrar mais aqui
    st.rerun()


# --- Histórico e Renderização ---
for item in st.session_state.log:
    role = item.get("role", "user")
    with st.chat_message(role):
        if role == "user":
            st.write(item.get("q", ""))
        else:
            render_assistant_message(
                item.get("out", {}), item.get("fig_bytes"))

# --- Relatório ---
with st.expander("📄 Gerar relatório (Markdown)"):
    if st.session_state.df is None:
        st.info("Envie um CSV para gerar o relatório.")
    else:
        md = build_markdown_report()
        st.code(md, language="markdown")
        st.download_button("⬇️ Baixar relatório.md", data=md.encode(
            "utf-8"), file_name="relatorio_agente_i2a2.md")


# --- Input de Chat (rodapé fixo) ---
user_text = st.chat_input(polite_greeting())

if user_text:

    # 1. Registro da pergunta do usuário
    st.session_state.log.append({"role": "user", "q": user_text})
    df = st.session_state.df

    # 2. Resposta rápida (saudação/despedida)
    if re.search(r"\b(tchau|obrigad[ao]|valeu|bye|até\s+logo|até\s+mais)\b", user_text.lower()):
        st.session_state.log.append(
            {"role": "assistant", "out": {"result": polite_farewell(user_text)}})
        st.rerun()

    # 3. Tratamento de CSV ausente
    if df is None:
        st.session_state.log.append({"role": "assistant", "out": {
                                    "result": "Para te responder, por favor envie um arquivo **CSV** na barra lateral. 😉"}, })
        st.rerun()

    # 4. Execução do Agente Principal
    with st.spinner("Gerando resposta..."):
        try:
            out, fig = run_agent(user_text, df)
        except Exception as e:
            out = {"error": f"Erro inesperado no Agente Core: {e}"}
            fig = None

    # 5. Converte figura para bytes para salvar no log (histórico)
    fig_bytes = None
    if fig is not None:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        fig_bytes = buf.read()
        plt.close(fig)

    # 6. Salva a resposta no log e recarrega
    st.session_state.log.append(
        {"role": "assistant", "out": out, "fig_bytes": fig_bytes})
    st.rerun()
