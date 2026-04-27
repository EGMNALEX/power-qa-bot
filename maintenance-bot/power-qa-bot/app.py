"""
Streamlit 前端界面
运行方式：streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

from rag_chain import build_chain, ask

st.set_page_config(
    page_title="电力检修智能问答助手",
    page_icon="⚡",
    layout="wide",
)

# ── 自定义样式 ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0;
    }
    .sub-title {
        font-size: 0.9rem;
        color: #888;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .stChatMessage { border-radius: 12px; }
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
        min-width: 200px !important;
        max-width: 200px !important;
    }
</style>
""", unsafe_allow_html=True)

# ── 侧边栏 ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ 电力检修助手")
    st.divider()

    show_sources = st.checkbox("显示参考来源", value=False)

    st.divider()
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chain = None
        st.rerun()

    st.divider()
    st.caption("基于 RAG + GLM | 硅基流动")

# ── 主界面标题 ────────────────────────────────────────────
st.markdown('<p class="main-title">⚡ 电力检修智能问答助手</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">涵盖变压器、断路器、电缆、配电柜等检修知识</p>', unsafe_allow_html=True)

# ── 检查知识库 ────────────────────────────────────────────
faiss_exists = os.path.exists("faiss_db") and len(os.listdir("faiss_db")) > 0
if not faiss_exists:
    st.warning("⚠️ 知识库未构建，请先运行 `python ingest.py`")
    st.stop()

if not os.getenv("SILICONFLOW_API_KEY"):
    st.error("⚠️ 请在 .env 文件中配置 SILICONFLOW_API_KEY")
    st.stop()

# ── 初始化 chain ──────────────────────────────────────────
if "chain" not in st.session_state or st.session_state.chain is None:
    with st.spinner("正在加载知识库..."):
        try:
            st.session_state.chain = build_chain()
        except Exception as e:
            st.error(f"初始化失败：{e}")
            st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── 提问框（必须在页面渲染早期声明）────────────────────
chat_input = st.chat_input("请输入电力检修相关问题...")

# ── 欢迎语（无历史消息时显示）────────────────────────────
if not st.session_state.messages:
    st.markdown("#### 你好，我是电力检修智能助手 👋")
    st.markdown("你可以问我关于电力设备检修、故障处理、安全操作等问题，例如：")
    cols = st.columns(3)
    examples = [
        "变压器过热如何处理？",
        "断路器拒合闸的原因？",
        "电缆故障如何定位？",
        "SF6气体泄漏怎么办？",
        "绝缘电阻吸收比标准？",
        "停电作业安全措施？",
    ]
    for i, col in enumerate(cols):
        with col:
            if st.button(examples[i*2], use_container_width=True):
                st.session_state.example_question = examples[i*2]
                st.rerun()
            if st.button(examples[i*2+1], use_container_width=True):
                st.session_state.example_question = examples[i*2+1]
                st.rerun()

# ── 展示历史消息 ──────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if show_sources and msg.get("sources"):
            with st.expander("📄 参考来源"):
                for i, src in enumerate(msg["sources"], 1):
                    source_name = src.metadata.get("source", "未知文档")
                    page = src.metadata.get("page", "")
                    page_info = f" 第{page+1}页" if page != "" else ""
                    st.markdown(f"**[{i}] {os.path.basename(source_name)}{page_info}**")
                    st.caption(src.page_content[:200] + "...")

# ── 处理示例问题点击 ──────────────────────────────────────
if "example_question" in st.session_state:
    prompt = st.session_state.pop("example_question")
else:
    prompt = chat_input

# ── 生成回答 ──────────────────────────────────────────────
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("正在检索知识库..."):
            try:
                answer, sources = ask(st.session_state.chain, prompt)
                st.markdown(answer)

                if show_sources and sources:
                    with st.expander("📄 参考来源"):
                        for i, src in enumerate(sources, 1):
                            source_name = src.metadata.get("source", "未知文档")
                            page = src.metadata.get("page", "")
                            page_info = f" 第{page+1}页" if page != "" else ""
                            st.markdown(f"**[{i}] {os.path.basename(source_name)}{page_info}**")
                            st.caption(src.page_content[:200] + "...")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })
            except Exception as e:
                err_msg = f"回答生成失败：{e}"
                st.error(err_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": err_msg,
                })
