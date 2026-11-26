"""AI Troubleshooter - RAG answers grounded in community experiences, with citations."""
import streamlit as st

import config
import ui_helpers as ui

engine = ui.bootstrap()

st.title("AI Troubleshooter")
st.caption("Answers are generated from retrieved community experiences and cite their sources. "
           "Community solutions are **unverified** unless marked accepted - never run commands blindly.")

if not ui.llm_available():
    st.info("🟠 Ollama isn't running, so I'll return the most relevant community experiences instead of a generated answer. "
            f"To enable AI answers: install Ollama, run `ollama pull {config.OLLAMA_MODEL}`, and keep it running.")

with st.expander("Narrow the search (optional)"):
    c1, c2 = st.columns(2)
    tech = c1.selectbox("Technology", ["Any"] + config.TECHNOLOGIES, key="ai_tech")
    osys = c2.selectbox("Operating system", ["Any"] + config.OPERATING_SYSTEMS, key="ai_os")

history = st.session_state.setdefault("chat_history", [])
if history and st.button("Clear conversation"):
    history.clear()
    st.rerun()


def render_sources(sources, cited):
    if not sources:
        return
    st.markdown("**Community evidence**")
    for s in sources:
        mark = " ← cited" if s["id"] in cited else ""
        accepted = " · ✅ accepted solution" if s["has_accepted"] else " · unverified"
        with st.expander(f"Experience #{s['id']} – {s['title'][:80]}{mark}  (relevance {s['score']:.2f}{accepted})"):
            st.caption(f"{s['technology']} {s['version']} · {s['operating_system']}" + (" · 🧪 sample record" if s["is_seed"] else ""))
            st.code(s["error_message"], language=None, wrap_lines=True)
            st.markdown("**Root cause**")
            st.text(s["root_cause"])
            st.markdown("**Solution**")
            st.text(s["working_solution"])
            if st.button("Open full experience", key=f"src_{len(history)}_{s['id']}_{id(sources)}"):
                ui.open_experience(s["id"])


for turn in history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant":
            if turn.get("notice"):
                st.caption(f"ℹ️ {turn['notice']}")
            render_sources(turn.get("sources", []), turn.get("cited", []))

question = st.chat_input("Describe your error, e.g. 'ModuleNotFoundError for pandas even after pip install'")
if question:
    history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving community experiences and generating an answer..."):
            result = engine.answer(question, technology=None if tech == "Any" else tech,
                                   operating_system=None if osys == "Any" else osys)
        st.markdown(result.answer)
        if result.notice:
            st.caption(f"ℹ️ {result.notice}")
        if result.mode == "llm":
            st.caption("Generated from retrieved community experiences. Verify before applying.")
    history.append({"role": "assistant", "content": result.answer, "notice": result.notice,
                    "sources": result.sources, "cited": result.cited_ids})
    st.rerun()
