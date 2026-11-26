"""Shared Streamlit helpers: startup, session user, escaping and reusable UI components."""
import re

import streamlit as st

import auth
import config
import database
from rag_engine import OllamaGenerator, RAGEngine
from seed_data import seed_if_empty
from validators import ValidationError, detect_secrets

_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$:&])")


def escape_md(text) -> str:
    """Escape user text so it renders literally inside st.markdown / st.caption."""
    return _MD_SPECIAL.sub(r"\\\1", str(text or ""))


def truncate(text, n=220) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


# ----------------------------------------------------------------- startup
@st.cache_resource(show_spinner="Starting DevResolve - preparing the knowledge index...")
def _startup() -> RAGEngine:
    database.configure()
    seed_if_empty()
    return RAGEngine()


def bootstrap() -> RAGEngine:
    return _startup()


@st.cache_data(ttl=30, show_spinner=False)
def llm_available() -> bool:
    return OllamaGenerator().available()


# ------------------------------------------------------------ session user
def current_user():
    return st.session_state.get("user")


def logout():
    st.session_state.pop("user", None)


def require_login(action="do this") -> bool:
    if current_user():
        return True
    st.info(f"Please log in or register from the sidebar to {action}.")
    return False


def sidebar_account(engine: RAGEngine):
    with st.sidebar:
        st.markdown("## 🛠️ DevResolve")
        st.caption("Share the error. Explain the fix. Help the next developer.")
        user = current_user()
        if user:
            st.success(f"Signed in as **{escape_md(user['username'])}**")
            if st.button("Log out", use_container_width=True):
                logout()
                st.rerun()
        else:
            tab_in, tab_up = st.tabs(["Log in", "Register"])
            with tab_in:
                with st.form("login_form"):
                    ident = st.text_input("Username or email")
                    pw = st.text_input("Password", type="password")
                    if st.form_submit_button("Log in", use_container_width=True):
                        try:
                            st.session_state["user"] = auth.login(ident, pw)
                            st.rerun()
                        except ValidationError as e:
                            st.error(e.errors[0])
            with tab_up:
                with st.form("register_form"):
                    name = st.text_input("Username")
                    email = st.text_input("Email")
                    pw2 = st.text_input("Password (min 8 chars)", type="password")
                    if st.form_submit_button("Create account", use_container_width=True):
                        try:
                            st.session_state["user"] = auth.register(name, email, pw2)
                            st.rerun()
                        except ValidationError as e:
                            for msg in e.errors:
                                st.error(msg)
        st.divider()
        status = engine.status()
        st.caption(f"Index: **{status['indexed']}/{status['total']}** experiences · embedder `{status['embedder']}`")
        if status["pending"]:
            st.warning(f"{status['pending']} experience(s) are saved but not yet indexed.")
            if st.button("Retry indexing", use_container_width=True):
                done, left = engine.index_pending()
                st.toast(f"Indexed {done}; {left} still pending.")
                st.rerun()
        st.caption("LLM: " + ("🟢 Ollama ready" if llm_available() else "🟠 Ollama not detected (retrieval-only mode)"))
        st.caption("⚠️ Local prototype - not a hardened multi-user service. "
                   "Never post passwords, API keys, tokens or confidential logs.")


# ---------------------------------------------------------------- components
def open_experience(experience_id: int):
    st.session_state["open_experience"] = experience_id
    st.switch_page("pages/1_Home.py")


def experience_card(exp: dict, key: str, score=None, navigate=True):
    with st.container(border=True):
        tags = f"**{escape_md(exp['technology'])}** · {escape_md(exp['category'])}"
        if exp.get("is_seed"):
            tags += " · 🧪 *sample record*"
        if score is not None:
            tags += f" · relevance **{score:.2f}**"
        st.markdown(tags)
        st.markdown(f"### {escape_md(exp['title'])}")
        ver = f" {exp['version']}" if exp["version"] else ""
        st.caption(escape_md(f"{exp['technology']}{ver} · {exp['operating_system']}"))
        st.code(truncate(exp["error_message"], 300), language=None, wrap_lines=True)
        st.caption(escape_md(truncate(exp["description"], 240)))
        c1, c2, c3 = st.columns([2, 2, 1])
        n = exp.get("solution_count", 0)
        c1.caption(f"💬 {n} solution{'s' if n != 1 else ''}" + ("  ·  ✅ Accepted solution" if exp.get("has_accepted") else ""))
        c2.caption(f"{exp['created_at']:%b %d, %Y} · by {escape_md(exp['author'])}")
        if navigate and c3.button("Open", key=f"open_{key}", use_container_width=True):
            open_experience(exp["id"])


def secrets_notice():
    st.warning("🔐 **Do not post** passwords, API keys, access tokens, private keys or confidential logs. "
               "Redact them (e.g. `<API_KEY>`) before submitting.")


def experience_detail(exp_id: int, engine: RAGEngine):
    exp = database.get_experience(exp_id)
    if not exp:
        st.error("Experience not found.")
        return
    if st.button("← Back to community feed"):
        st.session_state.pop("open_experience", None)
        st.rerun()

    st.markdown(f"## {escape_md(exp['title'])}")
    badges = f"**{escape_md(exp['technology'])}** {escape_md(exp['version'])} · {escape_md(exp['operating_system'])} · {escape_md(exp['category'])}"
    if exp["is_seed"]:
        badges += " · 🧪 *sample record (seeded, not a real community post)*"
    st.markdown(badges)
    st.caption(f"Experience #{exp['id']} · contributed by {escape_md(exp['author'])} on {exp['created_at']:%b %d, %Y}")

    st.markdown("**Exact error**")
    st.code(exp["error_message"], language=None, wrap_lines=True)
    st.markdown("**Problem description**")
    st.text(exp["description"])
    st.markdown("**Root cause**")
    st.info(exp["root_cause"])
    st.markdown("**Working solution** *(reported by the author - unverified unless a solution below is accepted)*")
    st.success(exp["working_solution"])

    st.divider()
    st.markdown(f"### Community solutions ({len(exp['solutions'])})")
    user = current_user()
    for sol in exp["solutions"]:
        with st.container(border=True):
            head = f"by **{escape_md(sol['author'])}** · {sol['created_at']:%b %d, %Y}"
            st.markdown(head + ("  ·  ✅ **Accepted**" if sol["is_accepted"] else ""))
            st.text(sol["solution"])
            if user and user["id"] == exp["user_id"] and not sol["is_accepted"]:
                if st.button("Accept this solution", key=f"accept_{sol['id']}"):
                    try:
                        database.accept_solution(sol["id"], user["id"])
                        engine.index_experience(exp["id"])       # accepted fix enriches the knowledge record
                        st.rerun()
                    except ValidationError as e:
                        st.error(e.errors[0])
    if not exp["solutions"]:
        st.caption("No alternative solutions yet.")

    st.markdown("**Propose a solution**")
    if require_login("propose a solution"):
        with st.form(f"solution_form_{exp['id']}", clear_on_submit=True):
            text = st.text_area("Your solution", height=120, placeholder="Describe what fixed it for you...")
            if st.form_submit_button("Submit solution"):
                try:
                    database.add_solution(exp["id"], user["id"], text)
                    st.success("Solution added.")
                    st.rerun()
                except ValidationError as e:
                    for msg in e.errors:
                        st.error(msg)
