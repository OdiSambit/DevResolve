"""Share Experience - structured submission form."""
import streamlit as st

import config
import database
import ui_helpers as ui
from validators import ValidationError, detect_secrets

engine = ui.bootstrap()

st.title("Share an Experience")
st.caption("Document a real error, its root cause and the fix that worked - so the next developer saves time.")
ui.secrets_notice()

user = ui.current_user()
if not ui.require_login("share an experience"):
    st.stop()

with st.form("share_form", clear_on_submit=False):
    title = st.text_input("Problem title *", placeholder="ModuleNotFoundError: No module named pandas")
    c1, c2, c3 = st.columns(3)
    technology = c1.selectbox("Technology *", config.TECHNOLOGIES)
    version = c2.text_input("Technology version", placeholder="3.12")
    operating_system = c3.selectbox("Operating system *", config.OPERATING_SYSTEMS)
    category = st.selectbox("Error category *", config.CATEGORIES)
    error_message = st.text_area("Exact error message *", height=110, placeholder="Paste the exact error text (redact secrets)")
    description = st.text_area("Problem description *", height=120, placeholder="What were you doing, and what happened?")
    root_cause = st.text_area("Root cause *", height=100, placeholder="Why did the error happen?")
    working_solution = st.text_area("Working solution *", height=120, placeholder="Exactly what fixed it?")
    submitted = st.form_submit_button("Publish experience", type="primary")

if submitted:
    data = dict(title=title, technology=technology, version=version, operating_system=operating_system,
                category=category, error_message=error_message, description=description,
                root_cause=root_cause, working_solution=working_solution)
    _, soft = detect_secrets("\n".join(str(v) for v in data.values()))
    for hint in soft:
        st.warning(f"Heads up: your text may contain {hint}. Double-check nothing confidential is included.")
    try:
        exp = database.create_experience(user["id"], data)         # 1. store in SQLite
    except ValidationError as e:
        for msg in e.errors:
            st.error(msg)
    else:
        ok = engine.index_experience(exp["id"])                    # 2-4. knowledge record + embedding + FAISS
        if ok:
            st.success(f"✅ Experience #{exp['id']} published and added to the searchable knowledge base.")
        else:
            st.warning(f"Experience #{exp['id']} was saved, but indexing failed. "
                       "Use 'Retry indexing' in the sidebar to make it searchable.")
        if st.button("View it in the feed"):
            ui.open_experience(exp["id"])
