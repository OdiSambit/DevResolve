"""Home / Community Feed + Experience Details."""
import streamlit as st

import config
import database
import ui_helpers as ui

engine = ui.bootstrap()

if st.session_state.get("open_experience"):
    ui.experience_detail(st.session_state["open_experience"], engine)
    st.stop()

st.title("Community Feed")
st.caption("Real-world errors, root causes and fixes shared by developers. Sample records are labelled 🧪.")

f1, f2, f3 = st.columns([2, 2, 3])
tech = f1.selectbox("Technology", ["All"] + config.TECHNOLOGIES)
cat = f2.selectbox("Error category", ["All"] + config.CATEGORIES)
q = f3.text_input("Search title or error message", placeholder="e.g. ModuleNotFoundError")

exps = database.list_experiences(technology=None if tech == "All" else tech,
                                 category=None if cat == "All" else cat, query=q, limit=100)
st.caption(f"{len(exps)} experience(s)")
for e in exps:
    ui.experience_card(e, key=f"feed_{e['id']}")
if not exps:
    st.info("No experiences match your filters. Try broadening the search or share a new experience!")
