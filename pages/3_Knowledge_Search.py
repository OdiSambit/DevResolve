"""Knowledge Search - semantic (FAISS), keyword and filtered search."""
import streamlit as st

import config
import database
import ui_helpers as ui

engine = ui.bootstrap()

st.title("Knowledge Search")
st.caption("Describe your problem in your own words - similar documented problems are found even if the wording differs.")

mode = st.radio("Search type", ["Semantic", "Keyword"], horizontal=True,
                help="Semantic finds conceptually similar problems; Keyword finds exact terms.")
query = st.text_area("Your problem or error message", height=90,
                     placeholder="Why is Python unable to import pandas even though I installed it?")
f1, f2, f3 = st.columns(3)
tech = f1.selectbox("Technology", ["All"] + config.TECHNOLOGIES, key="s_tech")
osys = f2.selectbox("Operating system", ["All"] + config.OPERATING_SYSTEMS, key="s_os")
cat = f3.selectbox("Error category", ["All"] + config.CATEGORIES, key="s_cat")
pick = lambda v: None if v == "All" else v

if st.button("Search", type="primary") and query.strip():
    if mode == "Semantic":
        results = engine.search(query, config.TOP_K, technology=pick(tech),
                                operating_system=pick(osys), category=pick(cat))
        st.session_state["search_results"] = [(r, r["score"]) for r in results]
    else:
        results = database.list_experiences(technology=pick(tech), operating_system=pick(osys),
                                            category=pick(cat), query=query, limit=10)
        st.session_state["search_results"] = [(r, None) for r in results]
    st.session_state["search_query"] = query

if "search_results" in st.session_state:
    shown = st.session_state["search_results"]
    st.subheader(f"{len(shown)} result(s)")
    for i, (exp, score) in enumerate(shown):
        ui.experience_card(exp, key=f"search_{i}_{exp['id']}", score=score)
    if not shown:
        st.info("No matching experiences. Try the exact error text, or share this problem as a new experience.")
    elif shown[0][1] is not None and shown[0][1] < engine.min_score:
        st.caption("These matches are weak - consider adding the exact error message or the technology version.")
