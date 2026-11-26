"""DevResolve - entry point. Run with:  streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="DevResolve", page_icon="🛠️", layout="wide")

import ui_helpers as ui  # noqa: E402  (must come after set_page_config)

engine = ui.bootstrap()
ui.sidebar_account(engine)

navigation = st.navigation([
    st.Page("pages/1_Home.py", title="Home", icon="🏠", default=True),
    st.Page("pages/2_Share_Experience.py", title="Share Experience", icon="✍️"),
    st.Page("pages/3_Knowledge_Search.py", title="Knowledge Search", icon="🔎"),
    st.Page("pages/4_AI_Troubleshooter.py", title="AI Troubleshooter", icon="🤖"),
    st.Page("pages/5_My_Contributions.py", title="My Contributions", icon="📚"),
])
navigation.run()
