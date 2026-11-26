"""My Contributions - experiences submitted by the signed-in user."""
import streamlit as st

import database
import ui_helpers as ui

engine = ui.bootstrap()

st.title("My Contributions")
user = ui.current_user()
if not ui.require_login("see your contributions"):
    st.stop()

mine = database.list_experiences(user_id=user["id"], limit=200)
accepted = sum(1 for e in mine if e["has_accepted"])
c1, c2, c3 = st.columns(3)
c1.metric("Experiences shared", len(mine))
c2.metric("With accepted solution", accepted)
c3.metric("Community solutions received", sum(e["solution_count"] for e in mine))

for e in mine:
    ui.experience_card(e, key=f"mine_{e['id']}")
if not mine:
    st.info("You haven't shared anything yet. Head to **Share Experience** to document your first fix!")
