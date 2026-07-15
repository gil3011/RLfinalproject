"""
Shared UI scaffolding used by every room.

Keeps the app visually consistent per Plan.md:
  * a top bar = short task description + a row of KPI st.metric tiles,
  * a two-column main view (60% live board / 40% analytics tabs),
  * minimal on-screen text; all parameter explanations live in `help=` tooltips.
"""
from __future__ import annotations

import streamlit as st

ROOMS = {
    1: "Room 1 · Dynamic Programming",
    2: "Room 2 · Monte Carlo",
    3: "Room 3 · SARSA",
    4: "Room 4 · Q-Learning",
    5: "Room 5 · Deep Q-Learning",
    6: "Room 6 · Advanced DQL (Radar)",
}


def configure_page():
    st.set_page_config(
        page_title="RL Escape Room",
        page_icon="🧊",
        layout="wide",
    )


def room_selector() -> int:
    """Render the sidebar room picker and return the selected room number."""
    st.sidebar.title("🧊 RL Escape Room")
    label = st.sidebar.selectbox(
        "Room",
        options=list(ROOMS.keys()),
        format_func=lambda n: ROOMS[n],
        help="Each room escalates the RL algorithm and the difficulty of the "
        "icy environment, from full-model Dynamic Programming to radar-guided "
        "Deep Q-Learning.",
    )
    st.sidebar.divider()
    return label
