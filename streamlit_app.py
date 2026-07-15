"""
RL Escape Room — interactive Streamlit playground.

A 6-room escape game where each room escalates the reinforcement-learning
algorithm and the difficulty of a unifying slippery-ice environment. Rooms are
built and enabled one at a time.
"""
import streamlit as st

from core.layout import ROOMS, configure_page, room_selector
from rooms import room1_dp

configure_page()
room = room_selector()

if room == 1:
    room1_dp.render()
else:
    st.info(f"🚧 {ROOMS[room]} is not built yet — coming soon.")
