"""
RL Escape Room — interactive Streamlit playground.

A 6-room escape game where each room escalates the reinforcement-learning
algorithm and the difficulty of a unifying slippery-ice environment. Rooms are
built and enabled one at a time.
"""
import streamlit as st

from core.layout import ROOMS, configure_page, render_room_completion_controls, room_selector
from rooms import room1_dp, room2_mc, room3_sarsa, room4_qlearning, room5_dqn

configure_page()
room = room_selector()

if room == 1:
    room1_dp.render()
    render_room_completion_controls(1)
elif room == 2:
    room2_mc.render()
    render_room_completion_controls(2)
elif room == 3:
    room3_sarsa.render()
    render_room_completion_controls(3)
elif room == 4:
    room4_qlearning.render()
    render_room_completion_controls(4)
elif room == 5:
    room5_dqn.render()
    render_room_completion_controls(5)
else:
    st.info(f"🚧 {ROOMS[room]} is not built yet — coming soon.")
