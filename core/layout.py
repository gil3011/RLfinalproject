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

# Rooms that are actually built and reachable from the nav bar.
NAV_ROOMS = [1, 2, 3, 4, 5]


def configure_page():
    st.set_page_config(
        page_title="RL Escape Room",
        page_icon="🧊",
        layout="wide",
    )


def _get_completed_rooms() -> set[int]:
    completed = st.session_state.get("completed_rooms", set())
    if not isinstance(completed, set):
        completed = set(completed)
        st.session_state["completed_rooms"] = completed
    return completed


def _is_room_unlocked(room_number: int) -> bool:
    if room_number == 1:
        return True
    return room_number - 1 in _get_completed_rooms()


def room_selector() -> int:
    """Render a top navigation bar for the available rooms and return the selected room number."""
    st.session_state.setdefault("selected_room", 1)
    completed_rooms = _get_completed_rooms()

    selected_room = st.session_state["selected_room"]
    if not _is_room_unlocked(selected_room):
        selected_room = 1
        st.session_state["selected_room"] = 1

    st.markdown("---")
    nav_cols = st.columns(len(NAV_ROOMS))
    for idx, room_number in enumerate(NAV_ROOMS):
        unlocked = _is_room_unlocked(room_number)
        completed = room_number in completed_rooms
        is_active = room_number == selected_room
        label = ROOMS[room_number]
        if completed:
            label = f"✅ {label}"
        elif not unlocked:
            label = f"🔒 {label}"

        with nav_cols[idx]:
            if st.button(
                label,
                key=f"nav_room_{room_number}",
                use_container_width=True,
                disabled=not unlocked,
                # Highlight the room the user is currently viewing.
                type="primary" if is_active else "secondary",
            ):
                st.session_state["selected_room"] = room_number
                st.rerun()

    st.caption("Complete each room to unlock the next one.")
    st.markdown("---")
    return st.session_state["selected_room"]


def _has_trained(room_number: int) -> bool:
    """True once the user has run training at least once in this room.

    Every room stores a `room{N}_trained_sig` in session_state the moment its
    Train button is clicked (and clears it when the board is regenerated), so
    the key's presence is a reliable "training has happened" signal.
    """
    return st.session_state.get(f"room{room_number}_trained_sig") is not None


def render_room_completion_controls(room_number: int) -> None:
    """Show the completion control for a room.

    Rendered *after* the room body so the trained-signal is already set on the
    same run the user clicks Train — no one-rerun lag before the button unlocks.
    """
    completed_rooms = _get_completed_rooms()
    completed = room_number in completed_rooms
    trained = _has_trained(room_number)

    st.markdown("---")
    if completed:
        st.success("✅ This room is marked complete.")
    elif not trained:
        st.info("🔒 Train the agent at least once to complete this room.")

    next_room = room_number + 1
    has_next = next_room in NAV_ROOMS

    col1, col2 = st.columns([3, 1])
    with col1:
        # Once complete, offer a one-click jump forward instead of making the
        # user scroll back up to the top nav bar.
        if completed and has_next and _is_room_unlocked(next_room):
            if st.button(
                f"Continue to {ROOMS[next_room]} →",
                key=f"continue_room_{room_number}",
                use_container_width=True,
            ):
                st.session_state["selected_room"] = next_room
                st.rerun()
    with col2:
        if st.button(
            "✅ Mark room complete",
            key=f"complete_room_{room_number}",
            use_container_width=True,
            type="primary",
            # Can't complete a room that's already done or hasn't been trained.
            disabled=completed or not trained,
            help=None if trained else "Run training first to unlock completion.",
        ):
            completed_rooms.add(room_number)
            st.session_state["completed_rooms"] = completed_rooms
            st.rerun()
