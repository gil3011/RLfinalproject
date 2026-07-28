"""
Shared UI scaffolding used by every room.

Keeps the app visually consistent per Plan.md:
  * a top navigation bar of the six rooms (all freely accessible — no gating),
  * a "Continue to the next room" button at the bottom of each room,
  * minimal on-screen text; all parameter explanations live in `help=` tooltips.

Rooms are NOT locked: the user can jump to any room at any time, and does not need
to train or "complete" a room to move on. Navigating (via the nav bar or the
Continue button) scrolls back to the top of the newly selected room.
"""
from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

ROOMS = {
    1: "Room 1 · Dynamic Programming",
    2: "Room 2 · Monte Carlo",
    3: "Room 3 · SARSA",
    4: "Room 4 · Q-Learning",
    5: "Room 5 · Deep Q-Learning",
    6: "Room 6 · Advanced DQL (Radar)",
}

# Rooms that are actually built and reachable from the nav bar.
NAV_ROOMS = [1, 2, 3, 4, 5, 6]


def configure_page():
    st.set_page_config(
        page_title="RL Escape Room",
        page_icon="🧊",
        layout="wide",
    )


def _scroll_to_top(nonce: int) -> None:
    """Scroll the app back to the top. Streamlit keeps the scroll position across a
    rerun, so after navigating from a bottom-of-page button the new room would open
    scrolled down; this jumps it back up. Injected as a 0-height component whose JS
    runs against the parent document (same-origin), covering the scroll container
    across Streamlit versions.

    `nonce` (incremented per navigation) is embedded so the component's HTML differs
    every time — otherwise Streamlit reuses the identical iframe across reruns and
    the script never re-fires."""
    components.html(
        f"""
        <script>
        // nav {nonce}
        const scroll = () => {{
            const d = window.parent.document;
            const els = [d.querySelector('section.main'),
                         d.querySelector('[data-testid="stMain"]'),
                         d.querySelector('[data-testid="stAppViewContainer"]'),
                         d.querySelector('.main'),
                         d.scrollingElement, d.documentElement, d.body];
            for (const el of els) {{
                if (!el) continue;
                try {{ el.scrollTo(0, 0); }} catch (e) {{}}
                try {{ el.scrollTop = 0; }} catch (e) {{}}
            }}
        }};
        scroll();
        // run again after layout settles (Plotly boards can shift the scroll height)
        requestAnimationFrame(scroll);
        setTimeout(scroll, 60);
        </script>
        """,
        height=0,
    )


def _go_to_room(room_number: int) -> None:
    """Select a room and request a scroll-to-top on the next run."""
    st.session_state["selected_room"] = room_number
    st.session_state["_nav_nonce"] = st.session_state.get("_nav_nonce", 0) + 1
    st.session_state["_scroll_top"] = True
    st.rerun()


def room_selector() -> int:
    """Render the top navigation bar (all rooms always accessible) and return the
    selected room number."""
    st.session_state.setdefault("selected_room", 1)

    # If a navigation just happened, jump the view back to the top of the room.
    if st.session_state.pop("_scroll_top", False):
        _scroll_to_top(st.session_state.get("_nav_nonce", 0))

    selected_room = st.session_state["selected_room"]

    st.markdown("---")
    nav_cols = st.columns(len(NAV_ROOMS))
    for idx, room_number in enumerate(NAV_ROOMS):
        is_active = room_number == selected_room
        with nav_cols[idx]:
            if st.button(
                ROOMS[room_number],
                key=f"nav_room_{room_number}",
                use_container_width=True,
                # Highlight the room the user is currently viewing.
                type="primary" if is_active else "secondary",
            ):
                _go_to_room(room_number)

    st.caption("Jump to any room — no need to finish one to move on.")
    st.markdown("---")
    return st.session_state["selected_room"]


def render_room_completion_controls(room_number: int) -> None:
    """Show the "continue to the next room" control at the bottom of a room.

    No gating: the button is always available. Clicking it selects the next room
    and scrolls to the top. The final room shows a closing note instead."""
    next_room = room_number + 1
    has_next = next_room in NAV_ROOMS

    st.markdown("---")
    if has_next:
        if st.button(
            f"Continue to {ROOMS[next_room]} →",
            key=f"continue_room_{room_number}",
            use_container_width=True,
            type="primary",
        ):
            _go_to_room(next_room)
    else:
        st.success("🎉 That's the last room — you've been through all six algorithms.")
