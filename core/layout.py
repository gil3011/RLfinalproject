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
    """Scroll the app back to the top after a navigation. `nonce` (bumped per
    navigation) is embedded so the component's HTML differs each time and the
    script re-fires."""
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
    """Render the top navigation bar and return the selected room number."""
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

    st.markdown("---")
    return st.session_state["selected_room"]


def render_room_completion_controls(room_number: int) -> None:
    """Show the "continue to the next room" control at the bottom of a room.
    Rooms 1-5 get a Continue button; the final room shows a closing note, but
    only once it has been trained."""
    next_room = room_number + 1
    has_next = next_room in NAV_ROOMS

    if not has_next:
        # Final room: only celebrate once the user has actually TRAINED this room
        # (its render sets `room{n}_trained_sig` on ▶️ Train). Before that, show
        # nothing — no divider, no banner.
        if st.session_state.get(f"room{room_number}_trained_sig") is not None:
            st.markdown("---")
            st.success("🎉 That's the last room — you've been through all six algorithms.")
        return

    st.markdown("---")
    # Blue "next room" button — scoped to this button's key so the other
    # primary buttons (Train / Play) keep the theme's default colour.
    st.markdown(
        """<style>
        [class*="st-key-continue_room_"] button {
            background-color: #2563eb !important;
            border-color: #2563eb !important;
            color: #ffffff !important;
        }
        [class*="st-key-continue_room_"] button:hover {
            background-color: #1d4ed8 !important;
            border-color: #1d4ed8 !important;
        }
        [class*="st-key-continue_room_"] button:active,
        [class*="st-key-continue_room_"] button:focus {
            background-color: #1e40af !important;
            border-color: #1e40af !important;
        }
        </style>""",
        unsafe_allow_html=True,
    )
    if st.button(
        f"Continue to {ROOMS[next_room]} →",
        key=f"continue_room_{room_number}",
        use_container_width=True,
        type="primary",
    ):
        _go_to_room(next_room)
