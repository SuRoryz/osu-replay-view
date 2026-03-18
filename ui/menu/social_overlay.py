"""Global social overlay with chat drawer and online users panel."""

from __future__ import annotations

from dataclasses import dataclass

from social.client import SocialClient
from social.models import format_chat_payload
from ui.design import (
    InteractionState,
    draw_button,
    draw_dropdown_menu,
    draw_supporting_text,
    draw_surface,
    draw_tab,
    draw_text_field,
)
from ui.menu.animation import AnimatedFloat, ease_in_out_cubic, ease_out_back, ease_out_cubic
from ui.menu.commands import RenderCommandBuffer
from ui.menu.layout import Rect


@dataclass(slots=True)
class _OptionRect:
    rect: Rect
    value: str


class SocialOverlay:
    def __init__(self, app, client: SocialClient) -> None:
        self.app = app
        self.client = client
        self._drawer_open = False
        self._drawer_anim = AnimatedFloat(0.0, 0.0, 10.0)
        self._panel_anim = AnimatedFloat(0.0, 0.0, 8.0)
        self._menu_anim = AnimatedFloat(0.0, 0.0, 16.0)
        self._input_focus_anim = AnimatedFloat(0.0, 0.0, 14.0)
        self._search_focus_anim = AnimatedFloat(0.0, 0.0, 14.0)
        self._room_focus_anim = AnimatedFloat(0.0, 0.0, 14.0)
        self._chat_input = ""
        self._search_input = ""
        self._room_input = ""
        self._input_focus = False
        self._search_focus = False
        self._room_focus = False
        self._mouse_x = 0
        self._mouse_y = 0
        self._message_scroll = 0.0
        self._user_scroll = 0.0
        self._channel_tab_rects: list[tuple[Rect, str]] = []
        self._sender_rects: list[tuple[Rect, str]] = []
        self._message_link_rects: list[tuple[Rect, object]] = []
        self._user_rects: list[tuple[Rect, str]] = []
        self._menu_rects: list[_OptionRect] = []
        self._menu_anchor: Rect | None = None
        self._menu_user_uuid: str | None = None
        self._users_tab = "all"
        self._users_panel_rect = Rect(0.0, 0.0, 0.0, 0.0)
        self._messages_rect = Rect(0.0, 0.0, 0.0, 0.0)
        self._input_rect = Rect(0.0, 0.0, 0.0, 0.0)
        self._search_rect = Rect(0.0, 0.0, 0.0, 0.0)
        self._room_rect = Rect(0.0, 0.0, 0.0, 0.0)
        self._users_all_rect = Rect(0.0, 0.0, 0.0, 0.0)
        self._users_friends_rect = Rect(0.0, 0.0, 0.0, 0.0)
        self._message_enter_progress: dict[str, float] = {}
        self._message_counts: dict[str, int] = {}
        self._message_max_scrolls: dict[str, float] = {}
        self._last_active_channel_id: str | None = None

    def _set_focus(self, target: str | None) -> None:
        self._input_focus = target == "input"
        self._search_focus = target == "search"
        self._room_focus = target == "room"
        self._input_focus_anim.set_target(1.0 if self._input_focus else 0.0)
        self._search_focus_anim.set_target(1.0 if self._search_focus else 0.0)
        self._room_focus_anim.set_target(1.0 if self._room_focus else 0.0)

    def _set_drawer_open(self, is_open: bool, *, focus_input: bool = False) -> None:
        self._drawer_open = bool(is_open)
        self._drawer_anim.set_target(1.0 if self._drawer_open else 0.0)
        self._panel_anim.set_target(1.0 if self._drawer_open else 0.0)
        if self._drawer_open and self.client.active_channel_id is None and self.client.channel_order:
            self.client.select_channel(self.client.channel_order[0])
        self._set_focus("input" if self._drawer_open and focus_input else None)

    @property
    def is_visible(self) -> bool:
        return (
            self._drawer_open
            or self._drawer_anim.value > 0.001
            or self._panel_anim.value > 0.001
            or self._menu_anim.value > 0.001
        )

    def toggle(self) -> None:
        self._set_drawer_open(not self._drawer_open, focus_input=not self._drawer_open)

    def close(self) -> None:
        self._set_drawer_open(False)
        self._menu_anim.set_target(0.0)

    def handle_key_event(self, key, action) -> bool:
        keys = self.app.wnd.keys
        if action != keys.ACTION_PRESS:
            return False
        slash_key = getattr(keys, "SLASH", None)
        if key == slash_key and not (self._input_focus or self._search_focus or self._room_focus):
            if not self._drawer_open:
                self._set_drawer_open(True, focus_input=True)
            else:
                self._set_focus("input")
            return True
        if key == keys.F9:
            self.toggle()
            return True
        if not self.is_visible:
            return False
        if self._menu_user_uuid and key == keys.ESCAPE:
            self._close_menu()
            return True
        if key == keys.TAB:
            if self._drawer_open:
                if self._input_focus:
                    self._set_focus("search")
                elif self._search_focus:
                    self._set_focus("room")
                else:
                    self._set_focus("input")
                return True
        if key == keys.ESCAPE and self._drawer_open:
            self.close()
            return True
        active_text_focus = self._input_focus or self._search_focus or self._room_focus
        if active_text_focus and key == keys.BACKSPACE:
            self._delete_last_char()
            return True
        if key == keys.ENTER:
            if self._room_focus and self._room_input.strip():
                self.client.create_room(self._room_input.strip())
                self._room_input = ""
                return True
            if self._input_focus and self._chat_input.strip():
                self.client.send_message_input(self._chat_input.strip())
                self._chat_input = ""
                return True
        return self._drawer_open

    def handle_text(self, char: str) -> bool:
        if not char or ord(char[0]) < 32:
            return False
        if self._input_focus:
            self._chat_input += char[0]
            return True
        if self._search_focus:
            self._search_input += char[0]
            return True
        if self._room_focus:
            self._room_input += char[0]
            return True
        return False

    def handle_mouse_press(self, x: int, y: int, button: int) -> bool:
        self._mouse_x = x
        self._mouse_y = y
        if button != 1:
            return self.is_visible
        if self._menu_user_uuid is not None:
            if self._menu_anchor is not None and not self._menu_anchor.contains(x, y):
                for option in self._menu_rects:
                    if option.rect.contains(x, y):
                        self._trigger_user_option(option.value, self._menu_user_uuid)
                        self._close_menu()
                        return True
                self._close_menu()
        for rect, channel_id in self._channel_tab_rects:
            if rect.contains(x, y):
                self.client.select_channel(channel_id)
                self._set_focus("input")
                return True
        for rect, user_uuid in self._sender_rects:
            if rect.contains(x, y):
                self._open_menu(rect, user_uuid)
                return True
        for rect, payload in self._message_link_rects:
            if rect.contains(x, y) and self.client.activate_message_payload(payload):
                return True
        for rect, user_uuid in self._user_rects:
            if rect.contains(x, y):
                self._open_menu(rect, user_uuid)
                return True
        if self._input_rect.contains(x, y):
            self._set_focus("input")
            return True
        if self._search_rect.contains(x, y):
            self._set_focus("search")
            return True
        if self._room_rect.contains(x, y):
            self._set_focus("room")
            return True
        if self._users_all_rect.contains(x, y):
            self._users_tab = "all"
            return True
        if self._users_friends_rect.contains(x, y):
            self._users_tab = "friends"
            return True
        if self._drawer_open and not self._messages_rect.contains(x, y) and not self._users_panel_rect.contains(x, y):
            self.close()
            return True
        return self.is_visible

    def handle_mouse_move(self, x: int, y: int) -> bool:
        self._mouse_x = x
        self._mouse_y = y
        return self.is_visible

    def handle_mouse_release(self, button: int) -> bool:
        return self.is_visible and button == 1

    def handle_scroll(self, y_offset: float) -> bool:
        if self._messages_rect.contains(self._mouse_x, self._mouse_y):
            self._message_scroll = max(0.0, self._message_scroll - y_offset * 26.0)
            return True
        if self._users_panel_rect.contains(self._mouse_x, self._mouse_y):
            self._user_scroll = max(0.0, self._user_scroll - y_offset * 26.0)
            return True
        return self.is_visible

    def wants_hand_cursor(self) -> bool:
        if not self.is_visible:
            return False
        for rect, _channel_id in self._channel_tab_rects:
            if rect.contains(self._mouse_x, self._mouse_y):
                return True
        for rect, _user_uuid in self._sender_rects:
            if rect.contains(self._mouse_x, self._mouse_y):
                return True
        for rect, _payload in self._message_link_rects:
            if rect.contains(self._mouse_x, self._mouse_y):
                return True
        for rect, _user_uuid in self._user_rects:
            if rect.contains(self._mouse_x, self._mouse_y):
                return True
        for option in self._menu_rects:
            if option.rect.contains(self._mouse_x, self._mouse_y):
                return True
        return any(
            rect.contains(self._mouse_x, self._mouse_y)
            for rect in (
                self._users_all_rect,
                self._users_friends_rect,
            )
        )

    def wants_text_cursor(self) -> bool:
        if not self.is_visible:
            return False
        return any(
            rect.contains(self._mouse_x, self._mouse_y)
            for rect in (
                self._input_rect,
                self._search_rect,
                self._room_rect,
            )
        )

    def draw(self, dt: float) -> None:
        self._drawer_anim.update(dt)
        self._panel_anim.update(dt)
        self._menu_anim.update(dt)
        self._input_focus_anim.update(dt)
        self._search_focus_anim.update(dt)
        self._room_focus_anim.update(dt)
        if (
            self._menu_anim.target <= 0.001
            and self._menu_anim.value <= 0.001
            and self._menu_user_uuid is not None
        ):
            self._menu_user_uuid = None
            self._menu_anchor = None
            self._menu_rects = []
        if not self.is_visible:
            return
        self.app.set_settings_button(None, visible=False)
        layout = self.app.menu_context()
        theme = layout.theme
        density = layout.density
        colors = theme.colors
        commands = RenderCommandBuffer()
        edge_pad = 16.0 * density
        panel_progress = ease_out_cubic(self._panel_anim.value)
        drawer_progress = ease_out_cubic(self._drawer_anim.value)
        users_panel_w = min(336.0 * density, layout.viewport.w * 0.24)
        users_panel_h = layout.viewport.h - edge_pad * 2.0
        users_panel_x = layout.viewport.right - edge_pad - users_panel_w
        users_panel_y = edge_pad
        drawer_h = max(270.0 * density, layout.viewport.h * 0.36)
        drawer_w = max(520.0 * density, users_panel_x - edge_pad * 2.0)
        drawer_y = layout.viewport.bottom - edge_pad - drawer_h * drawer_progress
        drawer_rect = Rect(edge_pad, drawer_y, drawer_w, drawer_h)
        panel_rect = Rect(
            users_panel_x + (1.0 - panel_progress) * (44.0 * density),
            users_panel_y,
            users_panel_w,
            users_panel_h,
        )
        self._users_panel_rect = panel_rect
        if self._drawer_anim.value > 0.001 or self._panel_anim.value > 0.001:
            draw_surface(
                commands,
                theme,
                Rect(0.0, 0.0, layout.viewport.w, layout.viewport.h),
                role="scrim",
                radius=0.0,
                alpha=0.44 * max(drawer_progress, panel_progress),
                border_width=0.0,
            )
            commands.panel(
                Rect(0.0, 0.0, layout.viewport.w, layout.viewport.h),
                radius=0.0,
                color=(colors.surface[0], colors.surface[1], colors.surface[2], 0.12 * max(drawer_progress, panel_progress)),
                border_color=(0.0, 0.0, 0.0, 0.0),
                border_width=0.0,
            )
        draw_surface(
            commands,
            theme,
            panel_rect,
            role="drawer",
            radius=layout.tokens.radius_l,
            alpha=0.82 * panel_progress,
            border_width=0.0,
        )
        users = self.client.filtered_users(self._users_tab, self._search_input)
        online_count = sum(1 for user in self.client.users.values() if user.online)
        if self.client.connected:
            status_text = "Connected"
            status_color = colors.success
        elif self.client.connection_error:
            status_text = "Error"
            status_color = colors.error
        else:
            status_text = "Connecting..."
            status_color = colors.focus_ring
        accent_rect = Rect(panel_rect.x + 12.0 * density, panel_rect.y + 14.0 * density, 3.0 * density, 54.0 * density)
        commands.panel(
            accent_rect,
            radius=1.5 * density,
            color=(colors.focus_ring[0], colors.focus_ring[1], colors.focus_ring[2], 0.84 * panel_progress),
            border_color=(0.0, 0.0, 0.0, 0.0),
            border_width=0.0,
        )
        status_w = self.app.text.measure(status_text, layout.tokens.typography.caption)[0] + 28.0 * density
        status_rect = Rect(panel_rect.right - status_w - 16.0 * density, panel_rect.y + 14.0 * density, status_w, 20.0 * density)
        commands.panel(
            status_rect,
            radius=status_rect.h * 0.5,
            color=(colors.surface_variant_soft[0], colors.surface_variant_soft[1], colors.surface_variant_soft[2], 0.42 * panel_progress),
            border_color=(0.0, 0.0, 0.0, 0.0),
            border_width=0.0,
        )
        commands.panel(
            Rect(status_rect.x + 8.0 * density, status_rect.y + 6.0 * density, 8.0 * density, 8.0 * density),
            radius=4.0 * density,
            color=(status_color[0], status_color[1], status_color[2], 0.94 * panel_progress),
            border_color=(0.0, 0.0, 0.0, 0.0),
            border_width=0.0,
        )
        commands.text(
            status_text,
            status_rect.x + 21.0 * density,
            status_rect.y + 1.0 * density,
            layout.tokens.typography.caption,
            color=colors.text_secondary,
            alpha=0.90 * panel_progress,
        )
        commands.text(
            "Current online",
            panel_rect.x + 24.0 * density,
            panel_rect.y + 18.0 * density,
            layout.tokens.typography.caption,
            color=colors.text_muted,
            alpha=0.82 * panel_progress,
        )
        online_value = str(online_count)
        commands.text(
            online_value,
            panel_rect.x + 24.0 * density,
            panel_rect.y + 36.0 * density,
            layout.tokens.typography.title_m,
            color=colors.text_primary,
            alpha=0.98 * panel_progress,
        )
        tabs_strip_rect = Rect(panel_rect.x + 16.0 * density, panel_rect.y + 76.0 * density, 164.0 * density, 34.0 * density)
        commands.panel(
            tabs_strip_rect,
            radius=10.0 * density,
            color=(colors.surface_variant_soft[0], colors.surface_variant_soft[1], colors.surface_variant_soft[2], 0.16 * panel_progress),
            border_color=(0.0, 0.0, 0.0, 0.0),
            border_width=0.0,
        )
        all_rect = Rect(tabs_strip_rect.x + 8.0 * density, tabs_strip_rect.y + 5.0 * density, 64.0 * density, 24.0 * density)
        friends_rect = Rect(all_rect.right + 8.0 * density, all_rect.y, 84.0 * density, all_rect.h)
        self._users_all_rect = all_rect
        self._users_friends_rect = friends_rect
        for tab_rect, label, selected in (
            (all_rect, "All", self._users_tab == "all"),
            (friends_rect, "Friends", self._users_tab == "friends"),
        ):
            draw_tab(
                commands,
                self.app.text,
                theme,
                tab_rect,
                label=label,
                size=layout.tokens.typography.caption,
                selected=selected,
                hovered=tab_rect.contains(self._mouse_x, self._mouse_y),
                alpha=0.92 * panel_progress,
            )
        self._search_rect = Rect(panel_rect.x + 16.0 * density, friends_rect.bottom + 16.0 * density, panel_rect.w - 32.0 * density, 34.0 * density)
        draw_text_field(
            commands,
            self.app.text,
            theme,
            self._search_rect,
            value=self._search_input or "Search players",
            placeholder=not self._search_input,
            focused=self._search_focus_anim.value,
            size=layout.tokens.typography.body_s,
            density=density,
            alpha=0.94 * panel_progress,
        )
        list_rect = Rect(self._search_rect.x, self._search_rect.bottom + 10.0 * density, self._search_rect.w, panel_rect.bottom - self._search_rect.bottom - 22.0 * density)
        self._user_rects = []
        commands.clip_push(list_rect)
        row_y = list_rect.y + 2.0 * density - self._user_scroll
        for user in users:
            has_status = bool(user.online and user.status_text)
            row_h = (50.0 if has_status else 34.0) * density
            if row_y + row_h < list_rect.y:
                row_y += row_h + 4.0 * density
                continue
            if row_y > list_rect.bottom:
                break
            row_rect = Rect(list_rect.x, row_y, list_rect.w, row_h)
            hovered = row_rect.contains(self._mouse_x, self._mouse_y)
            row_draw_rect = row_rect.translate(dx=(1.0 - panel_progress) * 18.0 * density)
            draw_surface(commands, theme, row_draw_rect, role="section", radius=9.0 * density, alpha=(0.38 + (0.14 if hovered else 0.0)) * panel_progress, border_width=0.0)
            dot_color = colors.success if user.online else colors.text_muted
            dot_y = row_draw_rect.y + (12.0 if not has_status else 10.0) * density
            commands.panel(Rect(row_draw_rect.x + 10.0 * density, dot_y, 8.0 * density, 8.0 * density), radius=4.0 * density, color=(dot_color[0], dot_color[1], dot_color[2], 0.95 * panel_progress), border_color=(0.0, 0.0, 0.0, 0.0), border_width=0.0)
            nickname = self.app.text.truncate(user.nickname, layout.tokens.typography.body_s, row_draw_rect.w - 88.0 * density)
            name_y = row_draw_rect.y + (6.0 if has_status else 8.0) * density
            commands.text(nickname, row_draw_rect.x + 26.0 * density, name_y, layout.tokens.typography.body_s, color=colors.text_primary if user.online else colors.text_secondary, alpha=0.94 * panel_progress)
            suffix = "Friend" if user.is_friend else ("Blocked" if user.is_blocked else "")
            if suffix:
                draw_supporting_text(commands, theme, suffix, row_draw_rect.right - 52.0 * density, row_draw_rect.y + (7.0 if has_status else 9.0) * density, layout.tokens.typography.caption, alpha=0.78 * panel_progress, tone="muted")
            if has_status:
                status_text = self.app.text.truncate(
                    user.status_text,
                    layout.tokens.typography.caption,
                    row_draw_rect.w - 38.0 * density,
                )
                commands.text(
                    status_text,
                    row_draw_rect.x + 26.0 * density,
                    row_draw_rect.y + 25.0 * density,
                    layout.tokens.typography.caption,
                    color=colors.text_muted,
                    alpha=0.88 * panel_progress,
                )
            self._user_rects.append((row_draw_rect, user.player_uuid))
            row_y += row_h + 4.0 * density
        commands.clip_pop()

        self._channel_tab_rects = []
        self._sender_rects = []
        self._message_link_rects = []
        if self._drawer_anim.value > 0.001:
            room_w = 164.0 * density
            tabs_strip_h = 34.0 * density
            composer_h = 44.0 * density
            draw_surface(commands, theme, drawer_rect, role="drawer", radius=layout.tokens.radius_l, alpha=0.90 * drawer_progress, border_width=0.0)
            chat_accent_rect = Rect(drawer_rect.x + 12.0 * density, drawer_rect.y + 14.0 * density, 3.0 * density, 42.0 * density)
            commands.panel(
                chat_accent_rect,
                radius=1.5 * density,
                color=(colors.focus_ring[0], colors.focus_ring[1], colors.focus_ring[2], 0.84 * drawer_progress),
                border_color=(0.0, 0.0, 0.0, 0.0),
                border_width=0.0,
            )
            top_row_y = drawer_rect.y + 12.0 * density
            self._room_rect = Rect(drawer_rect.right - room_w - 14.0 * density, top_row_y, room_w, 32.0 * density)
            draw_text_field(
                commands,
                self.app.text,
                theme,
                self._room_rect,
                value=self._room_input or "Create room",
                placeholder=not self._room_input,
                focused=self._room_focus_anim.value,
                size=layout.tokens.typography.caption,
                density=density,
                alpha=0.94 * drawer_progress,
            )
            room_cue_rect = Rect(self._room_rect.right - 22.0 * density, self._room_rect.y + 6.0 * density, 20.0 * density, 20.0 * density)
            room_cue_hover = self._room_focus or bool(self._room_input.strip())
            commands.panel(
                room_cue_rect,
                radius=room_cue_rect.w * 0.5,
                color=(
                    colors.focus_ring[0],
                    colors.focus_ring[1],
                    colors.focus_ring[2],
                    (0.32 if room_cue_hover else 0.16) * drawer_progress,
                ),
                border_color=(0.0, 0.0, 0.0, 0.0),
                border_width=0.0,
            )
            commands.text(
                "+",
                room_cue_rect.x + 6.0 * density,
                room_cue_rect.y + 0.0 * density,
                layout.tokens.typography.caption,
                color=colors.text_primary,
                alpha=(0.92 if room_cue_hover else 0.68) * drawer_progress,
            )
            tabs_strip_rect = Rect(
                drawer_rect.x + 24.0 * density,
                top_row_y,
                max(140.0 * density, self._room_rect.x - drawer_rect.x - 34.0 * density),
                tabs_strip_h,
            )
            commands.panel(
                tabs_strip_rect,
                radius=10.0 * density,
                color=(colors.surface_variant_soft[0], colors.surface_variant_soft[1], colors.surface_variant_soft[2], 0.20 * drawer_progress),
                border_color=(0.0, 0.0, 0.0, 0.0),
                border_width=0.0,
            )
            tab_y = tabs_strip_rect.y + 5.0 * density
            tab_x = tabs_strip_rect.x + 8.0 * density
            tab_h = 24.0 * density
            for channel_id in self.client.channel_order[:8]:
                channel = self.client.channels.get(channel_id)
                if channel is None:
                    continue
                label = self._channel_label(channel)
                label_w, _ = self.app.text.measure(label, layout.tokens.typography.caption)
                tab_w = min(160.0 * density, max(76.0 * density, label_w + 22.0 * density))
                tab_rect = Rect(tab_x, tab_y, tab_w, tab_h)
                draw_tab(
                    commands,
                    self.app.text,
                    theme,
                    tab_rect,
                    label=label,
                    size=layout.tokens.typography.caption,
                    selected=channel_id == self.client.active_channel_id,
                    hovered=tab_rect.contains(self._mouse_x, self._mouse_y),
                    alpha=0.92 * drawer_progress,
                )
                self._channel_tab_rects.append((tab_rect, channel_id))
                tab_x += tab_w + 8.0 * density
                if tab_x > tabs_strip_rect.right - 112.0 * density:
                    break
            self._messages_rect = Rect(
                drawer_rect.x + 14.0 * density,
                drawer_rect.y + 28.0 * density,
                drawer_rect.w - 28.0 * density,
                drawer_rect.bottom - (drawer_rect.y + 40.0 * density) - composer_h,
            )
            draw_surface(commands, theme, self._messages_rect, role="section", radius=12.0 * density, alpha=0.26 * drawer_progress, border_width=0.0)
            active_channel_id = self.client.active_channel_id or ""
            active_messages = self.client.messages.get(active_channel_id, [])
            row_h = 30.0 * density
            content_h = max(0.0, len(active_messages) * row_h)
            max_scroll = max(0.0, content_h - (self._messages_rect.h - 32.0 * density))
            prev_count = self._message_counts.get(active_channel_id, 0)
            prev_max_scroll = self._message_max_scrolls.get(active_channel_id, 0.0)
            same_channel = active_channel_id == self._last_active_channel_id
            stick_to_bottom = (
                not same_channel
                or self._message_scroll >= max(0.0, prev_max_scroll - 12.0 * density)
            )
            if not same_channel:
                self._message_scroll = max_scroll
            elif len(active_messages) > prev_count and stick_to_bottom:
                self._message_scroll = max_scroll
            else:
                self._message_scroll = min(self._message_scroll, max_scroll)
            self._message_counts[active_channel_id] = len(active_messages)
            self._message_max_scrolls[active_channel_id] = max_scroll
            new_start_idx = prev_count if same_channel else len(active_messages)
            active_ids = {message.message_id for message in active_messages}
            for idx, message in enumerate(active_messages):
                if message.message_id not in self._message_enter_progress:
                    self._message_enter_progress[message.message_id] = 0.0 if idx >= new_start_idx else 1.0
                self._message_enter_progress[message.message_id] = min(
                    1.0,
                    self._message_enter_progress[message.message_id] + dt * 8.0,
                )
            stale_ids = [message_id for message_id in self._message_enter_progress if message_id not in active_ids]
            for message_id in stale_ids:
                self._message_enter_progress.pop(message_id, None)
            self._last_active_channel_id = active_channel_id
            messages_clip = Rect(
                self._messages_rect.x + 10.0 * density,
                self._messages_rect.y + 20.0 * density,
                self._messages_rect.w - 20.0 * density,
                self._messages_rect.h - 28.0 * density,
            )
            commands.clip_push(messages_clip)
            msg_y = messages_clip.y + 2.0 * density - self._message_scroll
            for message in active_messages:
                if msg_y + row_h < messages_clip.y:
                    msg_y += row_h
                    continue
                if msg_y > messages_clip.bottom:
                    break
                sender_name = message.sender_name or "system"
                if message.is_action:
                    sender_name = f"* {sender_name}"
                row_progress = ease_out_cubic(self._message_enter_progress.get(message.message_id, 1.0))
                row_alpha = max(0.0, min(1.0, row_progress)) * drawer_progress
                row_y = msg_y + (1.0 - row_progress) * 12.0 * density
                sender_w, _ = self.app.text.measure(sender_name, layout.tokens.typography.caption)
                bubble_rect = Rect(messages_clip.x + 2.0 * density, row_y + 1.0 * density, messages_clip.w - 4.0 * density, row_h - 2.0 * density)
                bubble_hover = bubble_rect.contains(self._mouse_x, self._mouse_y)
                if bubble_hover or message.local_only:
                    commands.panel(
                        bubble_rect,
                        radius=7.0 * density,
                        color=(
                            colors.surface_variant_soft[0],
                            colors.surface_variant_soft[1],
                            colors.surface_variant_soft[2],
                            (0.14 if bubble_hover else 0.08) * row_alpha,
                        ),
                        border_color=(0.0, 0.0, 0.0, 0.0),
                        border_width=0.0,
                    )
                sender_rect = Rect(messages_clip.x + 10.0 * density, row_y + 4.0 * density, sender_w + 14.0 * density, 18.0 * density)
                commands.panel(
                    sender_rect,
                    radius=sender_rect.h * 0.5,
                    color=(
                        colors.surface_container[0],
                        colors.surface_container[1],
                        colors.surface_container[2],
                        (0.30 if message.sender_uuid else 0.18) * row_alpha,
                    ),
                    border_color=(0.0, 0.0, 0.0, 0.0),
                    border_width=0.0,
                )
                if message.sender_uuid:
                    self._sender_rects.append((sender_rect, message.sender_uuid))
                commands.text(
                    sender_name,
                    sender_rect.x + 7.0 * density,
                    sender_rect.y + 4.0 * density,
                    layout.tokens.typography.caption,
                    color=colors.focus_ring if message.sender_uuid else colors.text_muted,
                    alpha=0.92 * row_alpha,
                )
                text_x = sender_rect.right + 8.0 * density
                available_w = messages_clip.right - text_x - 8.0 * density
                message_value = format_chat_payload(message.payload) or message.content
                message_text = self.app.text.truncate(message_value, layout.tokens.typography.body_s, available_w)
                message_hover = False
                link_rect = None
                if message.payload is not None:
                    text_w, _ = self.app.text.measure(message_text, layout.tokens.typography.body_s)
                    link_rect = Rect(text_x - 2.0 * density, row_y + 4.0 * density, text_w + 4.0 * density, 20.0 * density)
                    message_hover = link_rect.contains(self._mouse_x, self._mouse_y)
                commands.text(
                    message_text,
                    text_x,
                    row_y + 7.0 * density,
                    layout.tokens.typography.body_s,
                    color=(
                        colors.focus_ring
                        if message.payload is not None
                        else (colors.text_primary if not message.local_only else colors.text_secondary)
                    ),
                    alpha=0.94 * row_alpha,
                )
                if link_rect is not None:
                    self._message_link_rects.append((link_rect, message.payload))
                    if message_hover:
                        commands.panel(
                            Rect(link_rect.x, link_rect.bottom - 2.0 * density, link_rect.w, 2.0 * density),
                            radius=1.0 * density,
                            color=(colors.focus_ring[0], colors.focus_ring[1], colors.focus_ring[2], 0.88 * row_alpha),
                            border_color=(0.0, 0.0, 0.0, 0.0),
                            border_width=0.0,
                        )
                msg_y += row_h
            commands.clip_pop()
            composer_rect = Rect(
                drawer_rect.x + 14.0 * density,
                drawer_rect.bottom - composer_h - 10.0 * density,
                drawer_rect.w - 28.0 * density,
                composer_h,
            )
            commands.panel(
                composer_rect,
                radius=12.0 * density,
                color=(colors.surface_container_low[0], colors.surface_container_low[1], colors.surface_container_low[2], 0.24 * drawer_progress),
                border_color=(0.0, 0.0, 0.0, 0.0),
                border_width=0.0,
            )
            self._input_rect = Rect(composer_rect.x + 8.0 * density, composer_rect.y + 6.0 * density, composer_rect.w - 16.0 * density, composer_rect.h - 12.0 * density)
            draw_text_field(
                commands,
                self.app.text,
                theme,
                self._input_rect,
                value=self._chat_input or "Type message or /command",
                placeholder=not self._chat_input,
                focused=self._input_focus_anim.value,
                size=layout.tokens.typography.body_s,
                density=density,
                alpha=0.96 * drawer_progress,
            )

        if self._menu_user_uuid and self._menu_anchor is not None:
            user = self.client.users.get(self._menu_user_uuid)
            if user is not None:
                labels = [
                    "Start DM",
                    "Remove friend" if user.is_friend else "Add friend",
                    "Unblock" if user.is_blocked else "Block",
                ]
                menu_w = 148.0 * density
                item_h = 28.0 * density
                menu_progress = ease_out_back(self._menu_anim.value)
                menu_rect = Rect(
                    min(self._menu_anchor.x, layout.viewport.w - menu_w - 12.0 * density),
                    min(self._menu_anchor.bottom + 4.0 * density, layout.viewport.h - item_h * len(labels) - 12.0 * density) + (1.0 - menu_progress) * 10.0 * density,
                    menu_w,
                    item_h * len(labels),
                )
                option_rects = [
                    Rect(menu_rect.x + 4.0 * density, menu_rect.y + idx * item_h, menu_rect.w - 8.0 * density, item_h)
                    for idx in range(len(labels))
                ]
                self._menu_rects = [
                    _OptionRect(option_rects[0], "dm"),
                    _OptionRect(option_rects[1], "friend"),
                    _OptionRect(option_rects[2], "block"),
                ]
                draw_dropdown_menu(
                    commands,
                    self.app.text,
                    theme,
                    menu_rect,
                    option_rects=option_rects,
                    labels=labels,
                    selected_index=-1,
                    hovered_index=next((idx for idx, item in enumerate(self._menu_rects) if item.rect.contains(self._mouse_x, self._mouse_y)), -1),
                    size=layout.tokens.typography.caption,
                    alpha=0.98 * menu_progress,
                    progress=self._menu_anim.value,
                    edge_padding=4.0 * density,
                )

        commands.flush(
            ctx=self.app.ctx,
            text=self.app.text,
            panels=self.app.panels,
            window_height=self.app.wnd.buffer_size[1],
        )

    def _channel_label(self, channel) -> str:
        if channel.kind != "dm":
            return channel.name
        for user in self.client.users.values():
            if user.player_uuid != self.client.player_uuid and user.player_uuid in channel.name:
                return f"@{user.nickname}"
        return "@dm"

    def _open_menu(self, rect: Rect, user_uuid: str) -> None:
        self._menu_user_uuid = user_uuid
        self._menu_anchor = rect
        self._menu_anim.snap(0.0)
        self._menu_anim.set_target(1.0)

    def _close_menu(self, *, instant: bool = False) -> None:
        if instant:
            self._menu_user_uuid = None
            self._menu_anchor = None
            self._menu_rects = []
            self._menu_anim.snap(0.0)
            return
        self._menu_anim.set_target(0.0)

    def _delete_last_char(self) -> None:
        if self._input_focus and self._chat_input:
            self._chat_input = self._chat_input[:-1]
        elif self._search_focus and self._search_input:
            self._search_input = self._search_input[:-1]
        elif self._room_focus and self._room_input:
            self._room_input = self._room_input[:-1]

    def _trigger_user_option(self, action: str, user_uuid: str) -> None:
        user = self.client.users.get(user_uuid)
        if user is None:
            return
        if action == "dm":
            self.client.open_dm(user_uuid)
            self._set_drawer_open(True, focus_input=True)
        elif action == "friend":
            self.client.set_friend(user_uuid, not user.is_friend)
        elif action == "block":
            self.client.set_blocked(user_uuid, not user.is_blocked)
