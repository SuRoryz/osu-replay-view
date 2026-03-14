from __future__ import annotations


def _share_from_context(client, command: str):
    provider = getattr(client, "command_context_provider", None)
    if provider is None:
        return None
    context = provider()
    if context is None or not hasattr(context, "build_chat_share"):
        return None
    return context.build_chat_share(command)


def execute_social_command(client, raw_value: str) -> str | None:
    text = raw_value.strip()
    if not text.startswith("/"):
        return None
    parts = text.split()
    command = parts[0].lower()
    args = parts[1:]

    if command == "/np":
        share = _share_from_context(client, command)
        if share is None:
            client.push_system_message("No selected beatmap to share.")
            return "handled"
        content, payload = share
        client.send_message(content, payload=payload)
        return "handled"
    if command == "/npr":
        share = _share_from_context(client, command)
        if share is None:
            client.push_system_message("No selected replays to share.")
            return "handled"
        content, payload = share
        client.send_message(content, payload=payload)
        return "handled"
    if command in {"/help", "/h"}:
        client.push_system_message(
            "Commands: /join <room>, /msg <user> <text>, /query <user>, /me <action>, /clear, /close, /away, /np, /npr."
        )
        return "handled"
    if command == "/join":
        if not args:
            client.push_system_message("Usage: /join <room>")
            return "handled"
        client.create_room(" ".join(args))
        return "handled"
    if command in {"/query", "/chat"}:
        if not args:
            client.push_system_message("Usage: /query <user>")
            return "handled"
        client.open_dm_by_name(" ".join(args))
        return "handled"
    if command == "/msg":
        if len(args) < 2:
            client.push_system_message("Usage: /msg <user> <message>")
            return "handled"
        client.send_private_message(args[0], " ".join(args[1:]))
        return "handled"
    if command == "/me":
        if not args:
            client.push_system_message("Usage: /me <action>")
            return "handled"
        client.send_message(" ".join(args), is_action=True)
        return "handled"
    if command == "/clear":
        client.clear_active_channel_messages()
        return "handled"
    if command in {"/close", "/part"}:
        client.close_active_channel()
        return "handled"
    if command == "/away":
        client.push_system_message("Away state is not persisted yet.")
        return "handled"
    client.push_system_message(f"Unsupported command: {command}")
    return "handled"
