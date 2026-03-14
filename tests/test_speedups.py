import speedups


def test_interpolate_cursor_query_prefers_rust_backend(monkeypatch) -> None:
    calls: list[str] = []

    def rust_impl(*_args):
        calls.append("rust")
        return (3, 128.0, 64.0)

    def cython_impl(*_args):
        calls.append("cython")
        return (1, 64.0, 32.0)

    monkeypatch.setattr(speedups, "_rust_interpolate_cursor_query", rust_impl)
    monkeypatch.setattr(speedups, "_cy_interpolate_cursor_query", cython_impl)

    result = speedups.interpolate_cursor_query([], [], [], 10.0, 0)

    assert result == (3, 128.0, 64.0)
    assert calls == ["rust"]


def test_keys_index_at_returns_none_without_native_backends(monkeypatch) -> None:
    monkeypatch.setattr(speedups, "_rust_keys_index_at", None)
    monkeypatch.setattr(speedups, "_cy_keys_index_at", None)

    assert speedups.keys_index_at([], 25.0, 0) is None


def test_compute_slider_ball_instances_returns_negative_one_without_native_backends(monkeypatch) -> None:
    monkeypatch.setattr(speedups, "_rust_compute_slider_ball_instances", None)
    monkeypatch.setattr(speedups, "_cy_compute_slider_ball_instances", None)

    result = speedups.compute_slider_ball_instances(
        0.0,
        32.0,
        (1.0, 1.0, 1.0),
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        8,
    )

    assert result == -1
