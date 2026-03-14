from types import SimpleNamespace

from audio.engine import _SubprocessProxy, _pydub_hidden_popen


def test_hidden_popen_adds_no_window_flags_on_windows(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class StartupInfo:
        def __init__(self) -> None:
            self.dwFlags = 0
            self.wShowWindow = None

    def fake_popen(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "ok"

    monkeypatch.setattr("audio.engine.sys.platform", "win32")
    monkeypatch.setattr("audio.engine.subprocess.CREATE_NO_WINDOW", 0x08000000, raising=False)
    monkeypatch.setattr("audio.engine.subprocess.STARTF_USESHOWWINDOW", 0x00000001, raising=False)
    monkeypatch.setattr("audio.engine.subprocess.SW_HIDE", 0, raising=False)
    monkeypatch.setattr("audio.engine.subprocess.STARTUPINFO", StartupInfo, raising=False)

    wrapped = _pydub_hidden_popen(fake_popen)
    result = wrapped(["ffmpeg", "-version"])

    assert result == "ok"
    assert captured["kwargs"]["creationflags"] == 0x08000000
    startupinfo = captured["kwargs"]["startupinfo"]
    assert startupinfo.dwFlags == 0x00000001
    assert startupinfo.wShowWindow == 0


def test_subprocess_proxy_delegates_other_attributes() -> None:
    module = SimpleNamespace(Popen=lambda *args, **kwargs: None, PIPE="pipe", DEVNULL="devnull")
    proxy = _SubprocessProxy(module)

    assert proxy.PIPE == "pipe"
    assert proxy.DEVNULL == "devnull"
