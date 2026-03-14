# DISCLAIMER

`osu_replay_view` originally was as an experiment in creating a modern and FAST Python application using moderngl. The codebase, the decisions made (especially the experiments with Rust and Cython), and the functionality iself can be low-quality.

## What is in the repo

- `main.py`: client entry point
- `app.py`: main application and settings model
- `scenes/`, `ui/`, `render/`, `audio/`: client code
- `replay/`, `osu_map/`, `cursor/`: replay and beatmap logic
- `speedups/`: optional Cython speedups
- `rust_speedups/`: optional Rust speedups
- `server/`: optional FastAPI social backend

## Quick start

If you want to use latest stable version, download it from releases. If you want to modify code by yourself, here's guide:

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
python main.py
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
python main.py
```

On first run the app will create local runtime folders like `maps/`, `replays/`, and `skins/hitsounds/` if they do not exist.

Copy `.env.example` to `.env` before local development if you want to configure the social server URL.

The repo root `.env` file is also used when you run `python main.py`. Set `OSU_REPLAY_SERVER_URL` there if you want local dev builds to point at a specific social server.

The settings screen build label is versioned as `major.minor.build-CHANNEL-branch`.
During local dev it shows `last_prod_build + 1` with a `DEV` channel, and packaged builds show the exact `PROD` build that was created.

## Development notes

The client does not require the native extensions to start. If the Cython or Rust modules are missing, it falls back to pure Python.

If you want the native speedups during local development:

```powershell
python setup_native.py build_ext --inplace
python -m maturin develop --release --manifest-path rust_speedups/Cargo.toml
```

The Rust extension is optional. If you do not have a Rust toolchain installed, the client still works.

## Running tests

```powershell
python -m pytest
```

Current state of tests is suboptimal.

## Optional social server

The social backend lives under `server/`. You can run it directly during development, or ship it with Docker.

### Local dev run without docker

The API expects PostgreSQL. Install the server dependencies first:

```powershell
pip install -r server/requirements.txt
```

Then start it from the repo root:

```powershell
python -m uvicorn app.main:app --app-dir server --host 127.0.0.1 --port 8000 --reload
```

Useful environment variables:

- `OSU_SERVER_DATABASE_URL`
- `OSU_SERVER_DB_HOST`
- `OSU_SERVER_DB_PORT`
- `OSU_SERVER_DB_USER`
- `OSU_SERVER_DB_PASSWORD`
- `OSU_SERVER_DB_NAME`
- `OSU_SERVER_STORAGE`
- `OSU_SERVER_CORS`

### Docker deployment

The `server/` folder already includes `Dockerfile`, `docker-compose.yml`, and `deploy.sh`.

Fastest path:

```bash
cd server
./deploy.sh
```

That script checks Docker, writes a default `.env` if you do not have one yet, builds the images, starts PostgreSQL and the API, and waits until both services are healthy.

If you want to run Compose yourself:

```bash
cd server
docker compose up -d --build
```

By default the API is published on port `8000`, replay files are stored under `server/storage/replays`, and PostgreSQL data is stored in the `postgres_data` Docker volume.

Client social-server config comes from the repo root `.env` file via `OSU_REPLAY_SERVER_URL`.

## Building a Windows release

The Windows packaging script builds native speedups when possible, bundles FFmpeg, and produces a clean release folder under `dist/ship`.

Before you intentionally bump the release line, edit `version_state.json` and update only `major` and/or `minor` by hand.
The `build` number is owned by the build script and is incremented automatically for each successful packaged build.

```powershell
powershell -ExecutionPolicy Bypass -File .\build_windows.ps1
```

If you want the packaged app to use the same client `.env`, pass it explicitly during the build:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_windows.ps1 -EnvFile .\.env
```

That copies `.env` into the shipping folder so the built executable uses the same `OSU_REPLAY_SERVER_URL` value as local `python main.py` runs.

The build command also snapshots the branch name and writes version metadata into the release so the settings menu shows values like `0.2.7-PROD-main`.
If you run the client from source, it shows the next dev version instead, for example `0.2.8-DEV-main`.
If you are building outside a git checkout, the branch suffix falls back to `local`.

If you want to force a specific interpreter, pass `-PythonExe`. Example:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_windows.ps1 -PythonExe .\.venv\Scripts\python.exe -EnvFile .\.env
```

By default the build keeps personal/local data out of the release. It creates empty `maps/` and `replays/` folders instead of bundling your local content.

If you really want to bundle local runtime data, you can opt in:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_windows.ps1 -EnvFile .\.env -IncludeMaps -IncludeReplays -IncludeSettings
```

