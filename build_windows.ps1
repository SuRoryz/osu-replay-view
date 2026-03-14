[CmdletBinding()]
param(
    [string]$PythonExe = "",
    [string]$OutputDir = "",
    [string]$FfmpegDir = "",
    [string]$FfmpegZipUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
    [string]$EnvFile = "",
    [switch]$IncludeMaps,
    [switch]$IncludeReplays,
    [switch]$IncludeSettings
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSCommandPath

function Resolve-AbsolutePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$InputPath
    )

    if ([System.IO.Path]::IsPathRooted($InputPath)) {
        return [System.IO.Path]::GetFullPath($InputPath)
    }

    return [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $InputPath))
}

function Ensure-Directory {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PathValue
    )

    if (-not (Test-Path $PathValue)) {
        New-Item -ItemType Directory -Path $PathValue | Out-Null
    }
}

function Import-EnvFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PathValue
    )

    if (-not (Test-Path $PathValue)) {
        throw "Env file not found: $PathValue"
    }

    Get-Content -LiteralPath $PathValue | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#")) {
            return
        }
        if ($line.StartsWith("export ")) {
            $line = $line.Substring(7).Trim()
        }
        $eqIndex = $line.IndexOf("=")
        if ($eqIndex -lt 1) {
            return
        }
        $key = $line.Substring(0, $eqIndex).Trim()
        $value = $line.Substring($eqIndex + 1).Trim()
        if ($value.Length -ge 2) {
            $quote = $value[0]
            if (($quote -eq '"' -or $quote -eq "'") -and $value[$value.Length - 1] -eq $quote) {
                $value = $value.Substring(1, $value.Length - 2)
            }
        }
        [System.Environment]::SetEnvironmentVariable($key, $value, "Process")
    }
}

function Resolve-PythonInterpreter {
    param(
        [string]$RequestedPython
    )

    $candidates = @()
    if ($RequestedPython) {
        $candidates += (Resolve-AbsolutePath $RequestedPython)
    }

    $candidates += @(
        (Join-Path $ProjectRoot ".venv\Scripts\python.exe"),
        (Join-Path $ProjectRoot ".venv313\Scripts\python.exe"),
        (Join-Path $ProjectRoot ".venv312\Scripts\python.exe"),
        (Join-Path $ProjectRoot "venv\Scripts\python.exe"),
        (Join-Path $ProjectRoot ".venv311\Scripts\python.exe")
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return @($candidate)
        }
    }

    try {
        $pythonCmd = Get-Command python -ErrorAction Stop
        return @($pythonCmd.Source)
    } catch {
    }

    try {
        $pyCmd = Get-Command py -ErrorAction Stop
        foreach ($version in @("3.13", "3.12", "3.11", "3.10")) {
            try {
                & $pyCmd.Source "-$version" "-c" "import sys; print(sys.version)"
                if ($LASTEXITCODE -eq 0) {
                    return @($pyCmd.Source, "-$version")
                }
            } catch {
            }
        }
    } catch {
    }

    throw "Could not find a usable Python interpreter. Pass -PythonExe explicitly."
}

function Invoke-Python {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $launcher = @($script:PythonLauncher)
    if ($launcher.Length -gt 1) {
        $launcherArgs = @($launcher[1..($launcher.Length - 1)])
        & $launcher[0] @launcherArgs @Arguments
    } else {
        & $launcher[0] @Arguments
    }

    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed: $Arguments"
    }
}

function Copy-DirectoryContents {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SourceDir,
        [Parameter(Mandatory = $true)]
        [string]$DestinationDir
    )

    Ensure-Directory $DestinationDir
    if (-not (Test-Path $SourceDir)) {
        return
    }

    Get-ChildItem -LiteralPath $SourceDir -Force -Recurse | ForEach-Object {
        if ($_.FullName -match '[\\/]+__pycache__(?:[\\/]|$)' -or $_.Extension -eq ".pyc") {
            return
        }

        $relativePath = $_.FullName.Substring($SourceDir.Length).TrimStart('\', '/')
        if (-not $relativePath) {
            return
        }

        $target = Join-Path $DestinationDir $relativePath
        if ($_.PSIsContainer) {
            Ensure-Directory $target
            return
        }

        Ensure-Directory (Split-Path -Parent $target)
        Copy-Item -LiteralPath $_.FullName -Destination $target -Force
    }
}

function Resolve-FfmpegSourceDir {
    param(
        [string]$RequestedDir,
        [string]$ZipUrl
    )

    $cacheRoot = Join-Path $ProjectRoot ".build-cache"

    if ($RequestedDir) {
        $resolved = Resolve-AbsolutePath $RequestedDir
        $ffmpegExe = Join-Path $resolved "ffmpeg.exe"
        $ffprobeExe = Join-Path $resolved "ffprobe.exe"
        if (-not (Test-Path $ffmpegExe) -or -not (Test-Path $ffprobeExe)) {
            throw "The supplied -FfmpegDir must contain ffmpeg.exe and ffprobe.exe."
        }
        return $resolved
    }

    try {
        $systemFfmpeg = (Get-Command ffmpeg -ErrorAction Stop).Source
        $systemFfprobe = (Get-Command ffprobe -ErrorAction Stop).Source
        $systemCacheDir = Join-Path $cacheRoot "system-ffmpeg"
        Ensure-Directory $cacheRoot
        if (Test-Path $systemCacheDir) {
            Remove-Item -LiteralPath $systemCacheDir -Recurse -Force
        }
        Ensure-Directory $systemCacheDir
        Copy-Item -LiteralPath $systemFfmpeg -Destination (Join-Path $systemCacheDir "ffmpeg.exe") -Force
        Copy-Item -LiteralPath $systemFfprobe -Destination (Join-Path $systemCacheDir "ffprobe.exe") -Force
        return $systemCacheDir
    } catch {
    }

    $ffmpegCacheDir = Join-Path $cacheRoot "ffmpeg"
    $ffmpegExe = Join-Path $ffmpegCacheDir "ffmpeg.exe"
    $ffprobeExe = Join-Path $ffmpegCacheDir "ffprobe.exe"
    if ((Test-Path $ffmpegExe) -and (Test-Path $ffprobeExe)) {
        return $ffmpegCacheDir
    }

    $zipPath = Join-Path $cacheRoot "ffmpeg-release-essentials.zip"
    $extractDir = Join-Path $cacheRoot "ffmpeg-extract"
    Ensure-Directory $cacheRoot

    if (Test-Path $extractDir) {
        Remove-Item -LiteralPath $extractDir -Recurse -Force
    }

    Write-Host "Downloading ffmpeg..."
    Invoke-WebRequest -Uri $ZipUrl -OutFile $zipPath
    Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force

    $downloadedFfmpeg = Get-ChildItem -Path $extractDir -Recurse -Filter "ffmpeg.exe" | Select-Object -First 1
    $downloadedFfprobe = Get-ChildItem -Path $extractDir -Recurse -Filter "ffprobe.exe" | Select-Object -First 1
    if ($null -eq $downloadedFfmpeg -or $null -eq $downloadedFfprobe) {
        throw "Downloaded ffmpeg archive did not contain ffmpeg.exe and ffprobe.exe."
    }

    if (Test-Path $ffmpegCacheDir) {
        Remove-Item -LiteralPath $ffmpegCacheDir -Recurse -Force
    }
    Ensure-Directory $ffmpegCacheDir
    Copy-Item -LiteralPath $downloadedFfmpeg.FullName -Destination $ffmpegExe -Force
    Copy-Item -LiteralPath $downloadedFfprobe.FullName -Destination $ffprobeExe -Force
    return $ffmpegCacheDir
}


Set-Location $ProjectRoot

$resolvedPython = Resolve-PythonInterpreter -RequestedPython $PythonExe
$script:PythonLauncher = $resolvedPython

$resolvedOutputDir = if ($OutputDir) {
    Resolve-AbsolutePath $OutputDir
} else {
    Join-Path $ProjectRoot "dist\ship"
}

$resolvedEnvFile = ""
if ($EnvFile) {
    $resolvedEnvFile = Resolve-AbsolutePath $EnvFile
    Import-EnvFile -PathValue $resolvedEnvFile
}

$specPath = Join-Path $ProjectRoot "osu_replay_v2.spec"
$distRoot = Join-Path $ProjectRoot "dist"
$buildRoot = Join-Path $ProjectRoot "build"
$pyinstallerOutputDir = Join-Path $distRoot "osu_replay"
$versionStateSource = Join-Path $ProjectRoot "version_state.json"
$buildMetadataSource = Join-Path $ProjectRoot "build_metadata.json"
$runtimeRequirements = Join-Path $ProjectRoot "requirements.txt"
$buildRequirements = Join-Path $ProjectRoot "requirements-build.txt"
$mapsSource = Join-Path $ProjectRoot "maps"
$replaysSource = Join-Path $ProjectRoot "replays"
$hitsoundsSource = Join-Path $ProjectRoot "skins\hitsounds"
$settingsSource = Join-Path $ProjectRoot "app_settings.json"

Write-Host "Using Python: $($script:PythonLauncher -join ' ')"
Invoke-Python -Arguments @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")
if (Test-Path $runtimeRequirements) {
    Invoke-Python -Arguments @("-m", "pip", "install", "-r", $runtimeRequirements)
}
if (Test-Path $buildRequirements) {
    Invoke-Python -Arguments @("-m", "pip", "install", "-r", $buildRequirements)
}
Invoke-Python -Arguments @("setup_native.py", "build_ext", "--inplace")
try {
    $cargoCmd = Get-Command cargo -ErrorAction Stop
    Write-Host "Building optional Rust speedups..."
    $rustWheelDir = Join-Path $ProjectRoot ".build-cache\rust-wheels"
    if (Test-Path $rustWheelDir) {
        Remove-Item -LiteralPath $rustWheelDir -Recurse -Force
    }
    Ensure-Directory $rustWheelDir
    Invoke-Python -Arguments @("-m", "maturin", "build", "--release", "--manifest-path", (Join-Path $ProjectRoot "rust_speedups\Cargo.toml"), "--out", $rustWheelDir)
    $rustWheel = Get-ChildItem -Path $rustWheelDir -Filter "*.whl" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -ne $rustWheel) {
        Invoke-Python -Arguments @("-m", "pip", "install", "--force-reinstall", $rustWheel.FullName)
    }
} catch {
    Write-Host "Rust toolchain not found; skipping optional Rust speedups."
}

$ffmpegSourceDir = Resolve-FfmpegSourceDir -RequestedDir $FfmpegDir -ZipUrl $FfmpegZipUrl

foreach ($pathValue in @($buildRoot, $pyinstallerOutputDir, $resolvedOutputDir)) {
    if (Test-Path $pathValue) {
        Remove-Item -LiteralPath $pathValue -Recurse -Force
    }
}

Write-Host "Preparing build version..."
Invoke-Python -Arguments @("build_version.py", "prepare-build")

Write-Host "Building executable..."
Invoke-Python -Arguments @("-m", "PyInstaller", "--noconfirm", "--clean", $specPath)

if (-not (Test-Path $pyinstallerOutputDir)) {
    throw "PyInstaller did not create the expected output folder: $pyinstallerOutputDir"
}

Write-Host "Finalizing build version..."
Invoke-Python -Arguments @("build_version.py", "finalize-build")

Ensure-Directory $resolvedOutputDir
Copy-DirectoryContents -SourceDir $pyinstallerOutputDir -DestinationDir $resolvedOutputDir

$shipMapsDir = Join-Path $resolvedOutputDir "maps"
$shipReplaysDir = Join-Path $resolvedOutputDir "replays"
$shipSkinsDir = Join-Path $resolvedOutputDir "skins"
$shipHitsoundsDir = Join-Path $shipSkinsDir "hitsounds"
$shipFfmpegDir = Join-Path $resolvedOutputDir "ffmpeg"

Ensure-Directory $shipMapsDir
Ensure-Directory $shipReplaysDir
Ensure-Directory $shipSkinsDir
Ensure-Directory $shipHitsoundsDir
Ensure-Directory $shipFfmpegDir

if ($IncludeMaps) {
    Copy-DirectoryContents -SourceDir $mapsSource -DestinationDir $shipMapsDir
}
if ($IncludeReplays) {
    Copy-DirectoryContents -SourceDir $replaysSource -DestinationDir $shipReplaysDir
}
Copy-DirectoryContents -SourceDir $hitsoundsSource -DestinationDir $shipHitsoundsDir

Copy-Item -LiteralPath (Join-Path $ffmpegSourceDir "ffmpeg.exe") -Destination (Join-Path $shipFfmpegDir "ffmpeg.exe") -Force
Copy-Item -LiteralPath (Join-Path $ffmpegSourceDir "ffprobe.exe") -Destination (Join-Path $shipFfmpegDir "ffprobe.exe") -Force
Copy-Item -LiteralPath $versionStateSource -Destination (Join-Path $resolvedOutputDir "version_state.json") -Force
Copy-Item -LiteralPath $buildMetadataSource -Destination (Join-Path $resolvedOutputDir "build_metadata.json") -Force

if ($resolvedEnvFile) {
    Copy-Item -LiteralPath $resolvedEnvFile -Destination (Join-Path $resolvedOutputDir ".env") -Force
}

if ($IncludeSettings -and (Test-Path $settingsSource)) {
    Copy-Item -LiteralPath $settingsSource -Destination (Join-Path $resolvedOutputDir "app_settings.json") -Force
}

$exePath = Join-Path $resolvedOutputDir "osu_replay.exe"
if (-not (Test-Path $exePath)) {
    throw "Expected built executable not found at $exePath"
}

Write-Host ""
Write-Host "Shipping build ready:"
Write-Host "  $resolvedOutputDir"
Write-Host ""
Write-Host "Included:"
Write-Host "  - osu_replay.exe"
Write-Host "  - empty maps and replays folders"
if ($IncludeMaps) {
    Write-Host "  - bundled maps"
}
if ($IncludeReplays) {
    Write-Host "  - bundled replays"
}
if (Test-Path $hitsoundsSource) {
    Write-Host "  - skins\hitsounds"
}
if ($IncludeSettings -and (Test-Path $settingsSource)) {
    Write-Host "  - app_settings.json"
}
if ($resolvedEnvFile) {
    Write-Host "  - .env"
}
Write-Host "  - version_state.json"
Write-Host "  - build_metadata.json"
Write-Host "  - ffmpeg\\ffmpeg.exe"
Write-Host "  - ffmpeg\\ffprobe.exe"
