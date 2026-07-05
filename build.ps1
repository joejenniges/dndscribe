# build.ps1 — build the dndscribe-go binaries and bundle the sherpa-onnx +
# onnxruntime DLLs next to the output .exe.
#
# WHY the DLL copy is required: sherpa-onnx-go-windows links the bot against
# sherpa-onnx-c-api.dll and onnxruntime.dll (see the module's
# build_windows_amd64.go cgo LDFLAGS). Those DLLs live in the Go module cache,
# not on PATH, so a freshly built .exe fails to start with a missing-DLL error
# unless they sit next to it. This is the single most common sherpa-onnx
# packaging gotcha, so we handle it explicitly here.
#
# NOTE: build from the MSYS2 UCRT64 shell (or with C:\msys64\ucrt64\bin ahead of
# any other mingw bin on PATH) — same cgo toolchain requirement as libopus /
# whisper. See build.md.
[CmdletBinding()]
param(
    [string]$OutDir = "bin",
    [switch]$Smoke   # also build cmd/sherpa-smoke
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$env:CGO_ENABLED = "1"

Write-Host "Building cmd/bot ..."
go build -o (Join-Path $OutDir "dndscribe-go.exe") ./cmd/bot
if ($LASTEXITCODE -ne 0) { throw "go build ./cmd/bot failed" }

if ($Smoke) {
    Write-Host "Building cmd/sherpa-smoke ..."
    go build -o (Join-Path $OutDir "sherpa-smoke.exe") ./cmd/sherpa-smoke
    if ($LASTEXITCODE -ne 0) { throw "go build ./cmd/sherpa-smoke failed" }
}

# Locate the sherpa-onnx-go-windows module in the module cache and copy its
# amd64 DLLs next to the binaries.
Write-Host "Locating sherpa-onnx-go-windows module ..."
$modDir = (go list -m -f '{{.Dir}}' github.com/k2-fsa/sherpa-onnx-go-windows).Trim()
if (-not $modDir -or -not (Test-Path $modDir)) {
    throw "could not locate sherpa-onnx-go-windows module dir (got '$modDir')"
}
$libDir = Join-Path $modDir "lib/x86_64-pc-windows-gnu"
$dlls = @("onnxruntime.dll", "sherpa-onnx-c-api.dll", "sherpa-onnx-cxx-api.dll")

foreach ($dll in $dlls) {
    $src = Join-Path $libDir $dll
    if (-not (Test-Path $src)) { throw "missing expected DLL: $src" }
    Copy-Item $src -Destination $OutDir -Force
    Write-Host "  bundled $dll"
}

Write-Host "Build complete. Binaries + DLLs in $OutDir"
Write-Host "Run from the repo root so web/build and config.yaml resolve:"
Write-Host "  ./$OutDir/dndscribe-go.exe"
