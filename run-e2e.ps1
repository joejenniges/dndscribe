# run-e2e.ps1 — build and run the functional pipeline tests (internal/e2e).
#
# The tests stream the four standard capture WAVs (internal/e2e/testdata/)
# through the full pipeline: opus packets -> voice layer -> engine ->
# punctuation -> diarization -> Postgres. See internal/e2e/e2e_test.go.
#
# WHY a script: the test binary links sherpa-onnx and MUST run from bin\ next
# to the bundled DLLs — from anywhere else, System32's stale onnxruntime.dll
# wins the DLL search and crashes the process. The tests are also gated on
# DNDSCRIBE_E2E=1 so a plain `go test ./...` skips them instead of crashing.
#
#   ./run-e2e.ps1               # sherpa + whisper variants
#   ./run-e2e.ps1 -SkipWhisper  # sherpa only (whisper adds ~1-2 min)
#   ./run-e2e.ps1 -Run Sherpa   # -test.run filter passthrough
[CmdletBinding()]
param(
    [switch]$SkipWhisper,
    [string]$Run = ""
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

Write-Host "Building e2e test binary ..."
Push-Location $root
try {
    go test -c -o bin/e2e.test.exe ./internal/e2e
    if ($LASTEXITCODE -ne 0) { throw "test build failed" }

    $env:DNDSCRIBE_E2E = "1"
    $testArgs = @("-test.v=true")
    if ($SkipWhisper) { $testArgs += "-test.short=true" }
    if ($Run)         { $testArgs += "-test.run=$Run" }

    Set-Location (Join-Path $root "bin")
    ./e2e.test.exe @testArgs
    if ($LASTEXITCODE -ne 0) { throw "e2e tests FAILED" }
    Write-Host "e2e tests passed."
}
finally {
    Pop-Location
    $env:DNDSCRIBE_E2E = $null
}
