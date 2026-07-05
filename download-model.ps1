# download-model.ps1 — fetch + extract the sherpa-onnx models the streaming
# engine uses: a streaming Zipformer ASR model and the English punctuation +
# truecasing model.
#
# Archives extract into models/<name>/, which is what the bot's
# transcribe.sherpa.model_dir / punctuation.model_dir should point at.
#
# ASR default is the 20M English streaming model — smaller + lower latency,
# matching the sub-second goal. Pass -Model for a different one.
#
#   ./download-model.ps1                                   # ASR (20M) + punct
#   ./download-model.ps1 -Model sherpa-onnx-streaming-zipformer-en-2023-06-26
#   ./download-model.ps1 -SkipPunct                        # ASR only
[CmdletBinding()]
param(
    [string]$Model = "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17",
    [string]$Punct = "sherpa-onnx-online-punct-en-2024-08-06",
    [string]$Dest  = "models",
    [switch]$SkipPunct
)

$ErrorActionPreference = "Stop"

function Fetch-Model {
    param([string]$ReleaseTag, [string]$Name, [string]$Sentinel)

    $outDir = Join-Path $Dest $Name
    if (Test-Path (Join-Path $outDir $Sentinel)) {
        Write-Host "$Name already present — skipping."
        return
    }

    New-Item -ItemType Directory -Force -Path $Dest | Out-Null
    $url     = "https://github.com/k2-fsa/sherpa-onnx/releases/download/$ReleaseTag/$Name.tar.bz2"
    $archive = Join-Path $Dest "$Name.tar.bz2"

    Write-Host "Downloading $url ..."
    Invoke-WebRequest -Uri $url -OutFile $archive

    Write-Host "Extracting ..."
    # bsdtar (shipped as tar.exe on Windows 10+) handles .tar.bz2 directly.
    tar -xjf $archive -C $Dest
    if ($LASTEXITCODE -ne 0) { throw "tar extraction failed for $Name (exit $LASTEXITCODE)" }
    Remove-Item $archive -Force

    Write-Host "Ready at $outDir"
}

Fetch-Model -ReleaseTag "asr-models" -Name $Model -Sentinel "tokens.txt"
if (-not $SkipPunct) {
    Fetch-Model -ReleaseTag "punctuation-models" -Name $Punct -Sentinel "bpe.vocab"
}

Write-Host "Done."
