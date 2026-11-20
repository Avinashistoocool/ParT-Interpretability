# Push the Kaggle kernels under kaggle/<slug>/.
#   powershell -ExecutionPolicy Bypass -File scripts\push_kaggle.ps1            # push all
#   powershell -ExecutionPolicy Bypass -File scripts\push_kaggle.ps1 -Only part-interp-part
# Username: $env:KAGGLE_USERNAME, else `kaggle config view`, else ~/.kaggle/kaggle.json.
param([string]$Only = "", [switch]$DryRun, [switch]$UpdateBundle)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

$user = $env:KAGGLE_USERNAME
if (-not $user) {
    $cfg = (kaggle config view 2>$null) -join "`n"
    if ($cfg -match "username:\s*(\S+)") { $user = $Matches[1] }
}
if (-not $user) {
    $kj = Join-Path $HOME ".kaggle\kaggle.json"
    if (Test-Path $kj) { $user = (Get-Content $kj -Raw | ConvertFrom-Json).username }
}
if (-not $user -or $user -eq "None") { throw "Kaggle username not found: set KAGGLE_USERNAME or run 'kaggle config set -n username -v <name>'" }
Write-Host "Kaggle user: $user"

# Kaggle sessions on unverified accounts have no internet, so the kernels read code, wheels and
# checkpoints from the private dataset <user>/part-interp-bundle. -UpdateBundle refreshes its code.
if ($UpdateBundle) {
    $b = Join-Path $root "kaggle\_staged\bundle"
    if (-not (Test-Path (Join-Path $b "dataset-metadata.json"))) { throw "bundle staging dir missing: $b (see README, Kaggle section)" }
    Remove-Item -Recurse -Force (Join-Path $b "code") -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force (Join-Path $b "code") | Out-Null
    Copy-Item -Recurse -Force (Join-Path $root "src"), (Join-Path $root "scripts") (Join-Path $b "code")
    Get-ChildItem -Recurse (Join-Path $b "code") -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
    kaggle datasets version -p $b -m "code update" --dir-mode zip
    if ($LASTEXITCODE -ne 0) { throw "bundle update failed" }
    Write-Host "waiting for the bundle version to be ready..."
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep 10
        if ((kaggle datasets status "$user/part-interp-bundle") -match "ready") { break }
    }
}

$template = Get-Content (Join-Path $root "kaggle\kernel_template.py") -Raw
$kernels = @{ "part-interp-part" = "part"; "part-interp-gds" = "gds" }
foreach ($slug in $kernels.Keys) {
    if ($Only -and $Only -ne $slug) { continue }
    $dir = Join-Path $root "kaggle\$slug"
    New-Item -ItemType Directory -Force $dir | Out-Null
    $model = $kernels[$slug]
    [IO.File]::WriteAllText((Join-Path $dir "run.py"), $template.Replace("__MODEL__", $model))
    $metaPath = Join-Path $dir "kernel-metadata.json"
    $meta = Get-Content $metaPath -Raw
    $staged = Join-Path $root "kaggle\_staged\$slug"
    New-Item -ItemType Directory -Force $staged | Out-Null
    Copy-Item (Join-Path $dir "run.py") $staged -Force
    [IO.File]::WriteAllText((Join-Path $staged "kernel-metadata.json"), $meta.Replace("KAGGLE_USERNAME", $user))
    if ($DryRun) { Write-Host "[dry-run] staged $staged"; continue }
    kaggle kernels push -p $staged
    if ($LASTEXITCODE -ne 0) { throw "push failed for $slug" }
    Write-Host "https://www.kaggle.com/code/$user/$slug"
}
