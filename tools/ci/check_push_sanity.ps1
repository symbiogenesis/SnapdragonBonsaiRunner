#Requires -Version 7.6
<# Admit a prepared native checker; preparation is explicit, never part of a push. #>
[CmdletBinding()]
param(
    [string]$Root = (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent),
    [string]$TargetDirectory = (Join-Path ([IO.Path]::GetTempPath()) 'Bonsai-Push-Sanity-Cargo'),
    [switch]$Prepare
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
$updates = if ([Console]::IsInputRedirected) { [Console]::In.ReadToEnd() } else { '' }
if (-not $updates.Trim() -and -not $Prepare) { return }
$started = [Diagnostics.Stopwatch]::StartNew()
$originalPwsh = $env:BONSAI_PUSH_PWSH_PATH
$env:BONSAI_PUSH_PWSH_PATH = [Environment]::ProcessPath
Add-Type -AssemblyName System.Windows.Forms
$power = [Windows.Forms.SystemInformation]::PowerStatus
Write-Host "Push sanity power before: AC=$($power.PowerLineStatus); charge=$($power.BatteryLifePercent); flags=$($power.BatteryChargeStatus); battery saver unknown."
$target = [IO.Path]::GetFullPath($TargetDirectory)
$scratch = Join-Path ([IO.Path]::GetTempPath()) ('bonsai-push-bootstrap-' + [guid]::NewGuid().ToString('N'))
[void][IO.Directory]::CreateDirectory($scratch)
try {
    $head = (git -C $Root rev-parse --verify HEAD).Trim()
    $inputs = @('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', '.cargo', 'tools/ci', 'src/artifacts', 'src/windows-process', 'src/windows-credentials')
    $objects = @($inputs | ForEach-Object { (git -C $Root rev-parse --verify ('{0}:{1}' -f $head, $_)).Trim() })
    $sourceId = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes(($objects -join "`n")))).ToLowerInvariant()
    $tool = Join-Path $target 'debug/bonsai-ci.exe'
    $receiptPath = Join-Path $target 'push-checker.json'
    # Ordinary pushes never compile. A missing/stale helper is a prerequisite
    # failure, avoiding an unsupervised cold compiler inside the push allowance.
    if (-not $Prepare) {
        if (-not (Test-Path -LiteralPath $tool -PathType Leaf) -or -not (Test-Path -LiteralPath $receiptPath -PathType Leaf)) { throw 'Push blocked: prepare the committed checker with tools/ci/check_push_sanity.ps1 -Prepare in a declared source-build allowance.' }
        $receipt = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json -AsHashtable
        if ($receipt.Count -ne 3 -or $receipt.SchemaVersion -ne 1 -or $receipt.SourceIdentity -cne $sourceId -or $receipt.Sha256 -cne (Get-FileHash -LiteralPath $tool -Algorithm SHA256).Hash.ToLowerInvariant()) { throw 'Push blocked: checker source or bytes changed; explicit preparation is required.' }
    }
    # Policy scripts and compiler configuration come from committed HEAD.
    $archive = Join-Path $scratch 'policy.zip'
    $policy = Join-Path $scratch 'source'
    git -C $Root archive --format=zip --output=$archive $head
    Expand-Archive -LiteralPath $archive -DestinationPath $policy
    if ($Prepare) {
        Push-Location -LiteralPath $policy
        try { cargo build --quiet --locked --offline --manifest-path (Join-Path $policy 'Cargo.toml') --package bonsai-ci --bin bonsai-ci --target-dir $target }
        finally { Pop-Location }
        @{ SchemaVersion = 1; SourceIdentity = $sourceId; Sha256 = (Get-FileHash -LiteralPath $tool -Algorithm SHA256).Hash.ToLowerInvariant() } | ConvertTo-Json | Set-Content -LiteralPath $receiptPath -Encoding utf8NoBOM
        'Prepared the committed native checker; this is not an outgoing-tip validation verdict.'
        return
    }
    $updates | & $tool push-sanity --root $Root --policy-root $policy --bootstrap-milliseconds $started.ElapsedMilliseconds
    if ($LASTEXITCODE) { throw 'Push blocked: source sanity did not pass.' }
} finally {
    $temporary = [IO.Path]::TrimEndingDirectorySeparator([IO.Path]::GetFullPath([IO.Path]::GetTempPath())) + [IO.Path]::DirectorySeparatorChar
    $absolute = [IO.Path]::GetFullPath($scratch)
    if (-not $absolute.StartsWith($temporary, [StringComparison]::OrdinalIgnoreCase) -or -not [IO.Path]::GetFileName($absolute).StartsWith('bonsai-push-bootstrap-', [StringComparison]::Ordinal)) { throw 'Push bootstrap cleanup escaped temporary storage.' }
    Remove-Item -LiteralPath $absolute -Recurse -Force
    $power = [Windows.Forms.SystemInformation]::PowerStatus
    Write-Host "Push sanity power after: AC=$($power.PowerLineStatus); charge=$($power.BatteryLifePercent); flags=$($power.BatteryChargeStatus); battery saver unknown; aggregate elapsed=$($started.Elapsed)."
    if ($null -eq $originalPwsh) { [Environment]::SetEnvironmentVariable('BONSAI_PUSH_PWSH_PATH', [NullString]::Value) }
    else { $env:BONSAI_PUSH_PWSH_PATH = $originalPwsh }
}
