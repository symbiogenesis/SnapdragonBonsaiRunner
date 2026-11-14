#Requires -Version 7.6
<# Bootstrap the native push checker; Git's original ref-update input stays on stdin. #>
[CmdletBinding()]
param(
    [string]$Root = (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent),
    [string]$TargetDirectory = (Join-Path ([IO.Path]::GetTempPath()) 'Bonsai-Push-Sanity-Cargo')
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
$updates = [Console]::In.ReadToEnd()
if (-not $updates.Trim()) { return }
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
    # Bootstrap from committed policy: unstaged edits cannot break or repair the
    # checker used for an otherwise valid outgoing source snapshot.
    $archive = Join-Path $scratch 'policy.zip'
    $policy = Join-Path $scratch 'source'
    git -C $Root archive --format=zip --output=$archive HEAD
    Expand-Archive -LiteralPath $archive -DestinationPath $policy
    cargo build --quiet --locked --offline --manifest-path (Join-Path $policy 'Cargo.toml') --package bonsai-ci --bin bonsai-ci --target-dir $target
    $updates | & (Join-Path $target 'debug/bonsai-ci.exe') push-sanity --root $Root --policy-root $policy --bootstrap-milliseconds $started.ElapsedMilliseconds
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
