#Requires -Version 7.6
<# Admit a prepared checker; recover or prepare helpers under native supervision. #>
[CmdletBinding()]
param(
    [string]$Root = (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent),
    [string]$TargetDirectory = (Join-Path ([IO.Path]::GetTempPath()) 'Bonsai-Push-Sanity-Cargo'),
    [switch]$Prepare
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
$Root = [IO.Path]::GetFullPath($Root)
function Test-PushHelperReceipt([string]$ReceiptPath, [string]$Image, [string]$SourceIdentity) {
    if (-not (Test-Path -LiteralPath $Image -PathType Leaf) -or -not (Test-Path -LiteralPath $ReceiptPath -PathType Leaf)) { return $false }
    try {
        $receipt = Get-Content -LiteralPath $ReceiptPath -Raw | ConvertFrom-Json -AsHashtable
        return $receipt -is [Collections.IDictionary] -and $receipt.Count -eq 3 -and
            $receipt.Contains('SchemaVersion') -and $receipt.Contains('SourceIdentity') -and $receipt.Contains('Sha256') -and
            $receipt.SchemaVersion -is [long] -and $receipt.SchemaVersion -eq 1 -and
            $receipt.SourceIdentity -is [string] -and $receipt.Sha256 -is [string] -and
            $receipt.SourceIdentity -ceq $SourceIdentity -and
            $receipt.Sha256 -ceq (Get-FileHash -LiteralPath $Image -Algorithm SHA256).Hash.ToLowerInvariant()
    } catch { return $false }
}
function Lock-PushSourceCache([string]$Target, [Diagnostics.Stopwatch]$Started) {
    [void][IO.Directory]::CreateDirectory($Target)
    $lockPath = Join-Path $Target '.bonsai-source.lock'
    while ($true) {
        try {
            return [IO.File]::Open($lockPath, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
        } catch [IO.IOException] {
            if ($Started.Elapsed.TotalSeconds -ge 270) {
                throw 'Push blocked: timed out waiting for the source cache lock.'
            }
            [Threading.Thread]::Sleep(10)
        }
    }
}
function Remove-PushCacheEntry([IO.FileSystemInfo]$Entry) {
    if ($Entry.Attributes -band [IO.FileAttributes]::ReadOnly) {
        $Entry.Attributes = $Entry.Attributes -band (-bnot [IO.FileAttributes]::ReadOnly)
    }
    if ($Entry -is [IO.DirectoryInfo] -and -not ($Entry.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        $stack = [Collections.Generic.Stack[IO.DirectoryInfo]]::new()
        $dirs = [Collections.Generic.Stack[IO.DirectoryInfo]]::new()
        $stack.Push($Entry)
        while ($stack.Count -gt 0) {
            $dir = $stack.Pop()
            $dirs.Push($dir)
            foreach ($child in $dir.EnumerateFileSystemInfos()) {
                if ($child.Attributes -band [IO.FileAttributes]::ReadOnly) {
                    $child.Attributes = $child.Attributes -band (-bnot [IO.FileAttributes]::ReadOnly)
                }
                if ($child -is [IO.DirectoryInfo] -and -not ($child.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
                    $stack.Push($child)
                } else {
                    $child.Delete()
                }
            }
        }
        while ($dirs.Count -gt 0) {
            $dirs.Pop().Delete()
        }
    } else {
        $Entry.Delete()
    }
}
function Sync-PushSourceTree([string]$Source, [string]$Destination) {
    $srcRoot = [IO.DirectoryInfo]::new($Source)
    if ($srcRoot.Attributes -band [IO.FileAttributes]::ReparsePoint) {
        throw "Push blocked: source tree root '$Source' cannot be a symlink."
    }
    if ([IO.File]::Exists($Destination)) {
        Remove-PushCacheEntry ([IO.FileInfo]::new($Destination))
    } elseif ([IO.Directory]::Exists($Destination)) {
        $dstRootInfo = [IO.DirectoryInfo]::new($Destination)
        if ($dstRootInfo.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            Remove-PushCacheEntry $dstRootInfo
        }
    }
    [void][IO.Directory]::CreateDirectory($Destination)
    $pending = [Collections.Generic.Stack[object[]]]::new()
    $pending.Push(@($srcRoot, [IO.DirectoryInfo]::new($Destination)))
    while ($pending.Count -gt 0) {
        $pair = $pending.Pop()
        $srcDir = [IO.DirectoryInfo]$pair[0]
        $dstDir = [IO.DirectoryInfo]$pair[1]
        $srcMap = [Collections.Generic.Dictionary[string, IO.FileSystemInfo]]::new([StringComparer]::Ordinal)
        foreach ($entry in $srcDir.EnumerateFileSystemInfos()) {
            if ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "Push blocked: staged source entry '$($entry.FullName)' cannot be a symlink."
            }
            $srcMap[$entry.Name] = $entry
        }
        $kept = [Collections.Generic.Dictionary[string, IO.FileSystemInfo]]::new([StringComparer]::Ordinal)
        foreach ($dstEntry in @($dstDir.EnumerateFileSystemInfos())) {
            [IO.FileSystemInfo]$matched = $null
            if (-not $srcMap.TryGetValue($dstEntry.Name, [ref]$matched) -or
                ($dstEntry.Attributes -band [IO.FileAttributes]::ReparsePoint) -or
                (($dstEntry -is [IO.DirectoryInfo]) -ne ($matched -is [IO.DirectoryInfo]))) {
                Remove-PushCacheEntry $dstEntry
            } else {
                $kept[$dstEntry.Name] = $dstEntry
            }
        }
        foreach ($item in $srcMap.GetEnumerator()) {
            $name = $item.Key
            $srcEntry = $item.Value
            $dstPath = Join-Path $dstDir.FullName $name
            [IO.FileSystemInfo]$dstEntry = $null
            $hasKept = $kept.TryGetValue($name, [ref]$dstEntry)
            if ($srcEntry -is [IO.DirectoryInfo]) {
                if (-not $hasKept) {
                    [void][IO.Directory]::CreateDirectory($dstPath)
                }
                $pending.Push(@($srcEntry, [IO.DirectoryInfo]::new($dstPath)))
            } else {
                $srcFile = [IO.FileInfo]$srcEntry
                $same = $false
                if ($hasKept) {
                    $dstFile = [IO.FileInfo]$dstEntry
                    if ($srcFile.Length -eq $dstFile.Length) {
                        $srcBytes = [IO.File]::ReadAllBytes($srcFile.FullName)
                        $dstBytes = [IO.File]::ReadAllBytes($dstFile.FullName)
                        if ([Convert]::ToBase64String($srcBytes) -ceq [Convert]::ToBase64String($dstBytes)) {
                            $same = $true
                        }
                    }
                    if (-not $same -and ($dstFile.Attributes -band [IO.FileAttributes]::ReadOnly)) {
                        $dstFile.Attributes = $dstFile.Attributes -band (-bnot [IO.FileAttributes]::ReadOnly)
                    }
                }
                if (-not $same) {
                    [void]$srcFile.CopyTo($dstPath, $true)
                }
            }
        }
    }
}
$updates = if (-not $Prepare -and [Console]::IsInputRedirected) { [Console]::In.ReadToEnd() } else { '' }
if (-not $updates.Trim() -and -not $Prepare) { return }
$started = [Diagnostics.Stopwatch]::StartNew()
$originalPwsh = $env:BONSAI_PUSH_PWSH_PATH
$originalTargetDir = $env:BONSAI_PUSH_CARGO_TARGET_DIR
$originalChecksum = $env:CARGO_UNSTABLE_CHECKSUM_FRESHNESS
$originalFingerprint = $env:CARGO_BUILD_FINGERPRINT
$originalIncremental = $env:CARGO_INCREMENTAL
$env:BONSAI_PUSH_PWSH_PATH = [Environment]::ProcessPath
if ((Test-Path -LiteralPath (Join-Path $Root '.githooks/pre-push') -PathType Leaf) -and (Test-Path -LiteralPath (Join-Path $Root '.git'))) {
    $previousNativePref = $PSNativeCommandUseErrorActionPreference
    $PSNativeCommandUseErrorActionPreference = $false
    try {
        $hooksPath = [string](git -C $Root config --get core.hooksPath 2>$null)
        $hooksExit = $LASTEXITCODE
    } finally {
        $PSNativeCommandUseErrorActionPreference = $previousNativePref
    }
    if ($Prepare -and ($hooksExit -ne 0 -or -not $hooksPath.Trim())) {
        git -C $Root config core.hooksPath .githooks
        $hooksPath = '.githooks'
        $hooksExit = 0
    }
    if ($hooksExit -ne 0 -or $hooksPath.Trim() -cne '.githooks') {
        throw "Push blocked: git config core.hooksPath must be '.githooks' (got '$($hooksPath.Trim())'). Run 'git config core.hooksPath .githooks'."
    }
}
Add-Type -AssemblyName System.Windows.Forms, System.IO.Compression.FileSystem
$power = [Windows.Forms.SystemInformation]::PowerStatus
Write-Host "Push sanity power before: AC=$($power.PowerLineStatus); charge=$($power.BatteryLifePercent); flags=$($power.BatteryChargeStatus); battery saver unknown."
$target = [IO.Path]::GetFullPath($TargetDirectory)
$scratch = Join-Path ([IO.Path]::GetTempPath()) ('bonsai-push-bootstrap-' + [guid]::NewGuid().ToString('N'))
[void][IO.Directory]::CreateDirectory($scratch)
try {
    $inputs = @('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', '.cargo', 'tools/ci', 'src/artifacts', 'src/windows-process', 'src/windows-credentials', 'scripts/release/Release-Common.ps1')
    $revisions = @('HEAD') + @($inputs | ForEach-Object { 'HEAD:' + $_ })
    $resolved = @(git -C $Root rev-parse @revisions)
    if ($resolved.Count -ne ($inputs.Count + 1) -or @($resolved | Where-Object { $_ -cnotmatch '^(?:[0-9a-f]{40}|[0-9a-f]{64})$' }).Count) { throw 'Push blocked: committed checker input identities are incomplete.' }
    $head = $resolved[0].Trim()
    $objects = @($resolved[1..$inputs.Count])
    $sourceId = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes(($objects -join "`n")))).ToLowerInvariant()
    $tool = Join-Path $target 'aarch64-pc-windows-msvc/debug/bonsai-ci.exe'
    $receiptPath = Join-Path $target 'push-checker.json'
    $recoveryTool = Join-Path $target 'aarch64-pc-windows-msvc/debug/bonsai-push-recovery.exe'
    $recoveryReceiptPath = Join-Path $target 'push-recovery.json'
    $recover = -not $Prepare -and -not (Test-PushHelperReceipt $receiptPath $tool $sourceId)
    $prepareForPush = $false
    if ($Prepare -or $recover) {
        $recoveryInputs = @('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', '.cargo', 'src/windows-process')
        $recoveryObjects = @($recoveryInputs | ForEach-Object { $objects[[array]::IndexOf($inputs, $_)] })
        $recoveryId = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes(($recoveryObjects -join "`n")))).ToLowerInvariant()
    }
    # A failed explicit rebuild cannot leave an earlier preparation admitted.
    # Keep its binary for diagnosis/reuse, but publish a receipt only on success.
    if ($Prepare) {
        [void][IO.Directory]::CreateDirectory($target)
        [IO.File]::Delete($receiptPath)
        [IO.File]::Delete($recoveryReceiptPath)
    }
    if ($recover) {
        [IO.File]::Delete($receiptPath)
        $prepareForPush = -not (Test-PushHelperReceipt $recoveryReceiptPath $recoveryTool $recoveryId)
        if ($prepareForPush) {
            [IO.File]::Delete($recoveryReceiptPath)
            Write-Host 'Push helpers need preparation; automatically running committed -Prepare under native supervision.'
        } else {
            Write-Host 'Push checker cache is missing, stale or invalid; attempting one bounded offline incremental recovery.'
        }
    }
    # A valid checker needs only committed checking policy. Both preparation
    # and recovery snapshot the complete committed Cargo input closure.
    $archive = Join-Path $scratch 'policy.zip'
    $policy = Join-Path $scratch 'source'
    $archiveArguments = @('-C', $Root, 'archive', '--format=zip', "--output=$archive", $head)
    if (-not $Prepare -and -not $recover) {
        $archiveArguments += @('--',
            'tools/ci/check_push_snapshot.ps1',
            'tools/ci/check_owned_languages.ps1',
            'tools/ci/check_powershell_syntax.ps1',
            'tools/ci/check_workflow_startup.ps1',
            'tools/ci/check_workflow_syntax.ps1',
            'tools/ci/native_ci_cache.ps1',
            'tools/ci/test_device_concurrency.ps1',
            'tools/ci/test_foundation_handoff.ps1',
            'tools/ci/test_native_ci_cache.ps1',
            'tools/ci/test_rust_toolchain.ps1',
            'tools/ci/test_tools_cache_inputs.ps1',
            'tools/ci/test_source_checks.ps1',
            'tools/ci/test_support.ps1')
    }
    git @archiveArguments
    [IO.Compression.ZipFile]::ExtractToDirectory($archive, $policy, $true)
    if ($Prepare -or $recover) {
        . (Join-Path $policy 'scripts/release/Release-Common.ps1')
        $toolchain = Get-ReleaseToolchain $policy
        if ($prepareForPush) {
            try {
                & (Join-Path $policy 'tools/ci/prepare_push_helpers.ps1') -Root $Root -SourceRoot $policy -TargetDirectory $target -ScratchDirectory $scratch -BootstrapMilliseconds $started.ElapsedMilliseconds
                if (-not (Test-PushHelperReceipt $recoveryReceiptPath $recoveryTool $recoveryId) -or
                    -not (Test-PushHelperReceipt $receiptPath $tool $sourceId)) {
                    throw 'Push blocked: automatic preparation did not produce verified helpers for the original committed source.'
                }
            } catch {
                [IO.File]::Delete($receiptPath)
                [IO.File]::Delete($recoveryReceiptPath)
                throw
            }
        } else {
            $stableSource = [IO.Path]::TrimEndingDirectorySeparator($target) + '-source'
            $cacheLock = Lock-PushSourceCache $target $started
            try {
                Sync-PushSourceTree $policy $stableSource
                if ($Prepare) {
                    $env:CARGO_UNSTABLE_CHECKSUM_FRESHNESS = 'true'
                    $env:CARGO_BUILD_FINGERPRINT = 'content'
                    $env:CARGO_INCREMENTAL = '1'
                    Push-Location -LiteralPath $stableSource
                    try {
                        # Extra target-only rustc arguments dirty these binary units,
                        # even if their cached output bytes were externally changed.
                        # Dependencies retain their normal fingerprints and cache.
                        $metadata = 'bonsai_push_prepare_' + [guid]::NewGuid().ToString('N')
                        cargo "+$toolchain" rustc --quiet --locked --offline --manifest-path (Join-Path $stableSource 'Cargo.toml') --package bonsai-ci --bin bonsai-ci --target aarch64-pc-windows-msvc --target-dir $target -- -C "metadata=$metadata"
                        if ($LASTEXITCODE) { throw 'Push blocked: bonsai-ci build failed during preparation.' }
                        cargo "+$toolchain" rustc --quiet --locked --offline --manifest-path (Join-Path $stableSource 'Cargo.toml') --package bonsai-windows-process --bin bonsai-push-recovery --target aarch64-pc-windows-msvc --target-dir $target -- -C "metadata=$metadata"
                        if ($LASTEXITCODE) { throw 'Push blocked: bonsai-push-recovery build failed during preparation.' }
                    }
                    finally { Pop-Location }
                    @{ SchemaVersion = 1; SourceIdentity = $recoveryId; Sha256 = (Get-FileHash -LiteralPath $recoveryTool -Algorithm SHA256).Hash.ToLowerInvariant() } | ConvertTo-Json | Set-Content -LiteralPath $recoveryReceiptPath -Encoding utf8NoBOM
                } else {
                    & $recoveryTool --source $stableSource --target-directory $target --toolchain $toolchain --bootstrap-milliseconds $started.ElapsedMilliseconds
                    if ($LASTEXITCODE) { throw 'Push blocked: bonsai-push-recovery failed; checker receipt remains invalid.' }
                }
            } finally {
                if ($null -ne $cacheLock) { $cacheLock.Dispose() }
            }
        }
        if (-not $prepareForPush) {
            @{ SchemaVersion = 1; SourceIdentity = $sourceId; Sha256 = (Get-FileHash -LiteralPath $tool -Algorithm SHA256).Hash.ToLowerInvariant() } | ConvertTo-Json | Set-Content -LiteralPath $receiptPath -Encoding utf8NoBOM
            if (-not (Test-PushHelperReceipt $receiptPath $tool $sourceId)) {
                [IO.File]::Delete($receiptPath)
                throw 'Push blocked: rebuilt checker did not pass receipt verification.'
            }
        }
    }
    if ($Prepare) {
        'Prepared the committed native checker and recovery supervisor; this is not an outgoing-tip validation verdict.'
        return
    }
    $env:CARGO_UNSTABLE_CHECKSUM_FRESHNESS = 'true'
    $env:CARGO_BUILD_FINGERPRINT = 'content'
    $env:CARGO_INCREMENTAL = '1'
    $env:BONSAI_PUSH_CARGO_TARGET_DIR = $target
    $scratchTool = Join-Path $scratch 'bonsai-ci.exe'
    [IO.File]::Copy($tool, $scratchTool, $true)
    try {
        $updates | & $scratchTool push-sanity --root $Root --policy-root $policy --bootstrap-milliseconds $started.ElapsedMilliseconds
        if ($LASTEXITCODE) { throw 'Push blocked: source sanity did not pass.' }
    } finally {
        if ((Test-Path -LiteralPath $receiptPath -PathType Leaf) -and -not (Test-PushHelperReceipt $receiptPath $tool $sourceId)) {
            [IO.File]::Copy($scratchTool, $tool, $true)
        }
    }
} finally {
    $temporary = [IO.Path]::TrimEndingDirectorySeparator([IO.Path]::GetFullPath([IO.Path]::GetTempPath())) + [IO.Path]::DirectorySeparatorChar
    $absolute = [IO.Path]::GetFullPath($scratch)
    if (-not $absolute.StartsWith($temporary, [StringComparison]::OrdinalIgnoreCase) -or -not [IO.Path]::GetFileName($absolute).StartsWith('bonsai-push-bootstrap-', [StringComparison]::Ordinal)) { throw 'Push bootstrap cleanup escaped temporary storage.' }
    Remove-Item -LiteralPath $absolute -Recurse -Force
    $power = [Windows.Forms.SystemInformation]::PowerStatus
    Write-Host "Push sanity power after: AC=$($power.PowerLineStatus); charge=$($power.BatteryLifePercent); flags=$($power.BatteryChargeStatus); battery saver unknown; aggregate elapsed=$($started.Elapsed)."
    if ($null -eq $originalPwsh) { [Environment]::SetEnvironmentVariable('BONSAI_PUSH_PWSH_PATH', [NullString]::Value) }
    else { $env:BONSAI_PUSH_PWSH_PATH = $originalPwsh }
    if ($null -eq $originalTargetDir) { [Environment]::SetEnvironmentVariable('BONSAI_PUSH_CARGO_TARGET_DIR', [NullString]::Value) }
    else { $env:BONSAI_PUSH_CARGO_TARGET_DIR = $originalTargetDir }
    if ($null -eq $originalChecksum) { [Environment]::SetEnvironmentVariable('CARGO_UNSTABLE_CHECKSUM_FRESHNESS', [NullString]::Value) }
    else { $env:CARGO_UNSTABLE_CHECKSUM_FRESHNESS = $originalChecksum }
    if ($null -eq $originalFingerprint) { [Environment]::SetEnvironmentVariable('CARGO_BUILD_FINGERPRINT', [NullString]::Value) }
    else { $env:CARGO_BUILD_FINGERPRINT = $originalFingerprint }
    if ($null -eq $originalIncremental) { [Environment]::SetEnvironmentVariable('CARGO_INCREMENTAL', [NullString]::Value) }
    else { $env:CARGO_INCREMENTAL = $originalIncremental }
}
