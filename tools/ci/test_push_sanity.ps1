#Requires -Version 7.6
param([string]$Case)
. (Join-Path $PSScriptRoot 'test_support.ps1')
$adapter=Join-Path $PSScriptRoot 'check_push_sanity.ps1'
$cases=[ordered]@{
 ordinary_push_archives_only_committed_policy={
    $scratch=New-CiScratch 'bonsai-push-policy-'
    try {
        $repository=Join-Path $scratch 'repository'; $target=Join-Path $scratch 'target'
        $inputs=@('Cargo.toml','Cargo.lock','rust-toolchain.toml','.cargo','tools/ci','src/artifacts','src/windows-process','src/windows-credentials','scripts/release/Release-Common.ps1')
        foreach ($path in @('Cargo.toml','Cargo.lock','rust-toolchain.toml','.cargo/fixture.txt','src/artifacts/fixture.txt','src/windows-process/fixture.txt','src/windows-credentials/fixture.txt','scripts/release/Release-Common.ps1','unrelated-source.txt')) { Write-CiFixture $repository $path 'Inert committed source; no compiler consumer.' }
        $policy=@('check_push_snapshot.ps1','check_owned_languages.ps1','check_powershell_syntax.ps1','check_workflow_startup.ps1','check_workflow_syntax.ps1','test_tools_cache_inputs.ps1','test_source_checks.ps1','test_support.ps1')
        foreach ($name in $policy) { Write-CiFixture $repository "tools/ci/$name" "Committed checking policy: $name" }
        & git -C $repository init --quiet
        if ($LASTEXITCODE) { throw 'Cannot initialize policy archive fixture.' }
        & git -C $repository add --all
        if ($LASTEXITCODE) { throw 'Cannot stage policy archive fixture.' }
        & git -C $repository -c user.name=Fixture -c user.email=fixture@example.invalid -c core.hooksPath= -c commit.gpgsign=false commit --quiet -m fixture
        if ($LASTEXITCODE) { throw 'Cannot commit policy archive fixture.' }
        $head=(& git -C $repository rev-parse HEAD).Trim()
        # Compute the legacy ordered identity independently, one input at a time.
        $objects=@($inputs | ForEach-Object { (& git -C $repository rev-parse --verify ('{0}:{1}' -f $head,$_)).Trim() })
        $sourceId=[Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes(($objects -join "`n")))).ToLowerInvariant()
        $binary=Join-Path $target 'aarch64-pc-windows-msvc/debug/bonsai-ci.exe'
        Write-CiFixture $target 'aarch64-pc-windows-msvc/debug/bonsai-ci.exe' 'Inert checker; archive interception must stop before execution.'
        @{SchemaVersion=1;SourceIdentity=$sourceId;Sha256=(Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash.ToLowerInvariant()} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $target 'push-checker.json') -Encoding utf8NoBOM
        foreach ($name in $policy) { Write-CiFixture $repository "tools/ci/$name" 'Uncommitted poison must not become checking policy.' }
        $driver=Join-Path $scratch 'archive-driver.ps1'
        @'
#Requires -Version 7.6
param([string]$Adapter,[string]$Repository,[string]$Target,[string]$GitPath,[string]$Support)
. $Support
$PSNativeCommandUseErrorActionPreference=$true
function git {
    & $GitPath @args
    if ($LASTEXITCODE) { throw 'Fixture Git command failed.' }
    if ($args -ccontains 'archive') {
        $archive=@($args | Where-Object { $_.StartsWith('--output=',[StringComparison]::Ordinal) })[0].Substring(9)
        $unpacked=Join-Path (Split-Path $Repository -Parent) 'unpacked'
        Expand-Archive -LiteralPath $archive -DestinationPath $unpacked
        $expected=@('check_push_snapshot.ps1','check_owned_languages.ps1','check_powershell_syntax.ps1','check_workflow_startup.ps1','check_workflow_syntax.ps1','test_tools_cache_inputs.ps1','test_source_checks.ps1','test_support.ps1')
        $actual=@(Get-ChildItem -LiteralPath $unpacked -File -Recurse | ForEach-Object { [IO.Path]::GetRelativePath($unpacked,$_.FullName).Replace('\','/') })
        Assert-CiSet $actual @($expected | ForEach-Object { "tools/ci/$_" })
        foreach ($name in $expected) { Assert-CiEqual ([IO.File]::ReadAllText((Join-Path $unpacked "tools/ci/$name"))) "Committed checking policy: $name" }
        throw 'Expected committed policy archive verified before helper execution.'
    }
}
try { & $Adapter -Root $Repository -TargetDirectory $Target; throw 'Ordinary push did not reach the archive assertion.' }
catch { if ($_.Exception.Message -cne 'Expected committed policy archive verified before helper execution.') { throw } }
'Committed policy-only bootstrap and legacy receipt identity passed.'
'@ | Set-Content -LiteralPath $driver -Encoding utf8NoBOM
        # Console stdin is populated only in an external PowerShell process.
        "refs/heads/main $head refs/heads/main $head" | & ([Environment]::ProcessPath) -NoLogo -NoProfile -NonInteractive -File $driver -Adapter $adapter -Repository $repository -Target $target -GitPath (Get-Command git -CommandType Application).Source -Support (Join-Path $PSScriptRoot 'test_support.ps1')
        if ($LASTEXITCODE) { throw 'Committed policy-only bootstrap fixture failed.' }
    } finally { Remove-CiScratch $scratch }
 }
 fresh_preparation_reaches_committed_bootstrap={
    $scratch=New-CiScratch 'bonsai-push-prepare-fresh-'
    try {
        $repository=Join-Path $scratch 'repository'; $target=Join-Path $scratch 'absent/cache/target'
        foreach ($path in @('Cargo.toml','Cargo.lock','rust-toolchain.toml','.cargo/fixture.txt','tools/ci/fixture.txt','src/artifacts/fixture.txt','src/windows-process/fixture.txt','src/windows-credentials/fixture.txt')) { Write-CiFixture $repository $path 'Inert source identity fixture; no compiler consumer.' }
        Write-CiFixture $repository 'scripts/release/Release-Common.ps1' "function Get-ReleaseToolchain([string]`$RepositoryRoot) { throw 'Cold-start reached committed bootstrap; Cargo must not run.' }"
        & git -C $repository init --quiet
        if ($LASTEXITCODE) { throw 'Cannot initialize fresh preparation fixture.' }
        & git -C $repository add --all
        if ($LASTEXITCODE) { throw 'Cannot stage fresh preparation fixture.' }
        & git -C $repository -c user.name=Fixture -c user.email=fixture@example.invalid -c core.hooksPath= -c commit.gpgsign=false commit --quiet -m fixture
        if ($LASTEXITCODE) { throw 'Cannot commit fresh preparation fixture.' }
        Assert-Ci (-not (Test-Path -LiteralPath $target)) 'Fresh preparation target already exists.'
        Assert-CiReject { & $adapter -Root $repository -TargetDirectory $target -Prepare } 'Cold-start reached committed bootstrap; Cargo must not run.'
        Assert-Ci (-not (Test-Path -LiteralPath (Join-Path $target 'push-checker.json'))) 'Failed fresh preparation published a receipt.'
    } finally { Remove-CiScratch $scratch }
 }
 failed_preparation_invalidates_existing_receipt={
    $scratch=New-CiScratch 'bonsai-push-prepare-'
    try {
        $repository=Join-Path $scratch 'repository'; $target=Join-Path $scratch 'target'
        foreach ($path in @('Cargo.toml','Cargo.lock','rust-toolchain.toml','.cargo/fixture.txt','tools/ci/fixture.txt','src/artifacts/fixture.txt','src/windows-process/fixture.txt','src/windows-credentials/fixture.txt')) { Write-CiFixture $repository $path 'Inert source identity fixture; no compiler consumer.' }
        Write-CiFixture $repository 'scripts/release/Release-Common.ps1' "function Get-ReleaseToolchain([string]`$RepositoryRoot) { throw 'Expected preparation failure before Cargo.' }"
        & git -C $repository init --quiet
        if ($LASTEXITCODE) { throw 'Cannot initialize push preparation fixture.' }
        & git -C $repository add --all
        if ($LASTEXITCODE) { throw 'Cannot stage push preparation fixture.' }
        & git -C $repository -c user.name=Fixture -c user.email=fixture@example.invalid -c core.hooksPath= -c commit.gpgsign=false commit --quiet -m fixture
        if ($LASTEXITCODE) { throw 'Cannot commit push preparation fixture.' }
        Write-CiFixture $target 'push-checker.json' '{"SchemaVersion":1,"SourceIdentity":"prior-source","Sha256":"prior-bytes"}'
        $binary='Existing helper retained for diagnosis; it must never execute.'
        Write-CiFixture $target 'aarch64-pc-windows-msvc/debug/bonsai-ci.exe' $binary
        Assert-CiReject { & $adapter -Root $repository -TargetDirectory $target -Prepare } 'Expected preparation failure before Cargo.'
        Assert-Ci (-not (Test-Path -LiteralPath (Join-Path $target 'push-checker.json'))) 'A failed rebuild retained its preparation receipt.'
        Assert-CiEqual ([IO.File]::ReadAllText((Join-Path $target 'aarch64-pc-windows-msvc/debug/bonsai-ci.exe'))) $binary
    } finally { Remove-CiScratch $scratch }
 }
}
Invoke-CiCases $cases $Case
