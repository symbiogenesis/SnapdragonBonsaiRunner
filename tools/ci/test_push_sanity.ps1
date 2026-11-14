#Requires -Version 7.6
param([string]$Case)
. (Join-Path $PSScriptRoot 'test_support.ps1')
$adapter=Join-Path $PSScriptRoot 'check_push_sanity.ps1'
$cases=[ordered]@{
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
