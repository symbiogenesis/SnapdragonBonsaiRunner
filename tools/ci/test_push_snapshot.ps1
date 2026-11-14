#Requires -Version 7.6
param([string]$Case)
. (Join-Path $PSScriptRoot 'test_support.ps1')
$adapter=Join-Path $PSScriptRoot 'check_push_snapshot.ps1'
$cases=[ordered]@{
 trusted_scripts_keep_child_scopes_and_named_arguments={
    $scratch=New-CiScratch 'bonsai-push-batch-scopes-'
    try {
        $policy=Join-Path $scratch 'policy'; $candidate=Join-Path $scratch 'candidate with spaces'
        [void][IO.Directory]::CreateDirectory($policy); [void][IO.Directory]::CreateDirectory($candidate)
        Copy-Item -LiteralPath $adapter -Destination (Join-Path $policy 'check_push_snapshot.ps1')
        Write-CiFixture $policy 'first.ps1' @'
param([string]$Root,[string]$Value)
function Get-PushLeakedFunction { throw 'A child function leaked.' }
$ErrorActionPreference='Continue'
[IO.File]::WriteAllText((Join-Path $Root 'dispatch.txt'),$Value)
exit 0
'@
        Write-CiFixture $policy 'second.ps1' @'
param([string]$Root)
if (Get-Command Get-PushLeakedFunction -ErrorAction SilentlyContinue) { throw 'A child function leaked.' }
if ($ErrorActionPreference -cne 'Stop') { throw 'A child preference leaked.' }
[IO.File]::AppendAllText((Join-Path $Root 'dispatch.txt'),'|second')
'@
        foreach ($name in @('first.ps1','second.ps1')) { Write-CiFixture $candidate $name "throw 'Candidate scripts must never execute.'" }
        $calls=@(
            @{script='first.ps1';parameters=@{Root=$candidate;Value='named value with spaces'}}
            @{script='second.ps1';parameters=@{Root=$candidate}}
        )
        $callsFile=Join-Path $scratch 'calls.json'; ConvertTo-Json -InputObject $calls -Depth 5 | Set-Content -LiteralPath $callsFile
        Push-Location -LiteralPath $candidate
        try { & (Join-Path $policy 'check_push_snapshot.ps1') -CallsFile $callsFile } finally { Pop-Location }
        Assert-CiEqual ([IO.File]::ReadAllText((Join-Path $candidate 'dispatch.txt'))) 'named value with spaces|second'
    } finally { Remove-CiScratch $scratch }
 }
 child_failures_stop_before_later_scripts={
    $scratch=New-CiScratch 'bonsai-push-batch-failures-'
    try {
        Copy-Item -LiteralPath $adapter -Destination (Join-Path $scratch 'check_push_snapshot.ps1')
        Write-CiFixture $scratch 'later.ps1' 'param([string]$Marker) [IO.File]::WriteAllText($Marker,"unexpected")'
        foreach ($failure in @(@{body='exit 13';phrase='exited 13'},@{body="throw 'Expected child rejection.'";phrase='Expected child rejection.'})) {
            Write-CiFixture $scratch 'reject.ps1' $failure.body
            $marker=Join-Path $scratch 'later.txt'
            $calls=@(@{script='reject.ps1';parameters=@{}},@{script='later.ps1';parameters=@{Marker=$marker}})
            $callsFile=Join-Path $scratch 'calls.json'; ConvertTo-Json -InputObject $calls -Depth 5 | Set-Content -LiteralPath $callsFile
            Assert-CiReject { & (Join-Path $scratch 'check_push_snapshot.ps1') -CallsFile $callsFile } $failure.phrase
            Assert-Ci (-not (Test-Path -LiteralPath $marker)) 'A rejected child allowed a later script.'
        }
    } finally { $global:LASTEXITCODE=0; Remove-CiScratch $scratch }
 }
}
Invoke-CiCases $cases $Case
