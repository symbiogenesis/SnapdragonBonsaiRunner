#Requires -Version 7.6
param([string]$Case,[string]$CandidateRoot)
. (Join-Path $PSScriptRoot 'test_support.ps1')
$repository=if ($CandidateRoot) { (Resolve-Path -LiteralPath $CandidateRoot).ProviderPath } else { Split-Path (Split-Path $PSScriptRoot -Parent) -Parent }
$workflows=@{}
foreach ($name in @('sandbox-qualification','tts-evaluation')) {
    $workflows[$name]=[IO.File]::ReadAllText((Join-Path $repository ".github/workflows/$name.yml")).Replace("`r`n","`n")
}
function Assert-SandboxDeviceLane([string]$Workflow) {
    $header=($Workflow -split '(?m)^jobs:\s*$',2)[0]
    # Only an explicit snapdragon target can take the shared physical-device mutex.
    Assert-Ci ($header -cmatch '(?m)^  group: \$\{\{ inputs\.target == ''snapdragon'' && ''swift-lora-snapdragon'' \|\| ''sandbox-qualification-hosted-arm64'' \}\}$') 'Expected conditional sandbox device group and hosted fallback.'
    Assert-Ci ($header -cmatch '(?m)^  cancel-in-progress: false$') 'Device captures must not be cancelled by another dispatch.'
}
$cases=[ordered]@{
 sandbox_hosted_target_never_reserves_device={
    $workflow=$workflows['sandbox-qualification']
    Assert-SandboxDeviceLane $workflow
    Assert-Ci ($workflow -cmatch '(?s)      target:\n.*?        default: hosted-arm64\n') 'sandbox-qualification must default to hosted-arm64.'
    Assert-CiReject { Assert-SandboxDeviceLane ($workflow -replace '(?m)^  group: .*$', '  group: swift-lora-snapdragon') } 'Expected conditional sandbox device group'
    Assert-CiReject { Assert-SandboxDeviceLane ($workflow.Replace('  cancel-in-progress: false','  cancel-in-progress: true')) } 'must not be cancelled'
    $jobs=Get-CiJobBlocks $workflow
    Assert-Ci ($jobs.prepare.Contains('runs-on: windows-11-arm')) 'Hosted preparation must run on windows-11-arm.'
    Assert-Ci (-not $jobs.prepare.Contains('self-hosted')) 'Hosted preparation must not reserve self-hosted runners.'
    Assert-Ci ('prepare' -cin (Get-CiDependencies $jobs.qualify)) 'qualify job must consume prepare.'
    Assert-Ci ($jobs.qualify -cmatch '(?m)^    runs-on: \$\{\{ fromJSON\(inputs\.target == ''snapdragon'' && ''\["self-hosted","Windows","ARM64","swift-bonsai-snapdragon"\]'' \|\| ''"windows-11-arm"''\) \}\}$') 'qualify must only use self-hosted runner when target is snapdragon.'
 }
 rejected_tts_target_stays_hosted={
    $workflow=$workflows['tts-evaluation']; $header=($workflow -split '(?m)^jobs:\s*$',2)[0]
    Assert-Ci (-not $header.Contains('swift-lora-snapdragon')) 'A rejected TTS option cannot reserve the physical device.'
    Assert-Ci ($header.Contains('  group: tts-evaluation-hosted-arm64'))
    $jobs=Get-CiJobBlocks $workflow
    Assert-Ci ($jobs.evaluate.Contains('runs-on: windows-11-arm'))
    Assert-Ci ($jobs.evaluate.IndexOf('Reject the unbounded owner-device matrix before setup') -lt $jobs.evaluate.IndexOf('uses: actions/checkout@'))
    Assert-Ci ($jobs.evaluate.Contains('$env:EVALUATION_TARGET -cne ''hosted-arm64'''))
 }
}
Invoke-CiCases $cases $Case
