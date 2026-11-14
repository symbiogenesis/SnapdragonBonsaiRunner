#Requires -Version 7.6
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$RepositoryRoot,
    [Parameter(Mandatory)][string]$AssetDirectory,
    [Parameter(Mandatory)][string]$Tag,
    [Parameter(Mandatory)][string]$SourceRevision,
    [Parameter(Mandatory)][bool]$Prerelease
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
. (Join-Path $PSScriptRoot '../../scripts/release/Release-Common.ps1')
. (Join-Path $PSScriptRoot '../../scripts/release/Release-Assets.ps1')
. (Join-Path $PSScriptRoot '../../scripts/release/Release-Publication.ps1')
$RepositoryRoot = Resolve-ReleasePath $RepositoryRoot
$AssetDirectory = Resolve-ReleasePath $AssetDirectory
Publish-PortableRelease -RepositoryRoot $RepositoryRoot -AssetDirectory $AssetDirectory -Tag $Tag -SourceRevision $SourceRevision -Prerelease $Prerelease -Repository $env:GITHUB_REPOSITORY
