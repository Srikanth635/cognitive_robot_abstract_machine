# %% command selection

param(
    [Parameter(Mandatory = $true)][string]$Module,
    [string[]]$Arguments = @(),
    [string]$Repository = (Resolve-Path "$PSScriptRoot/../..").Path,
    [long]$MinimumFreeBytes = 8GB,
    [ulong]$MaximumBytes = 4GB,
    [string]$StatisticsPath
)

# %% resource guard

$availableBytes = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory * 1KB
if ($MaximumBytes -le 0 -or $MaximumBytes -ge $MinimumFreeBytes) {
    throw "The memory cap must be positive and below the minimum available memory."
}
if ($availableBytes -lt $MinimumFreeBytes) {
    throw "Insufficient available memory for the bounded command."
}
if ($StatisticsPath) {
    if (Test-Path -LiteralPath $StatisticsPath) {
        throw "Choose a new statistics path to preserve previous measurements."
    }
    $StatisticsPath = [IO.Path]::GetFullPath($StatisticsPath)
    [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($StatisticsPath)) | Out-Null
}
# %% bounded execution

Write-Output ("Free memory: {0:N1} GiB; job limit: {1:N1} GiB." -f ($availableBytes / 1GB), ($MaximumBytes / 1GB))
Add-Type -Path "$PSScriptRoot/memory_job.cs"
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$exitCode = [MemoryJob]::Run(
    "$Repository/repository_knowledge/.venv/Scripts/python.exe",
    "$PSScriptRoot/command_worker.py",
    $Repository,
    (@($Module) + $Arguments),
    $MaximumBytes,
    $StatisticsPath
)
exit $exitCode
