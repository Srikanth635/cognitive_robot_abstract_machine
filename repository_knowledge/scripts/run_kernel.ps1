# %% kernel connection

param(
    [Parameter(Mandatory = $true)][string]$ConnectionFile
)

# %% bounded kernel

& "$PSScriptRoot/run_command.ps1" -Module ipykernel_launcher -Arguments @('-f', $ConnectionFile)
exit $LASTEXITCODE
