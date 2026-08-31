# Quick activate wrapper for PowerShell with automatic ExecutionPolicy Bypass for the current session
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Definition
& "$scriptPath\venv\Scripts\Activate.ps1"
