Get-Process | Where-Object { $_.ProcessName -like 'llama*' } | ForEach-Object {
  Write-Output ("killing " + $_.Id + " " + $_.ProcessName)
  Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
}
