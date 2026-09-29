# Navigate to project root
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -Path "$scriptDir\.."

$screenshotPath = "C:\Users\adars\.gemini\antigravity-ide\brain\881f8c9b-7d25-49c8-90d0-a62294c83b54\simulation_screenshot.png"

Write-Host "Running SUMO GUI headless to capture a screenshot..."
sumo-gui -c map.sumocfg --start -Q --delay 100 --window-size 1920,1080 --screenshot-output $screenshotPath --end 10

Write-Host "Screenshot saved to $screenshotPath"
