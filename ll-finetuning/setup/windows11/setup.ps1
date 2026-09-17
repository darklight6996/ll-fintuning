Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "  Windows 11 Setup Script for ll-finetuning" -ForegroundColor Cyan
Write-Host "====================================================="

Write-Host "[*] Installing Python dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host "`n[*] Running Assessment Tools Pre-Flight Check..." -ForegroundColor Yellow
python ..\check_tools.py

Write-Host "`nSetup complete." -ForegroundColor Green
