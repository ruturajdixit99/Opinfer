# Docusaurus Setup Script for Opinfer Documentation
# This script helps set up the documentation site

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Opinfer Documentation Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Node.js is installed
Write-Host "Checking for Node.js..." -ForegroundColor Yellow
try {
    $nodeVersion = node --version 2>&1
    $npmVersion = npm --version 2>&1
    Write-Host "✓ Node.js found: $nodeVersion" -ForegroundColor Green
    Write-Host "✓ npm found: $npmVersion" -ForegroundColor Green
    Write-Host ""
} catch {
    Write-Host "✗ Node.js is not installed!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Node.js first:" -ForegroundColor Yellow
    Write-Host "1. Go to https://nodejs.org/" -ForegroundColor White
    Write-Host "2. Download the LTS version" -ForegroundColor White
    Write-Host "3. Run the installer" -ForegroundColor White
    Write-Host "4. Restart PowerShell and run this script again" -ForegroundColor White
    Write-Host ""
    Write-Host "Or use winget: winget install OpenJS.NodeJS.LTS" -ForegroundColor Cyan
    exit 1
}

# Check if we're in the right directory
if (-not (Test-Path "package.json")) {
    Write-Host "✗ package.json not found!" -ForegroundColor Red
    Write-Host "Please run this script from the docs directory" -ForegroundColor Yellow
    exit 1
}

Write-Host "Installing dependencies..." -ForegroundColor Yellow
Write-Host "This may take a few minutes..." -ForegroundColor Gray
Write-Host ""

# Install dependencies
npm install

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Setup Complete!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "To start the development server, run:" -ForegroundColor Cyan
    Write-Host "  npm start" -ForegroundColor White
    Write-Host ""
    Write-Host "The documentation will be available at:" -ForegroundColor Cyan
    Write-Host "  http://localhost:3000" -ForegroundColor White
    Write-Host ""
    
    $start = Read-Host "Start development server now? (Y/n)"
    if ($start -eq "" -or $start -eq "Y" -or $start -eq "y") {
        Write-Host ""
        Write-Host "Starting development server..." -ForegroundColor Yellow
        npm start
    }
} else {
    Write-Host ""
    Write-Host "✗ Installation failed!" -ForegroundColor Red
    Write-Host "Try running: npm cache clean --force" -ForegroundColor Yellow
    exit 1
}





