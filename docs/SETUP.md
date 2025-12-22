# Docusaurus Setup Guide

This guide will help you set up the Docusaurus documentation site for Opinfer.

## Prerequisites

You need to install Node.js (which includes npm) first.

### Install Node.js

#### Windows

1. **Download Node.js**:
   - Go to https://nodejs.org/
   - Download the LTS version (recommended)
   - Choose the Windows Installer (.msi)

2. **Install Node.js**:
   - Run the installer
   - Follow the installation wizard
   - Make sure to check "Add to PATH" option

3. **Verify Installation**:
   Open PowerShell/Command Prompt and run:
   ```powershell
   node --version
   npm --version
   ```
   Both commands should show version numbers.

#### Alternative: Using Chocolatey (Windows)

If you have Chocolatey installed:
```powershell
choco install nodejs
```

#### Alternative: Using winget (Windows)

```powershell
winget install OpenJS.NodeJS.LTS
```

## Setup Documentation Site

Once Node.js is installed, follow these steps:

### Step 1: Navigate to docs directory

```powershell
cd SARD\MotionGated\docs
```

### Step 2: Install Dependencies

```powershell
npm install
```

This will install all required packages (takes 2-5 minutes).

### Step 3: Start Development Server

```powershell
npm start
```

This will:
- Start a local development server
- Open your browser to `http://localhost:3000`
- Enable hot-reload (changes appear instantly)

### Step 4: Build for Production (Optional)

To create a production build:

```powershell
npm run build
```

The built site will be in the `build` folder.

## Troubleshooting

### Port 3000 Already in Use

If port 3000 is busy, Docusaurus will use the next available port (3001, 3002, etc.).

### npm install Fails

Try clearing npm cache:
```powershell
npm cache clean --force
npm install
```

### Permission Errors

On Windows, run PowerShell as Administrator if you encounter permission errors.

### Slow Installation

The first `npm install` can take several minutes. Be patient!

## Quick Commands Reference

```powershell
# Install dependencies
npm install

# Start development server
npm start

# Build for production
npm run build

# Clear cache and reinstall
npm cache clean --force
npm install
```

## Next Steps

After setup is complete:

1. **View Documentation**: Open `http://localhost:3000` in your browser
2. **Edit Documentation**: Edit files in `docs/docs/` folder
3. **See Changes**: Changes appear automatically (hot-reload)
4. **Deploy**: See `DEPLOYMENT.md` for deployment instructions

## Need Help?

- Check Docusaurus docs: https://docusaurus.io/docs
- Check Node.js installation: https://nodejs.org/





