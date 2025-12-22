# Opinfer Documentation

This directory contains the Docusaurus documentation for the Opinfer project.

## Quick Start

### Prerequisites

**Node.js must be installed first!**

1. Download and install Node.js from https://nodejs.org/ (LTS version recommended)
2. Or use winget: `winget install OpenJS.NodeJS.LTS`
3. Verify installation: `node --version` and `npm --version`

### Setup (Windows)

**Option 1: Use Setup Script**
```powershell
.\setup.ps1
```

**Option 2: Manual Setup**
```powershell
# Install dependencies
npm install

# Start development server
npm start
```

### Setup (Linux/Mac)

```bash
# Install dependencies
npm install

# Start development server
npm start
```

## Development

### Start Development Server

```bash
npm start
```

This starts a local server at `http://localhost:3000` with hot-reload enabled.

### Build for Production

```bash
npm run build
```

This creates a `build` folder with static files ready for deployment.

### Serve Production Build

```bash
npm run serve
```

## Project Structure

```
docs/
├── docs/                    # Documentation markdown files
│   ├── intro.md            # Home page
│   ├── getting-started/    # Getting started guides
│   ├── concepts/           # Core concepts
│   ├── guides/             # Usage guides
│   ├── api/                # API reference
│   └── advanced/           # Advanced topics
├── src/                    # Source files (CSS, etc.)
├── static/                 # Static assets (images, etc.)
├── docusaurus.config.js    # Docusaurus configuration
├── sidebars.js            # Sidebar navigation
└── package.json           # Dependencies and scripts
```

## Adding Documentation

1. Create/edit markdown files in `docs/docs/`
2. Add entries to `sidebars.js` for navigation
3. Changes appear automatically in development mode

## Deployment

See [DEPLOYMENT.md](./DEPLOYMENT.md) for deployment instructions.

## Need Help?

- [Docusaurus Documentation](https://docusaurus.io/docs)
- [Setup Guide](./SETUP.md)
- [Deployment Guide](./DEPLOYMENT.md)
