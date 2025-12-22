# Docusaurus Deployment Guide

This guide explains how to deploy the Opinfer documentation site.

## Local Development

1. Install dependencies:

```bash
cd docs
npm install
```

2. Start the development server:

```bash
npm start
```

This starts a local server at `http://localhost:3000`

## Build for Production

```bash
npm run build
```

This creates a `build` folder with static files ready for deployment.

## Deploy to GitHub Pages

### Option 1: Automatic Deployment

The site can be automatically deployed using GitHub Actions. Create `.github/workflows/docs.yml`:

```yaml
name: Deploy Docs

on:
  push:
    branches:
      - main
    paths:
      - 'docs/**'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: 18
          cache: 'npm'
          cache-dependency-path: docs/package-lock.json
      
      - name: Install dependencies
        working-directory: ./docs
        run: npm ci
      
      - name: Build website
        working-directory: ./docs
        run: npm run build
      
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        if: github.ref == 'refs/heads/main'
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs/build
```

### Option 2: Manual Deployment

1. Build the site:

```bash
cd docs
npm run build
```

2. Deploy using `gh-pages` package:

```bash
npm install -g gh-pages
gh-pages -d build
```

## Configuration

Update `docusaurus.config.js` with your repository details:

```javascript
url: 'https://yourusername.github.io',
baseUrl: '/YourRepository/',
organizationName: 'yourusername',
projectName: 'YourRepository',
```

## Custom Domain

To use a custom domain:

1. Add a `CNAME` file to `docs/static/` with your domain
2. Configure DNS settings in your domain provider
3. Update `docusaurus.config.js` with your domain





