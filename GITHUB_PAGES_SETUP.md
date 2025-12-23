# GitHub Pages Setup - Action Required

## Current Status

✅ GitHub Actions workflow created and pushed  
✅ Docusaurus documentation is configured  
❌ GitHub Pages needs to be enabled in repository settings

## Required Steps (Must be done via GitHub Web Interface)

### Step 1: Enable GitHub Pages

1. Go to: https://github.com/ruturajdixit99/Opinfer/settings/pages
2. Under "Source", select: **"GitHub Actions"** (NOT "Deploy from a branch")
3. Click "Save"

### Step 2: Trigger the Workflow

The workflow will automatically run when:
- Changes are pushed to `main`, `master`, or `2025-12-22-xkve` branches
- Files in `docs/` folder are modified
- The workflow file itself is updated

Or manually trigger it:
1. Go to: https://github.com/ruturajdixit99/Opinfer/actions
2. Click "Deploy Docusaurus Docs to GitHub Pages"
3. Click "Run workflow" → Select branch → "Run workflow"

### Step 3: Wait for Deployment

1. Monitor the workflow run in the Actions tab
2. Once complete (green checkmark), your site will be live at:
   **https://ruturajdixit99.github.io/Opinfer/**

## Verification

After enabling GitHub Pages and the workflow completes:

- ✅ Visit: https://ruturajdixit99.github.io/Opinfer/
- ✅ Should see the Docusaurus documentation homepage
- ❌ If still 404, wait 1-2 minutes for DNS propagation

## Troubleshooting

### Still seeing 404?

1. **Check workflow status**: Go to Actions tab, ensure workflow completed successfully
2. **Check Pages settings**: Verify "Source" is set to "GitHub Actions"
3. **Wait a few minutes**: GitHub Pages can take 1-5 minutes to deploy
4. **Check the workflow logs**: Look for any errors in the deployment step

### Workflow not running?

- Ensure you're pushing to `main`, `master`, or `2025-12-22-xkve` branch
- Ensure files in `docs/` folder are being modified
- Check Actions tab to see if workflow is queued/running

## Configuration

The site is configured for:
- **URL**: https://ruturajdixit99.github.io
- **Base URL**: /Opinfer/
- **Full URL**: https://ruturajdixit99.github.io/Opinfer/

