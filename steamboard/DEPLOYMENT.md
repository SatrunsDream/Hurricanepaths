# Deploying Steamboard to Streamlit Community Cloud

This guide will walk you through deploying your Streamlit dashboard to Streamlit Community Cloud (free hosting).

## Prerequisites

1. **GitHub Account** - If you don't have one, sign up at [github.com](https://github.com)
2. **Streamlit Account** - Sign up at [share.streamlit.io](https://share.streamlit.io) (free)

## Step 1: Prepare Your Repository

### Option A: If your code is already on GitHub

1. Make sure all your files are committed and pushed:
   ```bash
   git add .
   git commit -m "Prepare for Streamlit deployment"
   git push origin main
   ```

### Option B: If you need to push to GitHub

1. **Create a new repository on GitHub** (if you don't have one):
   - Go to [github.com/new](https://github.com/new)
   - Name it (e.g., `hurricanepaths`)
   - Choose public or private
   - **Don't** initialize with README (you already have files)

2. **Push your code**:
   ```bash
   cd "c:\Users\sardo\OneDrive\Desktop\Classes\CSE150A\hurricanepaths"
   git remote add origin https://github.com/YOUR_USERNAME/hurricanepaths.git
   git branch -M main
   git push -u origin main
   ```

## Step 2: Ensure Required Files Are Present

Make sure these files exist in your repository:

- ✅ `steamboard/app.py` - Your main Streamlit app
- ✅ `steamboard/requirements.txt` - Python dependencies
- ✅ `steamboard/methodology.md` - Methodology content
- ✅ `steamboard/*.csv` - Your data files (5 CSV files)
- ✅ `figures/` - Your figure images (referenced in methodology.md)

**Important**: The CSV files and figures need to be in your repository for the app to work!

## Step 3: Deploy to Streamlit Community Cloud

1. **Sign up/Login**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

2. **Deploy New App**:
   - Click "New app" button
   - Select your repository (`hurricanepaths` or whatever you named it)
   - Select branch: `main` (or `master` if that's your default)

3. **Configure App Path**:
   - **Main file path**: `steamboard/app.py`
   - **App URL**: Choose a custom name (e.g., `hurricane-model-comparison`)
   - Click "Deploy!"

4. **Wait for Deployment**:
   - Streamlit will install dependencies from `requirements.txt`
   - This usually takes 2-5 minutes
   - You'll see build logs in real-time

## Step 4: Handle Data Files

Since your app loads CSV files from the `steamboard/` directory, make sure:

1. **CSV files are committed to git**:
   ```bash
   git add steamboard/*.csv
   git commit -m "Add CSV data files"
   git push
   ```

2. **Figures directory is committed**:
   ```bash
   git add figures/
   git commit -m "Add figure images"
   git push
   ```

## Step 5: Update App After Changes

Whenever you make changes:

1. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Update dashboard"
   git push
   ```

2. Streamlit will automatically redeploy (or you can manually trigger it from the dashboard)

## Troubleshooting

### "Module not found" errors
- Check that all dependencies are in `requirements.txt`
- Make sure version numbers are compatible

### "File not found" errors
- Ensure CSV files and figures are committed to git
- Check file paths in `app.py` are relative (not absolute)

### App won't start
- Check the build logs in Streamlit dashboard
- Verify `app.py` is in the correct location
- Make sure `requirements.txt` is in the same directory as `app.py` or specify the path

### Data files too large
- Streamlit Community Cloud has file size limits
- If your CSV files are very large (>100MB), consider:
  - Compressing them
  - Using a database (PostgreSQL, etc.)
  - Loading from external storage (S3, etc.)

## Alternative Deployment Options

If Streamlit Community Cloud doesn't work for you:

1. **Heroku** - Paid option, more control
2. **AWS/GCP/Azure** - Enterprise solutions
3. **Docker** - Self-hosted option
4. **Streamlit for Teams** - Paid Streamlit hosting

## Your App URL

Once deployed, your app will be available at:
```
https://YOUR_APP_NAME.streamlit.app
```

Or if you chose a custom name:
```
https://hurricane-model-comparison.streamlit.app
```

## Need Help?

- Streamlit Docs: [docs.streamlit.io](https://docs.streamlit.io)
- Community Forum: [discuss.streamlit.io](https://discuss.streamlit.io)
- GitHub Issues: Check your repo's issues section
