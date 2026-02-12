# Streamlit.io Deployment & Keep-Alive Guide

Complete steps to deploy the Streamboard dashboard to Streamlit.io and keep it awake.

---

## Part 1: Deploy the App to Streamlit.io

### Step 1: Push Your Code to GitHub

Your repo must be on GitHub. If not already:

```bash
git add .
git commit -m "Add Streamboard dashboard"
git push origin main
```

### Step 2: Run Preprocessing Locally (First Time)

Generate the CSV files the dashboard needs. From the project root:

```bash
cd streamboard
python preprocess_streamboard.py
cd ..
```

Commit and push the generated files in `streamboard/data/`:

```bash
git add streamboard/data/
git commit -m "Add preprocessing data"
git push
```

### Step 3: Deploy on Streamlit.io

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **"New app"**.
3. Fill in:
   - **Repository**: `YOUR-USERNAME/hurricanepaths`
   - **Branch**: `main`
   - **Main file path**: `streamboard/app.py`
   - **App URL**: Choose a subdomain (e.g. `hurricanepaths`)
4. Click **Deploy**.

### Step 4: Wait for Deployment

Streamlit builds and runs your app. Your URL will be:

`https://hurricanepaths.streamlit.app/`

---

## Part 2: Keep the App Awake

Streamlit Community Cloud puts apps to sleep after ~1 hour of inactivity. Use one of these options to keep it running.

### Option A: GitHub Actions (Recommended)

A workflow in this repo pings your app every 30 minutes.

1. Ensure the workflow exists at `.github/workflows/keep-streamlit-awake.yml`.
2. The URL is already set to `https://hurricanepaths.streamlit.app/`.
3. Push to GitHub. The workflow runs automatically on a schedule.
4. You can trigger it manually from the **Actions** tab in your GitHub repo.

### Option B: UptimeRobot (No Code)

1. Sign up at [uptimerobot.com](https://uptimerobot.com) (free).
2. Click **Add New Monitor**.
3. **Monitor Type**: HTTP(s)
4. **URL**: `https://hurricanepaths.streamlit.app/`
5. **Monitoring Interval**: 5 minutes
6. Click **Create Monitor**.

### Option C: Manual Ping Script

Run the included script on a schedule (e.g. Windows Task Scheduler, cron):

```bash
cd keep_alive
python ping.py
```

To run every 30 minutes on Windows Task Scheduler: create a basic task, set trigger to "Repeat task every 30 minutes," and action to run `python` with argument `ping.py` in the `keep_alive` folder.

---

## Summary

| Step | Action |
|------|--------|
| 1 | Push code to GitHub |
| 2 | Run `preprocess_streamboard.py`, commit data |
| 3 | Deploy at share.streamlit.io |
| 4 | Set up keep-alive (GitHub Actions, UptimeRobot, or ping script) |
