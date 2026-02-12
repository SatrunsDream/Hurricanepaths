# Hurricane Model Comparison Dashboard

A React dashboard for visualizing and comparing hurricane track forecasting model performance (Null Model vs Kalman Filter). Built with **Tailwind CSS**, **Plotly**, and **Lucide icons**.

## Quick Start

### 1. Install dependencies

```bash
cd dashboard
npm install
```

### 2. Ensure data files exist

Copy the CSV files from the Streamboard preprocessing output:

```
streamboard/data/*.csv  →  dashboard/public/data/
```

Required files:
- `model_comparison_summary.csv`
- `error_distributions.csv`
- `storm_performance.csv`
- `model_comparison_detailed.csv`
- `storm_metadata.csv`

Optional: Copy figures for the Methodology page:

```
figures/*  →  dashboard/public/figures/
```

### 3. Run the dashboard

```bash
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

### 4. Build for production

```bash
npm run build
```

The output is in `dist/`. Preview with:

```bash
npm run preview
```

---

## Deploy to Vercel

### Option A: Deploy from the dashboard folder

1. Install the Vercel CLI: `npm i -g vercel`
2. From the `dashboard` folder, run: `vercel`
3. Follow the prompts (link to existing project or create new)
4. Your app will be live at `https://your-project.vercel.app`

### Option B: Deploy from GitHub

1. Push this repo to GitHub
2. Go to [vercel.com](https://vercel.com) and sign in with GitHub
3. Click **New Project** and import your repository
4. Set the **Root Directory** to `dashboard`
5. Build settings:
   - **Framework Preset**: Vite
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
6. Click **Deploy**

### Environment

No environment variables are required. The dashboard loads data from the `public/` folder at build time.

---

## Project Structure

```
dashboard/
├── public/
│   ├── data/          # CSV files (copy from streamboard/data/)
│   └── figures/       # Optional figures for Methodology page
├── src/
│   ├── components/    # Layout, Chart wrapper
│   ├── lib/           # Data loading
│   ├── pages/         # Model Comparison, Error Distributions, etc.
│   └── types.ts
├── package.json
└── README.md
```

## Pages

1. **Model Comparison** – RMSE/mean error comparison, improvement metrics
2. **Error Distributions** – Histograms, KDE, cumulative distributions, box plots
3. **Storm Performance** – Scatter plot, pie chart, basin comparison
4. **Storm Tracker** – Individual storm error trends
5. **Methodology** – Project overview and figures
