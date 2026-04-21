# Next.js Dashboard Deployment Guide

This guide explains how to deploy the Next.js dashboard to Vercel (free tier).

## Prerequisites

- Vercel account (free at https://vercel.com)
- GitHub repository already connected
- Backend API deployed to Render

## Quick Deploy (Vercel)

### Step 1: Connect Vercel to GitHub

1. Go to https://vercel.com
2. Click **"Sign Up"** or log in
3. Click **"Add New..."** → **"Project"**
4. Click **"Import Git Repository"**
5. Connect your GitHub account if not already connected
6. Select `Mahesh2023/ollama-optimizer-v2` repository

### Step 2: Configure Vercel Project

Vercel will auto-detect the Next.js app. Configure:

- **Project Name**: `ollama-optimizer-dashboard`
- **Root Directory**: `web` (important - points to the web/ subdirectory)
- **Framework Preset**: Next.js
- **Build Command**: `npm run build` (auto-detected)
- **Output Directory**: `.next` (auto-detected)
- **Install Command**: `npm install` (auto-detected)

### Step 3: Add Environment Variables

Add these environment variables in Vercel:

```
NEXT_PUBLIC_API_URL=https://ollama-optimizer-v2.onrender.com
```

This tells the dashboard where to find the backend API.

### Step 4: Deploy

Click **"Deploy"**. Vercel will:
- Install dependencies
- Build the Next.js app
- Deploy to a URL like `https://ollama-optimizer-dashboard.vercel.app`

Deployment takes 1-2 minutes.

## Alternative: Netlify

If you prefer Netlify:

1. Go to https://app.netlify.com
2. Click **"Add new site"** → **"Import an existing project"**
3. Connect GitHub
4. Select `Mahesh2023/ollama-optimizer-v2` repository
5. Configure:
   - **Base directory**: `web`
   - **Build command**: `npm run build`
   - **Publish directory**: `web/.next`
6. Add environment variable: `NEXT_PUBLIC_API_URL=https://ollama-optimizer-v2.onrender.com`
7. Click **"Deploy site"**

## Local Development

To run the dashboard locally:

```bash
cd web
npm install
npm run dev
```

Visit `http://localhost:3000`

## Dashboard Features

Once deployed, the dashboard includes:

- **Main Page** (`/`): Real-time metrics, system info, GPU details
- **Playground** (`/playground`): Interactive chat completion testing
- **Sidebar Navigation**: Easy navigation between pages

## Updating API URL

If your Render URL changes, update the environment variable in Vercel:

1. Go to Vercel dashboard → Your project → **Settings** → **Environment Variables**
2. Update `NEXT_PUBLIC_API_URL` to the new Render URL
3. Vercel will auto-redeploy on the next git push

## Troubleshooting

### Build fails in Vercel

- Check that the **Root Directory** is set to `web`
- Verify `package.json` exists in the `web/` directory
- Check build logs in Vercel dashboard

### Dashboard can't connect to API

- Verify `NEXT_PUBLIC_API_URL` is set correctly
- Check that the Render API is running
- Test the API directly: `curl https://ollama-optimizer-v2.onrender.com/health`

### Next.js errors

- Ensure Node.js version is 18+ in Vercel settings
- Check that all dependencies are in `web/package.json`

## Custom Domain (Optional)

To use a custom domain:

1. In Vercel dashboard → **Settings** → **Domains**
2. Add your custom domain
3. Update DNS records as instructed by Vercel
4. Wait for SSL certificate to provision (5-10 minutes)
