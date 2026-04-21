# Render Deployment Guide (ngrok + Local Ollama)

This guide explains how to deploy the Ollama Optimizer v2 to Render while keeping Ollama running locally on your Mac M2.

## Architecture

```
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│   Local Mac M2  │         │   ngrok Tunnel  │         │   Render Cloud  │
│                 │         │                 │         │                 │
│  Ollama:11434   │────────▶│  Public HTTPS   │────────▶│  Optimizer API  │
│                 │         │                 │         │     :8000       │
└─────────────────┘         └─────────────────┘         └─────────────────┘
```

## Prerequisites

1. **Ollama running locally**
   ```bash
   # Start Ollama (if not already running)
   ollama serve
   ```

2. **ngrok installed**
   ```bash
   # On macOS with Homebrew
   brew install ngrok

   # Or download from https://ngrok.com/download
   ```

3. **Render account** (free tier at https://render.com)

4. **GitHub repository** with the code already pushed

---

## Step 1: Start ngrok Tunnel

Expose your local Ollama to the internet:

```bash
ngrok http 11434
```

You'll see output like:
```
Forwarding  https://abc123.ngrok-free.app -> http://localhost:11434
```

**Copy the ngrok HTTPS URL** (e.g., `https://abc123.ngrok-free.app`) — you'll need this in Step 3.

**Keep ngrok running** in a separate terminal window.

---

## Step 2: Update render.yaml

Edit `deploy/render.yaml` and replace the placeholder `OLLAMA_BASE_URL` with your ngrok URL:

```yaml
services:
  - type: web
    name: ollama-optimizer
    env: docker
    dockerfilePath: ./deploy/Dockerfile
    dockerContext: .
    plan: starter
    healthCheckPath: /health
    envVars:
      - key: REDIS_URL
        fromService:
          type: redis
          name: ollama-cache
          property: connectionString
      - key: OLLAMA_BASE_URL
        value: https://abc123.ngrok-free.app  # ← REPLACE WITH YOUR NGROK URL
      - key: LOG_LEVEL
        value: INFO

  - type: redis
    name: ollama-cache
    plan: starter
    ipAllowList: []
```

Commit and push this change:

```bash
git add deploy/render.yaml
git commit -m "Update render.yaml with ngrok URL"
git push origin feature/scaffold
```

---

## Step 3: Deploy to Render

1. Go to https://dashboard.render.com
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub account
4. Select `Mahesh2023/ollama-optimizer-v2` repository
5. Select the `feature/scaffold` branch
6. Render will auto-detect the `render.yaml` file
7. Click **"Deploy Web Service"**

Render will:
- Build the Docker image from `deploy/Dockerfile`
- Deploy the optimizer API
- Create a Redis instance for caching
- Configure environment variables from render.yaml

**Deployment takes 2-5 minutes.**

---

## Step 4: Get Your Render URL

After deployment completes, Render will provide:
- API URL: `https://ollama-optimizer.onrender.com`
- Redis URL: automatically configured

**Test the deployment:**

```bash
# Health check
curl https://ollama-optimizer.onrender.com/health

# System info
curl https://ollama-optimizer.onrender.com/system

# Metrics
curl https://ollama-optimizer.onrender.com/metrics
```

---

## Step 5: Test Full Stack

Test that the optimizer can reach your local Ollama via ngrok:

```bash
# Test chat completion (replace with your Render URL)
curl -X POST https://ollama-optimizer.onrender.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

If successful, you'll see a response from your local Ollama.

---

## Important Notes

### ngrok URL Changes

ngrok free tier URLs change every time you restart ngrok. When this happens:

1. Get the new ngrok URL
2. Update `deploy/render.yaml`
3. Push to GitHub
4. Render will auto-redeploy on push

**For production:** Consider upgrading to ngrok paid tier ($19/month) for static domains, or use a different deployment strategy.

### ngrok Autostart (macOS)

To keep ngrok running after system restarts:

```bash
# Create a LaunchAgent
cat > ~/Library/LaunchAgents/com.ngrok.ngrok.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.ngrok.ngrok</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/ngrok</string>
        <string>http</string>
        <string>11434</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
</dict>
</plist>
EOF

# Load it
launchctl load ~/Library/LaunchAgents/com.ngrok.ngrok.plist
```

### Render Free Tier Limits

- 512MB RAM
- 750 hours/month runtime
- No GPU (Ollama runs locally)
- Sleeps after 15 min inactivity (wakes on request)

### Security Considerations

- ngrok free tier provides HTTPS but no authentication
- Anyone with the ngrok URL can access your local Ollama
- For production, use ngrok paid tier with authentication or a VPN
- Consider adding API key authentication to the optimizer

---

## Troubleshooting

### ngrok connection issues

```bash
# Check ngrok is running
curl http://localhost:4040/api/tunnels

# Restart ngrok
pkill ngrok
ngrok http 11434
```

### Render build failures

Check Render logs in the dashboard:
- Build logs: Docker build errors
- Service logs: Runtime errors
- Common issue: Missing environment variables

### Optimizer can't reach Ollama

1. Verify ngrok is running: `curl http://localhost:11434/api/tags`
2. Check ngrok URL in Render environment variables
3. Check Render service logs for connection errors
4. Test ngrok URL directly: `curl https://your-ngrok-url.ngrok-free.app/api/tags`

### Redis connection issues

- Render auto-creates Redis instance
- Check Redis is running in Render dashboard
- Verify `REDIS_URL` is set in Render environment variables

---

## Alternative: Railway Deployment

If Render doesn't work well, try Railway (similar approach):

1. Create account at https://railway.app
2. Connect GitHub repository
3. Create new project from repo
4. Railway will detect Dockerfile
5. Add environment variables:
   - `OLLAMA_BASE_URL`: your ngrok URL
   - `REDIS_URL`: Railway will provide Redis
6. Deploy

Railway has better support for external service connections.

---

## Next Steps

Once deployed:

1. **Deploy the Next.js dashboard** to Vercel/Netlify
2. **Update dashboard API URL** to point to Render
3. **Set up monitoring** with Langfuse/MLflow
4. **Configure alerts** for drift detection

See `docs/QUICKSTART.md` for full usage instructions.
