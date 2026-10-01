# Step-by-Step Render Deployment Guide
## Crop Disease Detection System (Flask Backend & ML Inference)

This guide provides exact step-by-step instructions to deploy the **AgriScan AI - Crop Disease Detection System** to [Render](https://render.com) for free cloud hosting.

---

### 📋 Prerequisites
1. A GitHub account.
2. A free account on [Render.com](https://render.com).
3. Git installed on your computer.

---

### Step 1: Push Code to GitHub
1. Open terminal in your project directory:
   ```bash
   cd C:\Users\PC\.gemini\antigravity\scratch\crop_disease_detection
   ```
2. Initialize Git repository and commit changes:
   ```bash
   git init
   git add .
   git commit -m "Add Approach 1 Binary Gate Model, PDF generator, and Render deployment config"
   ```
3. Create a new repository on GitHub named `crop_disease_detection`.
4. Link local repository and push:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/crop_disease_detection.git
   git branch -M main
   git push -u origin main
   ```

---

### Step 2: Create a Web Service on Render
1. Log in to [dashboard.render.com](https://dashboard.render.com).
2. Click on **New +** button in the top right corner and select **Web Service**.
3. Select **Build and deploy from a Git repository** and click **Next**.
4. Connect your GitHub account and select your `crop_disease_detection` repository.

---

### Step 3: Configure Web Service Settings
Fill in the service details on Render:

- **Name:** `crop-disease-detection` (or your preferred unique service name)
- **Region:** Select the region closest to you (e.g., Singapore / Frankfurt / Oregon)
- **Branch:** `main`
- **Root Directory:** (Leave blank)
- **Runtime:** `Python 3`
- **Build Command:**
  ```bash
  pip install -r requirements.txt
  ```
- **Start Command:**
  ```bash
  gunicorn app:app
  ```
- **Instance Type:** Select **Free** tier.

---

### Step 4: Environment Variables (Optional but Recommended)
In the **Environment Variables** section on Render, add:
- `PYTHON_VERSION`: `3.12.0`
- `TF_CPP_MIN_LOG_LEVEL`: `2`
- `TF_ENABLE_ONEDNN_OPTS`: `0`

---

### Step 5: Deploy & Verify
1. Click **Create Web Service**.
2. Render will trigger an automated build, install dependencies from `requirements.txt`, and start `app.py` via Gunicorn.
3. Once the log says `Your service is live 🎉`, click the generated URL (e.g., `https://crop-disease-detection.onrender.com`).
4. Upload a leaf image to test Stage 1 Binary Gate verification and Stage 2 disease classification!
