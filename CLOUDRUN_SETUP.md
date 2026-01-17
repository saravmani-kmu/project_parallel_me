# Google Cloud Run Deployment Setup

## Prerequisites

1. **Google Cloud Project**: Create a GCP project if you don't have one
2. **Enable APIs**: Enable Cloud Run API in your GCP project
3. **Service Account**: Create a service account with necessary permissions

## Step 1: Create Service Account

```bash
# Set your project ID
gcloud config set project YOUR_PROJECT_ID

# Create service account
gcloud iam service-accounts create github-actions \
    --display-name="GitHub Actions"

# Grant necessary roles
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:github-actions@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/run.admin"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:github-actions@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/iam.serviceAccountUser"

# Create and download key
gcloud iam service-accounts keys create key.json \
    --iam-account=github-actions@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

## Step 2: Configure GitHub Secrets and Variables

### GitHub Secrets (Settings → Secrets and variables → Actions → Secrets)
- `GCP_SA_KEY`: Contents of the `key.json` file (entire JSON)

### GitHub Variables (Settings → Secrets and variables → Actions → Variables)
- `GCP_PROJECT_ID`: Your GCP project ID (e.g., `my-project-123456`)
- `GCP_REGION`: Cloud Run region (e.g., `us-central1`, `asia-south1`)
- `DOCKERHUB_USERNAME`: Your Docker Hub username (already configured)

## Step 3: Enable Required APIs

```bash
# Enable Cloud Run API
gcloud services enable run.googleapis.com

# Enable Container Registry API (if using GCR)
gcloud services enable containerregistry.googleapis.com
```

## Workflow Behavior

The deployment workflow will:
1. **Automatically trigger** after a successful Docker image build and push
2. **Can be manually triggered** via GitHub Actions UI (workflow_dispatch)
3. Deploy the latest image from Docker Hub to Cloud Run
4. Configure the service with:
   - Port: 8000
   - Memory: 512Mi
   - CPU: 1
   - Min instances: 0 (scales to zero)
   - Max instances: 10
   - Public access (--allow-unauthenticated)

## Customization

Edit `.github/workflows/deploy-to-cloudrun.yml` to adjust:
- Memory and CPU limits
- Scaling parameters (min/max instances)
- Port number
- Authentication settings
- Environment variables

## Testing

After setup, push to main branch:
```bash
git add .
git commit -m "Add Cloud Run deployment"
git push origin main
```

This will:
1. Build and push Docker image
2. Automatically deploy to Cloud Run
3. Display the service URL in the workflow logs
