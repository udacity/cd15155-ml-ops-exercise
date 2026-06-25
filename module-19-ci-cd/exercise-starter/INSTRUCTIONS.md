# Module 19 - CI/CD Pipeline with GitHub Actions

## Overview

In this exercise you will build a full CI/CD pipeline using GitHub Actions that runs on every push to `main`. The pipeline runs code quality checks, builds and pushes a Docker image to ECR, and deploys the updated image to ECS.

---

## Prerequisites

- The ECS cluster and service must be running (`beans-api-cluster` / `beans-api-service`)
- Your ECR repository (`beans-api`) must exist

---

## 1. Create your own GitHub repository

1. Go to [github.com](https://github.com) and create a new **public** repository (e.g. `mlops-ci-cd`)
2. Copy the contents of this `exercise-starter/` folder into the root of your new repo
3. Clone your repo into the workspace:

```bash
git clone https://github.com/<your-username>/mlops-ci-cd.git
cd mlops-ci-cd
```

---

## 2. Add GitHub secrets

The pipeline needs your AWS credentials to push to ECR and deploy to ECS. Add them as repository secrets using the GitHub CLI:

```bash
gh secret set AWS_ACCESS_KEY_ID     --body "<value>"
gh secret set AWS_SECRET_ACCESS_KEY --body "<value>"
gh secret set AWS_SESSION_TOKEN     --body "<value>"
gh secret set AWS_REGION            --body "us-east-1"
gh secret set ECR_REGISTRY          --body "<account_id>.dkr.ecr.us-east-1.amazonaws.com"
```

Or add them manually under **Settings > Secrets and variables > Actions** in your GitHub repository.

---

## 3. Complete `ci.yml`

Complete the TODOs in `ci.yml`, then place it under `.github/workflows/` at the root of your repository:

```bash
mkdir -p .github/workflows
mv ci.yml .github/workflows/ci.yml
```

The pipeline has four jobs that run in order:

**`lint`**: install `ruff` and run it against `main.py` and `quality_check.py`.

**`quality-check`**: run `quality_check.py`. The pipeline fails if the script exits with code 1.

**`build`**: build the Docker image tagged with the first 8 characters of the commit SHA plus a timestamp, push it, then tag and push as `:latest`.

**`deploy`** (needs: build): download the current ECS task definition, render it with the new `:latest` image, and deploy it.

Read more about AWS actions for GitHub: https://github.com/aws-actions

---

## 4. Push and verify the pipeline

Commit and push to `main` to trigger the workflow:

```bash
git add .github/workflows/ci.yml
git commit -m "add CI/CD pipeline"
git push origin main
```

Navigate to your repository on GitHub under **Actions** to monitor each job. A successful run will deploy the updated image to your ECS service automatically.