# SEMANTIC SPACE TRADING - DEPLOYMENT GUIDE

## Quick Start (Local Testing)

```bash
# Install dependencies
pip install -r requirements_api.txt

# Run API server
python api_server.py

# Test endpoint
curl http://localhost:8000/health
```

## Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop
docker-compose down
```

## Production Deployment Options

### Option 1: AWS ECS (Recommended for scale)

```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ECR_REGISTRY
docker build -t semantic-trading .
docker tag semantic-trading:latest YOUR_ECR_REGISTRY/semantic-trading:latest
docker push YOUR_ECR_REGISTRY/semantic-trading:latest

# Deploy to ECS using Fargate
# - Create ECS cluster
# - Create task definition (2 vCPU, 4GB RAM)
# - Create service with load balancer
# - Configure auto-scaling
```

**Estimated cost:** $150-300/month

### Option 2: Heroku (Easiest)

```bash
# Install Heroku CLI
# https://devcenter.heroku.com/articles/heroku-cli

# Login and create app
heroku login
heroku create semantic-trading-api

# Add buildpack
heroku buildpacks:set heroku/python

# Deploy
git push heroku main

# Scale
heroku ps:scale web=1:standard-2x
```

**Estimated cost:** $50-100/month

### Option 3: Railway (Fast deployment)

1. Go to https://railway.app
2. Click "New Project" → "Deploy from GitHub"
3. Select repository
4. Railway auto-detects Dockerfile and deploys
5. Add custom domain

**Estimated cost:** $20-50/month

### Option 4: Digital Ocean App Platform

```bash
# Create app.yaml
# Deploy via DO dashboard or doctl CLI
doctl apps create --spec app.yaml
```

**Estimated cost:** $25-75/month

## API Testing

```bash
# Get predictions (requires API key)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo_key_12345" \
  -d '{"confidence_threshold": 0.001}'

# Check performance metrics
curl http://localhost:8000/performance \
  -H "X-API-Key: demo_key_12345"
```

## Setting Up Stripe Payments

1. Create Stripe account at https://stripe.com
2. Get API keys (Dashboard → Developers → API keys)
3. Create products:
   - Starter: $2,500/month
   - Professional: $7,500/month
   - Enterprise: $15,000/month
4. Set up webhook for subscription events
5. Generate API keys on subscription creation

## Monitoring

### Health Checks

```bash
# Endpoint health
curl http://your-domain.com/health

# Should return:
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-11-14T..."
}
```

### Logging

Add logging service (recommend Papertrail or Datadog):

```python
import logging
logging.basicConfig(level=logging.INFO)
```

### Alerts

Set up alerts for:
- API response time > 5 seconds
- Error rate > 1%
- Model loading failures
- Memory usage > 80%

## Security Checklist

- [ ] Enable HTTPS (use Let's Encrypt or cloud provider SSL)
- [ ] Rotate API keys monthly
- [ ] Rate limit endpoints (100 requests/hour per key)
- [ ] Add request authentication
- [ ] Enable CORS only for trusted domains
- [ ] Set up DDoS protection (CloudFlare)
- [ ] Regular security updates

## Performance Optimization

1. **Caching:** Cache predictions for 1 hour (markets don't change that fast)
2. **Database:** Add PostgreSQL for API key management and usage tracking
3. **CDN:** Use CloudFlare for static assets and DDoS protection
4. **Auto-scaling:** Configure based on CPU > 70%

## Cost Estimates (Monthly)

**Minimal Setup:**
- Railway/Heroku: $50
- Domain: $10
- Total: **$60/month**

**Production Setup:**
- AWS ECS (2 vCPU, 4GB): $150
- Load balancer: $20
- Domain + SSL: $10
- Monitoring (Datadog): $15
- Total: **$195/month**

**With First Customer ($2,500/month):**
- Profit: $2,305/month (3,850% margin!)

## Next Steps

1. Deploy to Railway or Heroku (fastest)
2. Set up custom domain (semantic-trading.ai or similar)
3. Configure Stripe
4. Send outreach emails
5. Get first customer!
