# Tag
docker tag krimssmirk/rag:v1 asia-southeast1-docker.pkg.dev/sonic-citadel-475414-h4/rag-repo/rag:v1

# Push
docker push asia-southeast1-docker.pkg.dev/sonic-citadel-475414-h4/rag-repo/rag:v1

# Deploy
  gcloud run deploy rag-api \
  --image asia-southeast1-docker.pkg.dev/sonic-citadel-475414-h4/rag-repo/rag:v1 \
  --platform managed \
  --region asia-southeast1 \
  --allow-unauthenticated \
  --port 8080 \
  --cpu 1 \
  --memory 1024Mi \
  --min-instances 1 
  --env-vars-file env.yaml

