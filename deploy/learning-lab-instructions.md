
# ✅ Complete AWS Learning Lab Deployment Guide

## 🔧 Prerequisites (AWS Learning Lab)
- ✅ EC2 instance (Ubuntu-based, t3.medium recommended)
- ✅ Web terminal access via EC2 Instance Connect
- ✅ Your project code in GitHub OR local files

## 🚀 Step 1: Access EC2 via Web Terminal
1. Open **AWS Console**
2. Go to **EC2** → **Instances**
3. Click your instance → **Connect** → **EC2 Instance Connect**
4. **Connect** to open web terminal

## 📦 Step 2: Get Your Code on EC2

### Option A: From GitHub (Recommended)
```bash
sudo apt install git -y
git clone https://github.com/JosepBonetSaez/synthesis-II-With-Web.git
cd synthesis-II-With-Web
```

### Option B: Upload from Local
```bash
# From your local machine:
scp -i your-key.pem -r ./your-app-folder ubuntu@<EC2_PUBLIC_IP>:~/

# Then on EC2:
cd your-app-folder
```

## 🐳 Step 3: Deploy with One Command
```bash
# Make the deployment script executable
chmod +x deploy/ec2-deploy.sh

# Run the deployment (this handles everything!)
./deploy/ec2-deploy.sh
```

## 🔓 Step 4: Configure Security Group
Follow the instructions in `deploy/security-group-setup.md`

## ✅ Step 5: Test Your Deployment

### Health Check
```bash
curl http://<EC2_PUBLIC_IP>/api/health
```
Expected: `{"status":"healthy","message":"Server is running"}`

### Translation Test
```bash
curl -X POST http://<EC2_PUBLIC_IP>/api/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "source_lang": "english", "target_lang": "spanish"}'
```

### Web Interface
Open in browser: `http://<EC2_PUBLIC_IP>`

## 📈 Monitoring & Management

### View Logs
```bash
docker logs -f translation-app
```

### Restart App
```bash
docker restart translation-app
```

### Stop App
```bash
docker stop translation-app
```

### Check Container Status
```bash
docker ps -a
```

## 🐛 Troubleshooting

### If models are still downloading:
- Wait 1-5 minutes for first run
- Check logs: `docker logs -f translation-app`

### If port 80 is busy:
```bash
# Run on port 8080 instead
docker stop translation-app
docker rm translation-app
docker run -d --name translation-app -p 8080:5000 --restart unless-stopped translation-app:latest
```

### If build fails:
- Check available disk space: `df -h`
- Ensure Docker is running: `docker ps`

