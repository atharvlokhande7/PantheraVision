# Accessing PantheraVision

## 1. Web Interface
You can access the live stream and status page via your browser.

- **Live Stream**: `http://<YOUR_EC2_IP>:5000/video`
- **Status Page**: `http://<YOUR_EC2_IP>:5000/`
- **Health Check**: `http://<YOUR_EC2_IP>:5000/health`

## 2. Prerequisites (AWS Security Group)
Ensure your AWS EC2 Security Group allows inbound traffic on **Port 5000**.

1. Go to AWS Console > EC2 > Instances.
2. Select your instance > Security > Security Groups.
3. Edit Inbound Rules > Add Rule:
   - **Type**: Custom TCP
   - **Port**: `5000`
   - **Source**: `0.0.0.0/0` (or your IP)

## 3. SSH Tunneling (Alternative)
If you don't want to open port 5000 publicly, you can tunnel connections:

```bash
# Run this on your local machine
ssh -L 5000:localhost:5000 -i <your-key.pem> ubuntu@<YOUR_EC2_IP>
```

Then access locally: [http://localhost:5000/video](http://localhost:5000/video)
