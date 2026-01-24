# Deployment Guide

This guide covers deploying the Self-Training Sentiment Analysis System to various environments.

## 📋 Prerequisites

- Python 3.8 or higher
- MySQL 5.7+ or MySQL 8.0+
- 2GB+ RAM (4GB+ recommended)
- 10GB+ disk space for models and logs

## 🚀 Local Development Deployment

### Step 1: Clone Repository

```bash
git clone https://github.com/your-username/Setiment_analysis.git
cd Setiment_analysis
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Database

1. **Create MySQL database**:
```sql
CREATE DATABASE sentiment_analysis;
```

2. **Create tables** (see `database/data_flag.sql`):
```sql
CREATE TABLE twitter_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    clean_text TEXT,
    category INT
);

CREATE TABLE data_flag (
    id INT PRIMARY KEY,
    last_row_count INT DEFAULT 0
);
```

3. **Update `config.py`**:
```python
DB_CONFIG = {
    "host": "localhost",
    "user": "your_username",
    "password": "your_password",
    "database": "sentiment_analysis"
}
```

### Step 5: Initial Training

```bash
python -c "from ml.trainer import train_pipeline; train_pipeline(force_retrain=True)"
```

### Step 6: Run Application

```bash
python app.py
```

Access at: `http://localhost:5000`

### Step 7: Start Scheduler (Optional)

In a separate terminal:
```bash
python scheduler.py
```

---

## 🐳 Docker Deployment

### Dockerfile

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    default-libmysqlclient-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 5000

# Run application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "app:app"]
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    environment:
      - DB_HOST=db
      - DB_USER=root
      - DB_PASSWORD=password
      - DB_NAME=sentiment_analysis
    depends_on:
      - db
    volumes:
      - ./models_store:/app/models_store
      - ./logs:/app/logs
      - ./experiments:/app/experiments

  db:
    image: mysql:8.0
    environment:
      MYSQL_ROOT_PASSWORD: password
      MYSQL_DATABASE: sentiment_analysis
    volumes:
      - mysql_data:/var/lib/mysql
      - ./database/data_flag.sql:/docker-entrypoint-initdb.d/init.sql

  scheduler:
    build: .
    command: python scheduler.py
    environment:
      - DB_HOST=db
      - DB_USER=root
      - DB_PASSWORD=password
      - DB_NAME=sentiment_analysis
    depends_on:
      - db
      - web
    volumes:
      - ./models_store:/app/models_store
      - ./logs:/app/logs

volumes:
  mysql_data:
```

### Build and Run

```bash
# Build
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## ☁️ Production Deployment

### Option 1: Gunicorn + Nginx

#### 1. Install Gunicorn

```bash
pip install gunicorn
```

#### 2. Create Gunicorn Config (`gunicorn_config.py`)

```python
bind = "0.0.0.0:8000"
workers = 4
worker_class = "sync"
worker_connections = 1000
timeout = 30
keepalive = 2
max_requests = 1000
max_requests_jitter = 50
```

#### 3. Run with Gunicorn

```bash
gunicorn -c gunicorn_config.py app:app
```

#### 4. Nginx Configuration

Create `/etc/nginx/sites-available/sentiment-analysis`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /static {
        alias /path/to/Setiment_analysis/static;
    }
}
```

Enable site:
```bash
sudo ln -s /etc/nginx/sites-available/sentiment-analysis /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### Option 2: Systemd Service

Create `/etc/systemd/system/sentiment-analysis.service`:

```ini
[Unit]
Description=Sentiment Analysis Web Application
After=network.target mysql.service

[Service]
Type=notify
User=www-data
Group=www-data
WorkingDirectory=/path/to/Setiment_analysis
Environment="PATH=/path/to/Setiment_analysis/venv/bin"
ExecStart=/path/to/Setiment_analysis/venv/bin/gunicorn -c gunicorn_config.py app:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable sentiment-analysis
sudo systemctl start sentiment-analysis
sudo systemctl status sentiment-analysis
```

### Option 3: Cloud Platforms

#### Heroku

1. **Create `Procfile`**:
```
web: gunicorn app:app
worker: python scheduler.py
```

2. **Create `runtime.txt`**:
```
python-3.9.16
```

3. **Deploy**:
```bash
heroku create your-app-name
git push heroku main
```

#### AWS Elastic Beanstalk

1. **Install EB CLI**:
```bash
pip install awsebcli
```

2. **Initialize**:
```bash
eb init
eb create sentiment-analysis-env
```

3. **Deploy**:
```bash
eb deploy
```

#### Google Cloud Run

1. **Create `Dockerfile`** (see Docker section)

2. **Deploy**:
```bash
gcloud run deploy sentiment-analysis \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

---

## 🔒 Security Considerations

### 1. Environment Variables

Use environment variables for sensitive data:

```python
import os

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "sentiment_analysis")
}
```

Create `.env` file:
```
DB_HOST=localhost
DB_USER=your_user
DB_PASSWORD=your_password
DB_NAME=sentiment_analysis
```

### 2. HTTPS/SSL

Use SSL certificates (Let's Encrypt):

```bash
sudo certbot --nginx -d your-domain.com
```

### 3. Firewall

```bash
# Allow only necessary ports
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### 4. Database Security

- Use strong passwords
- Restrict database access by IP
- Enable SSL for database connections
- Regular backups

### 5. Application Security

- Input validation
- Rate limiting
- CORS configuration
- Error message sanitization

---

## 📊 Monitoring

### 1. Application Logs

Monitor logs:
```bash
tail -f logs/system.log
```

### 2. System Metrics

Use monitoring tools:
- **Prometheus** + **Grafana**
- **Datadog**
- **New Relic**

### 3. Health Checks

Create `/health` endpoint:

```python
@app.route('/health')
def health():
    return jsonify({"status": "healthy"}), 200
```

### 4. Database Monitoring

Monitor MySQL:
```sql
SHOW PROCESSLIST;
SHOW STATUS;
```

---

## 🔄 Backup Strategy

### 1. Database Backups

```bash
# Daily backup
mysqldump -u root -p sentiment_analysis > backup_$(date +%Y%m%d).sql

# Restore
mysql -u root -p sentiment_analysis < backup_20240120.sql
```

### 2. Model Backups

```bash
# Backup models
tar -czf models_backup_$(date +%Y%m%d).tar.gz models_store/
```

### 3. Automated Backups

Create backup script (`backup.sh`):

```bash
#!/bin/bash
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d)

# Database backup
mysqldump -u root -p$DB_PASSWORD sentiment_analysis > $BACKUP_DIR/db_$DATE.sql

# Models backup
tar -czf $BACKUP_DIR/models_$DATE.tar.gz models_store/

# Keep only last 7 days
find $BACKUP_DIR -name "*.sql" -mtime +7 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
```

Add to crontab:
```bash
0 2 * * * /path/to/backup.sh
```

---

## 🔧 Maintenance

### 1. Log Rotation

Configure logrotate (`/etc/logrotate.d/sentiment-analysis`):

```
/path/to/Setiment_analysis/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0640 www-data www-data
}
```

### 2. Model Cleanup

Remove old model versions:
```python
from utils.model_lifecycle import ModelLifecycleManager

manager = ModelLifecycleManager()
# Keep only last 10 versions
```

### 3. Database Maintenance

```sql
-- Optimize tables
OPTIMIZE TABLE twitter_data;

-- Check table status
CHECK TABLE twitter_data;
```

---

## 🚨 Troubleshooting

### Common Issues

1. **Database Connection Error**:
   - Check MySQL is running
   - Verify credentials in config.py
   - Check firewall rules

2. **Model Loading Error**:
   - Verify model files exist
   - Check file permissions
   - Ensure pickle compatibility

3. **Port Already in Use**:
   ```bash
   # Find process using port
   lsof -i :5000
   # Kill process
   kill -9 <PID>
   ```

4. **Memory Issues**:
   - Reduce number of workers
   - Increase server RAM
   - Optimize model loading

### Debug Mode

Enable debug logging:
```python
# In config.py
LOG_LEVEL = "DEBUG"
```

---

## 📈 Performance Optimization

1. **Caching**: Use Redis for model caching
2. **Connection Pooling**: Database connection pooling
3. **Async Processing**: Use Celery for background tasks
4. **CDN**: Serve static files via CDN
5. **Load Balancing**: Multiple app instances

---

## ✅ Deployment Checklist

- [ ] Database configured and accessible
- [ ] Environment variables set
- [ ] Initial model trained
- [ ] SSL certificate installed (production)
- [ ] Firewall configured
- [ ] Monitoring set up
- [ ] Backups configured
- [ ] Log rotation configured
- [ ] Health checks working
- [ ] Documentation updated

---

## 📞 Support

For deployment issues:
- Check logs: `logs/system.log`
- Review GitHub issues
- Contact maintainers
