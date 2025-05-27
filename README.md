SSH (SecureSexHealth) - Anonymous Sexual Health Risk Assessment
🏥 Privacy-preserving health risk analysis service using Homomorphic Encryption
An anonymous health evaluation service that uses Homomorphic Encryption (HE) to protect personal information while providing accurate sexual health risk assessments.
🔧 System Requirements
Hardware Requirements:

Disk Space: 1.73 GB minimum
Memory: Minimum 32 GB, preferably 64 GB

Software Requirements:

Docker Desktop
Windows 10/11 (with WSL2) / macOS / Linux

🚀 Installation & Setup
Step 1: Install Docker
1.1 Installation
Visit Docker Desktop Installation Guide and install the appropriate version of Docker for your operating system.
1.2 Settings for Windows
On macOS and Linux, you generally do not need to adjust memory settings separately. However, on Windows, you must manually configure the memory settings.
Create (or edit) the .wslconfig file in your Windows home directory (e.g., C:\Users\your-username) and add the following content:
ini[wsl2]
memory=32GB
After saving the file, run the following commands in PowerShell or Command Prompt:
bashwsl --shutdown
Then restart Docker Desktop to apply the new memory settings.
Step 2: Install HeaaN Stat SDK
2.1 Pull Docker Image
In your terminal, run:
bashdocker pull cryptolabinc/heaan-stat:1.0.0-cpu
2.2 Run Docker Container
bashdocker run -p 8888:8888 --rm -it cryptolabinc/heaan-stat:1.0.0-cpu
Press Enter to continue when prompted.
Enter "yes" when asked to proceed.
Step 3: Setup Inside Docker Container
Once you're inside the Docker container terminal, follow these steps:
3.1 Install Git and Required Tools
bash# Update package manager
apt update

# Install Git
apt install -y git

# Install additional tools (if needed)
apt install -y curl wget
3.2 Clone Repository
bash# Clone the project repository
git clone https://github.com/gom31/IS_pi_heaan.git

# Navigate to project directory
cd IS_pi_heaan
3.3 Install Python Dependencies
bash# Install Python requirements
pip install -r health_analysis/requirements.txt

# Install Django (if not in requirements)
pip install django
3.4 Setup Django Application
bash# Navigate to Django project directory
cd ssh_service

# Run database migrations
python3 manage.py makemigrations
python3 manage.py migrate

# Collect static files (if needed)
python3 manage.py collectstatic --noinput
3.5 Start Django Development Server
bash# Run the server (accessible from host machine)
python3 manage.py runserver 0.0.0.0:8000
Step 4: Access the Application
Open your web browser and navigate to:
http://localhost:8000
📊 API Endpoints
Health Analysis API

URL: POST /health-analysis/
Description: Analyze health risk using homomorphic encryption
Content-Type: application/json

API Status Check

URL: GET /health-analysis/
Description: Check API health status

Test API

URL: POST /health-test/
Description: Test API with predefined data

Form Structure

URL: GET /health-form/
Description: Get survey form structure

🧪 Testing the API
Test with cURL
bashcurl -X POST http://localhost:8000/health-analysis/ \
  -H "Content-Type: application/json" \
  -d '{
    "sex": 1,
    "age": 3,
    "edu_lvl": 2,
    "had_sex": 1,
    "n_s_part": 2,
    "con_use": 1,
    "r_use_con": 0,
    "r_sea": 1,
    "r_have_1sp": 0,
    "r_nhave_sex": 0,
    "hiv_mosq": 0,
    "h_sti": 1,
    "h_o_sti": 1,
    "e_t_hiv": 0,
    "p_t_hiv": 1,
    "s_test": 0,
    "t_in_lab": 0,
    "h_aids": 1
  }'
Quick Test API
bashcurl -X POST http://localhost:8000/health-test/
🤖 Model Training (Optional)
If you need to retrain the machine learning models:
Prerequisites

Place your HIV_AIDS_DataSet.csv file in the data/ directory
Ensure all dependencies are installed

Training Commands
bash# Navigate to ML training directory
cd health_analysis/ml_training

# Run model training
python3 train_model.py

# Or use Django management command
cd ../../ssh_service
python3 manage.py train_model --data-path ../data/HIV_AIDS_DataSet.csv
Model Stability Analysis
bash# Run 100 iterations for stability analysis
python3 manage.py analyze_model_stability \
  --data-path ../data/HIV_AIDS_DataSet.csv \
  --iterations 100 \
  --models all

# model statbility analyzer
python3 model_statbility_analyzer.py
 
# Security Test
bash# Run simple security demonstration
python3 data_block_security_test.py

# Performance Test 
Test with Homomorphic Encryption (Secure & Slow)
curl -X POST http://localhost:8000/health-analysis/ -H "Content-Type: application/json" -d '{"sex": 1, "age": 3, "edu_lvl": 2, "had_sex": 1, "n_s_part": 2, "con_use": 1, "r_use_con": 0, "r_sea": 1, "r_have_1sp": 0, "r_nhave_sex": 0, "hiv_mosq": 0, "h_sti": 1, "h_o_sti": 1, "e_t_hiv": 0, "p_t_hiv": 1, "s_test": 0, "t_in_lab": 0, "h_aids": 1, "use_he": true}'
Test without Homomorphic Encryption (Fast & Direct)
curl -X POST http://localhost:8000/health-analysis/ -H "Content-Type: application/json" -d '{"sex": 1, "age": 3, "edu_lvl": 2, "had_sex": 1, "n_s_part": 2, "con_use": 1, "r_use_con": 0, "r_sea": 1, "r_have_1sp": 0, "r_nhave_sex": 0, "hiv_mosq": 0, "h_sti": 1, "h_o_sti": 1, "e_t_hiv": 0, "p_t_hiv": 1, "s_test": 0, "t_in_lab": 0, "h_aids": 1, "use_he": false}'
Test with Default Settings (HE Enabled)
curl -X POST http://localhost:8000/health-analysis/ -H "Content-Type: application/json" -d '{"sex": 1, "age": 3, "edu_lvl": 2, "had_sex": 1, "n_s_part": 2, "con_use": 1, "r_use_con": 0, "r_sea": 1, "r_have_1sp": 0, "r_nhave_sex": 0, "hiv_mosq": 0, "h_sti": 1, "h_o_sti": 1, "e_t_hiv": 0, "p_t_hiv": 1, "s_test": 0, "t_in_lab": 0, "h_aids": 1}'
📊 Expected Response Format
With HE (use_he: true)
json{
  "success": true,
  "data": {
    "risk_probability": 0.65,
    "method": "Homomorphic Encryption",
    "timing_info": {
      "encryption_time_ms": 45.67,
      "computation_time_ms": 102.01,
      "decryption_time_ms": 23.45
    }
  },
  "performance_metrics": {
    "total_time_ms": 198.76,
    "analysis_time_ms": 174.70,
    "method": "HE"
  }
}
Without HE (use_he: false)
json{
  "success": true,
  "data": {
    "risk_probability": 0.65,
    "method": "Direct Computation",
    "timing_info": {
      "direct_computation_time_ms": 2.34
    }
  },
  "performance_metrics": {
    "total_time_ms": 25.43,
    "analysis_time_ms": 2.34,
    "method": "Direct"
  }
}
