#!/usr/bin/env python3
"""
Locust Load Testing for Lung Cancer Classification API
"""

from locust import HttpUser, task, between
import os
import random

class LungCancerAPIUser(HttpUser):
    """User class for load testing the lung cancer classification API"""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Called when a user starts"""
        # Check if test images exist
        self.test_images = []
        test_image_dir = "data/test"
        
        if os.path.exists(test_image_dir):
            for root, dirs, files in os.walk(test_image_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.test_images.append(os.path.join(root, file))
        
        if not self.test_images:
            print("Warning: No test images found. Creating dummy test.")
            # Create a dummy test image if none exist
            self.create_dummy_image()
    
    def create_dummy_image(self):
        """Create a dummy test image for testing"""
        try:
            from PIL import Image, ImageDraw
            
            # Create a simple test image
            img = Image.new('RGB', (224, 224), color='white')
            draw = ImageDraw.Draw(img)
            draw.rectangle([50, 50, 174, 174], outline='black', width=2)
            
            # Save dummy image
            os.makedirs("test_images", exist_ok=True)
            dummy_path = "test_images/dummy_test.png"
            img.save(dummy_path)
            self.test_images = [dummy_path]
            
        except ImportError:
            print("PIL not available, using file upload without image")
            self.test_images = []
    
    @task(3)
    def health_check(self):
        """Health check endpoint - high frequency"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    @task(2)
    def get_status(self):
        """Get training status"""
        with self.client.get("/status", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status check failed: {response.status_code}")
    
    @task(2)
    def get_metrics(self):
        """Get model metrics"""
        with self.client.get("/metrics", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Metrics check failed: {response.status_code}")
    
    @task(1)
    def get_history(self):
        """Get prediction history"""
        with self.client.get("/history", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"History check failed: {response.status_code}")
    
    @task(4)
    def single_prediction(self):
        """Single image prediction - main functionality"""
        if not self.test_images:
            # If no test images, just test the endpoint structure
            with self.client.post("/predict", 
                                files={"file": ("test.txt", "dummy content", "text/plain")},
                                catch_response=True) as response:
                if response.status_code in [400, 500]:  # Expected for invalid file
                    response.success()
                else:
                    response.failure(f"Unexpected response: {response.status_code}")
            return
        
        # Use a random test image
        test_image_path = random.choice(self.test_images)
        
        try:
            with open(test_image_path, "rb") as f:
                files = {"file": (os.path.basename(test_image_path), f, "image/png")}
                
                with self.client.post("/predict", 
                                    files=files,
                                    catch_response=True) as response:
                    if response.status_code == 200:
                        # Validate response structure
                        try:
                            data = response.json()
                            if "prediction" in data and "confidence" in data:
                                response.success()
                            else:
                                response.failure("Invalid response structure")
                        except:
                            response.failure("Invalid JSON response")
                    else:
                        response.failure(f"Prediction failed: {response.status_code}")
                        
        except Exception as e:
            self.client.post("/predict", 
                           files={"file": ("error.txt", str(e), "text/plain")},
                           catch_response=True)
    
    @task(1)
    def batch_prediction(self):
        """Batch prediction - lower frequency due to higher load"""
        if not self.test_images:
            return
        
        # Use up to 3 random test images
        num_images = min(3, len(self.test_images))
        selected_images = random.sample(self.test_images, num_images)
        
        try:
            files = []
            for img_path in selected_images:
                with open(img_path, "rb") as f:
                    files.append(("files", (os.path.basename(img_path), f, "image/png")))
            
            with self.client.post("/predict-batch", 
                                files=files,
                                catch_response=True) as response:
                if response.status_code == 200:
                    try:
                        data = response.json()
                        if "predictions" in data:
                            response.success()
                        else:
                            response.failure("Invalid batch response structure")
                    except:
                        response.failure("Invalid JSON response")
                else:
                    response.failure(f"Batch prediction failed: {response.status_code}")
                    
        except Exception as e:
            self.client.post("/predict-batch", 
                           files={"files": ("error.txt", str(e), "text/plain")},
                           catch_response=True)
    
    @task(1)
    def retrain_model(self):
        """Model retraining - very low frequency"""
        with self.client.post("/retrain", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Retrain failed: {response.status_code}")

class HighLoadUser(LungCancerAPIUser):
    """High load user for stress testing"""
    
    wait_time = between(0.5, 1.5)  # Faster requests
    
    @task(5)
    def rapid_predictions(self):
        """Rapid fire predictions for stress testing"""
        self.single_prediction()

class LowLoadUser(LungCancerAPIUser):
    """Low load user for baseline testing"""
    
    wait_time = between(5, 10)  # Slower requests
    
    @task(1)
    def detailed_analysis(self):
        """Detailed analysis with multiple endpoints"""
        self.health_check()
        self.get_status()
        self.get_metrics()
        self.single_prediction()

# Custom events for monitoring
from locust import events

@events.request.add_listener
def my_request_handler(request_type, name, response_time, response_length, response,
                      context, exception, start_time, url, **kwargs):
    """Custom request handler for detailed monitoring"""
    if exception:
        print(f"Request failed: {name} - {exception}")
    else:
        print(f"Request successful: {name} - {response_time}ms")

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when the test starts"""
    print("🚀 Starting Lung Cancer Classification API Load Test")
    print(f"Target URL: {environment.host}")
    print(f"Number of users: {environment.runner.user_count if hasattr(environment.runner, 'user_count') else 'Unknown'}")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when the test stops"""
    print("🏁 Load test completed")
    print("📊 Check the Locust web interface for detailed results")

# Configuration for different test scenarios
class Config:
    """Configuration for different load test scenarios"""
    
    @staticmethod
    def get_scenarios():
        """Get different test scenarios"""
        return {
            "baseline": {
                "users": 10,
                "spawn_rate": 2,
                "run_time": "2m"
            },
            "normal_load": {
                "users": 50,
                "spawn_rate": 5,
                "run_time": "5m"
            },
            "high_load": {
                "users": 100,
                "spawn_rate": 10,
                "run_time": "10m"
            },
            "stress_test": {
                "users": 200,
                "spawn_rate": 20,
                "run_time": "15m"
            },
            "spike_test": {
                "users": 500,
                "spawn_rate": 50,
                "run_time": "5m"
            }
        }

if __name__ == "__main__":
    print("🔬 Lung Cancer Classification API Load Testing")
    print("=" * 50)
    print("Available scenarios:")
    for name, config in Config.get_scenarios().items():
        print(f"  {name}: {config['users']} users, {config['spawn_rate']} spawn rate, {config['run_time']} duration")
    print("\nTo run a specific scenario:")
    print("locust -f locustfile.py --host=http://localhost:8000")
    print("\nOr use the web interface:")
    print("locust -f locustfile.py --host=http://localhost:8000 --web-host=0.0.0.0 --web-port=8089") 