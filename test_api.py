"""
Test the FastAPI endpoints
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("\n Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    return response.status_code == 200

def test_predict():
    """Test predict endpoint"""
    print("\n Testing predict endpoint...")
    
    test_texts = [
        "I am feeling so happy and excited today!",
        "I feel sad and lonely right now.",
        "I am absolutely furious about this situation!",
        "I love this beautiful weather.",
        "I'm scared of what might happen.",
        "Wow, I can't believe this surprise!"
    ]
    
    for text in test_texts:
        payload = {"text": text}
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n    Text: {text}")
            print(f"      Emotion: {result['predicted_emotion']} {result['emotion_emoji']}")
            print(f"      Confidence: {result['confidence']:.2%}")
            print(f"      Latency: {result['latency_ms']}ms")
        else:
            print(f"\n    Error: {response.status_code} - {response.text}")

def test_bulk_predict():
    """Test multiple predictions"""
    print("\n Testing bulk predictions...")
    
    texts = [
        "I am thrilled with this outcome!",
        "This is the worst day ever.",
        "I'm feeling indifferent about this.",
        "What a wonderful surprise!"
    ]
    
    for text in texts:
        payload = {"text": text}
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"   {text[:30]}...  {result['predicted_emotion']} ({result['confidence']:.2%})")

if __name__ == "__main__":
    print("=" * 60)
    print(" Testing Emotion Classification API")
    print("=" * 60)
    
    # Wait for server to start
    print("\n Waiting for server to start...")
    import time
    time.sleep(2)
    
    # Run tests
    if test_health():
        test_predict()
        test_bulk_predict()
    else:
        print("\n Server not responding. Make sure it's running.")
        print("   Run: python -m src.deploy")


