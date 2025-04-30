"""
Simulation script to test the Prompt Engine backend functionality.
This script acts as a client to test various endpoints and workflows.
"""

import requests
import json
import time
import sys
from unittest.mock import patch, MagicMock

# Configuration
BASE_URL = "http://localhost:12000"
ENDPOINTS = {
    "health": "/health",
    "optimize": "/api/optimize",
    "optimize_stream": "/api/optimize/stream"
}

# Test data
TEST_PROMPTS = [
    {
        "selected_LLM": "Claude 3.7 Sonnet",
        "selected_task": "back-end generation",
        "user_prompt": "Create a Flask API for a todo list application"
    },
    {
        "selected_LLM": "GPT-O4",
        "selected_task": "front-end debugging",
        "user_prompt": "My React component isn't rendering properly"
    },
    {
        "selected_LLM": "Invalid LLM",
        "selected_task": "back-end generation",
        "user_prompt": "This should fail with a template not found error"
    }
]

def test_health_endpoint():
    """Test the health check endpoint."""
    print("\n=== Testing Health Endpoint ===")
    try:
        response = requests.get(f"{BASE_URL}{ENDPOINTS['health']}")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def test_optimize_endpoint(prompt_data):
    """Test the optimize endpoint with the given prompt data."""
    print(f"\n=== Testing Optimize Endpoint with {prompt_data['selected_LLM']} ===")
    try:
        response = requests.post(
            f"{BASE_URL}{ENDPOINTS['optimize']}",
            json=prompt_data
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def mock_huggingface_api():
    """
    Mock the Huggingface API call in the app module.
    This allows us to test without making actual API calls.
    """
    # Create a mock response for the API
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": """Here's my analysis and enhancement:

<EnhancedPrompt>
# Flask Todo API Development Task

Create a RESTful Flask API for a todo list application with the following requirements:

## Core Functionality:
- Implement CRUD operations for todo items
- Each todo should have: id, title, description, due_date, priority, completed status
- Include proper error handling and status codes
- Implement input validation

## Technical Requirements:
- Use Flask-RESTful or Flask-Smorest for API structure
- Implement SQLAlchemy for database operations
- Include proper documentation with Swagger/OpenAPI
- Add authentication using JWT
- Implement rate limiting
- Include comprehensive logging

## Code Quality:
- Follow PEP 8 style guidelines
- Include docstrings for all functions and classes
- Write unit tests with pytest
- Implement proper exception handling
- Use environment variables for configuration

Please provide a well-structured, production-ready implementation with proper error handling, validation, and documentation.
</EnhancedPrompt>

I've enhanced your prompt to be more specific and comprehensive for Claude 3.7 Sonnet.
"""
                }
            }
        ]
    }
    
    # Create the patch
    return patch('requests.post', return_value=mock_response)

def run_simulation():
    """Run the full simulation."""
    print("Starting Prompt Engine Backend Simulation")
    print("=========================================")
    
    # Test health endpoint
    health_ok = test_health_endpoint()
    if not health_ok:
        print("Health endpoint test failed. Exiting simulation.")
        return False
    
    # Apply the mock to avoid actual API calls
    with mock_huggingface_api():
        # Test optimize endpoint with valid data
        optimize_ok = test_optimize_endpoint(TEST_PROMPTS[0])
        if not optimize_ok:
            print("Optimize endpoint test failed with valid data.")
            return False
        
        # Test optimize endpoint with error case
        error_test = test_optimize_endpoint(TEST_PROMPTS[2])
        if error_test:
            print("Error case test unexpectedly succeeded.")
            return False
    
    print("\n=========================================")
    print("Simulation completed successfully!")
    return True

if __name__ == "__main__":
    success = run_simulation()
    sys.exit(0 if success else 1)