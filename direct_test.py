"""
Direct testing script for the Prompt Engine backend.
This script tests the core functions directly without making HTTP requests.
"""

import os
import sys
import json
from unittest.mock import patch, MagicMock

# Set environment variables for testing
os.environ['FLASK_DEBUG'] = 'True'
os.environ['MOCK_API'] = 'True'

# Import app and functions
from app import app, get_template, fill_template, perform_sanity_checks, extract_enhanced_prompt, PromptEngineError

def test_get_template():
    """Test the template lookup function."""
    print("\n=== Testing get_template function ===")
    with app.app_context():
        try:
            # Test with valid LLM and task
            template = get_template("Claude 3.7 Sonnet", "back-end generation")
            print(f"Found template: {template}")
            print(f"LLM: {template.llm_name}")
            print(f"Task: {template.task_name}")
            print(f"Max tokens: {template.max_tokens}")
            print("Template lookup successful!")
            
            # Test with invalid LLM and task
            try:
                template = get_template("Invalid LLM", "Invalid Task")
                print("Error: Should have raised an exception for invalid template")
                return False
            except PromptEngineError as e:
                print(f"Correctly raised error for invalid template: {str(e)}")
            
            return True
        except Exception as e:
            print(f"Error: {str(e)}")
            return False

def test_fill_template():
    """Test the template filling function."""
    print("\n=== Testing fill_template function ===")
    try:
        template = "<PromptTemplate><UserInput>{{user_prompt}}</UserInput></PromptTemplate>"
        user_prompt = "Create a Flask API"
        
        filled_template = fill_template(template, user_prompt)
        expected = "<PromptTemplate><UserInput>Create a Flask API</UserInput></PromptTemplate>"
        
        print(f"Original template: {template}")
        print(f"User prompt: {user_prompt}")
        print(f"Filled template: {filled_template}")
        
        if filled_template == expected:
            print("Template filling successful!")
            return True
        else:
            print(f"Error: Template filling failed. Expected: {expected}, Got: {filled_template}")
            return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def test_extract_enhanced_prompt():
    """Test the enhanced prompt extraction function."""
    print("\n=== Testing extract_enhanced_prompt function ===")
    try:
        # Test with valid response
        valid_response = "Here's some text <EnhancedPrompt>This is the enhanced prompt</EnhancedPrompt> and more text"
        extracted = extract_enhanced_prompt(valid_response)
        print(f"Original response: {valid_response}")
        print(f"Extracted prompt: {extracted}")
        
        if extracted == "This is the enhanced prompt":
            print("Prompt extraction successful!")
        else:
            print(f"Error: Prompt extraction failed. Expected: 'This is the enhanced prompt', Got: {extracted}")
            return False
        
        # Test with missing tags
        try:
            extract_enhanced_prompt("Response without tags")
            print("Error: Should have raised an exception for missing tags")
            return False
        except PromptEngineError as e:
            print(f"Correctly raised error for missing tags: {str(e)}")
        
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def test_optimize_prompt_workflow():
    """Test the complete optimize prompt workflow."""
    print("\n=== Testing optimize_prompt workflow ===")
    with app.app_context():
        try:
            # Mock the requests.post function
            with patch('requests.post') as mock_post:
                # Configure the mock
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    'choices': [
                        {
                            'message': {
                                'content': '<EnhancedPrompt>Enhanced version of the prompt</EnhancedPrompt>'
                            }
                        }
                    ]
                }
                mock_post.return_value = mock_response
                
                # Get a template
                template = get_template("Claude 3.7 Sonnet", "back-end generation")
                print(f"Found template: {template.llm_name} - {template.task_name}")
                
                # Fill the template
                user_prompt = "Create a Flask API"
                filled_template = fill_template(template.template_body, user_prompt)
                print(f"Filled template (truncated): {filled_template[:50]}...")
                
                # Perform sanity checks
                perform_sanity_checks(template, filled_template, user_prompt)
                print("Sanity checks passed")
                
                # Simulate API call (already mocked)
                print("API call would be made here (mocked)")
                
                # Extract enhanced prompt
                response_text = '<EnhancedPrompt>Enhanced version of the prompt</EnhancedPrompt>'
                enhanced_prompt = extract_enhanced_prompt(response_text)
                print(f"Enhanced prompt: {enhanced_prompt}")
                
                print("Complete workflow successful!")
                return True
        except Exception as e:
            print(f"Error: {str(e)}")
            return False

def run_tests():
    """Run all tests."""
    print("Starting Direct Tests for Prompt Engine Backend")
    print("==============================================")
    
    # Initialize the database
    with app.app_context():
        from models import db
        db.create_all()
        
        # Create a test template if needed
        from models import Template
        if Template.query.count() == 0:
            print("Creating test templates...")
            templates = [
                Template(
                    llm_name="Claude 3.7 Sonnet",
                    task_name="back-end generation",
                    template_body="<PromptTemplate><UserInput>{{user_prompt}}</UserInput></PromptTemplate>",
                    max_tokens=200000,
                    forbidden_patterns="eval\\(,\\s*exec\\("
                ),
                Template(
                    llm_name="GPT-O4",
                    task_name="front-end debugging",
                    template_body="<PromptTemplate><UserInput>{{user_prompt}}</UserInput></PromptTemplate>",
                    max_tokens=128000,
                    forbidden_patterns="document\\.cookie"
                )
            ]
            
            for template in templates:
                db.session.add(template)
            
            db.session.commit()
    
    # Run tests
    tests = [
        test_get_template,
        test_fill_template,
        test_extract_enhanced_prompt,
        test_optimize_prompt_workflow
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    # Print summary
    print("\n==============================================")
    print("Test Summary:")
    for i, test in enumerate(tests):
        status = "PASSED" if results[i] else "FAILED"
        print(f"  {test.__name__}: {status}")
    
    all_passed = all(results)
    print(f"\nOverall: {'ALL TESTS PASSED!' if all_passed else 'SOME TESTS FAILED!'}")
    
    return all_passed

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)