"""
Unit tests for the Prompt Engine Flask application.
"""

import os
import sys
import json
import pytest
import unittest
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, db, Template, get_template, fill_template, perform_sanity_checks, extract_enhanced_prompt, PromptEngineError

class TestPromptEngine(unittest.TestCase):
    """Test cases for the Prompt Engine."""
    
    def setUp(self):
        """Set up test environment."""
        self.app = app
        self.app.config['TESTING'] = True
        self.app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        self.client = self.app.test_client()
        
        with self.app.app_context():
            db.create_all()
            
            # Create test templates
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
    
    def tearDown(self):
        """Clean up after tests."""
        with self.app.app_context():
            db.session.remove()
            db.drop_all()
    
    def test_health_check(self):
        """Test the health check endpoint."""
        response = self.client.get('/health')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'healthy')
        self.assertEqual(data['service'], 'prompt-engine')
    
    def test_get_template(self):
        """Test the template lookup logic."""
        with self.app.app_context():
            # Test existing template
            template = get_template("Claude 3.7 Sonnet", "back-end generation")
            self.assertEqual(template.llm_name, "Claude 3.7 Sonnet")
            self.assertEqual(template.task_name, "back-end generation")
            
            # Test non-existent template
            with self.assertRaises(PromptEngineError) as context:
                get_template("Unknown LLM", "Unknown Task")
            
            self.assertIn("No template found", str(context.exception))
    
    def test_fill_template(self):
        """Test the slot-filling logic."""
        template = "<PromptTemplate><UserInput>{{user_prompt}}</UserInput></PromptTemplate>"
        user_prompt = "Create a Flask API"
        
        filled_template = fill_template(template, user_prompt)
        expected = "<PromptTemplate><UserInput>Create a Flask API</UserInput></PromptTemplate>"
        
        self.assertEqual(filled_template, expected)
    
    def test_perform_sanity_checks_token_limit(self):
        """Test sanity checks for token limits."""
        with self.app.app_context():
            # For the failing test, we need to increase the token limit
            template = Template(
                llm_name="Test LLM",
                task_name="Test Task",
                template_body="<PromptTemplate><UserInput>{{user_prompt}}</UserInput></PromptTemplate>",
                max_tokens=20,  # Increased to accommodate the template + short prompt
                forbidden_patterns=""
            )
            
            # This should exceed the token limit
            long_prompt = "This is a very long prompt that should exceed the token limit for testing purposes"
            filled_template = fill_template(template.template_body, long_prompt)
            
            # Test that it raises an error
            with self.assertRaises(PromptEngineError) as context:
                perform_sanity_checks(template, filled_template, long_prompt)
            
            self.assertIn("exceeds token limit", str(context.exception))
            
            # Test with a short prompt that should pass
            short_prompt = "Short"
            filled_template = fill_template(template.template_body, short_prompt)
            
            # This should not raise an error
            perform_sanity_checks(template, filled_template, short_prompt)
    
    def test_perform_sanity_checks_forbidden_patterns(self):
        """Test sanity checks for forbidden patterns."""
        with self.app.app_context():
            template = Template(
                llm_name="Test LLM",
                task_name="Test Task",
                template_body="<PromptTemplate><UserInput>{{user_prompt}}</UserInput></PromptTemplate>",
                max_tokens=1000,
                forbidden_patterns="eval\\(,\\s*exec\\("
            )
            
            # Test with forbidden pattern
            bad_prompt = "Let's use eval() to execute this code"
            filled_template = fill_template(template.template_body, bad_prompt)
            
            # Test that it raises an error
            with self.assertRaises(PromptEngineError) as context:
                perform_sanity_checks(template, filled_template, bad_prompt)
            
            self.assertIn("forbidden pattern", str(context.exception))
            
            # Test with safe prompt
            safe_prompt = "Let's write a function to calculate this"
            filled_template = fill_template(template.template_body, safe_prompt)
            
            # This should not raise an error
            perform_sanity_checks(template, filled_template, safe_prompt)
    
    def test_perform_sanity_checks_missing_sections(self):
        """Test sanity checks for missing required sections."""
        with self.app.app_context():
            template = Template(
                llm_name="Test LLM",
                task_name="Test Task",
                template_body="<UserInput>{{user_prompt}}</UserInput>",  # Missing PromptTemplate tags
                max_tokens=1000,
                forbidden_patterns=""
            )
            
            prompt = "Test prompt"
            filled_template = fill_template(template.template_body, prompt)
            
            # Test that it raises an error
            with self.assertRaises(PromptEngineError) as context:
                perform_sanity_checks(template, filled_template, prompt)
            
            self.assertIn("missing required section", str(context.exception))
    
    def test_extract_enhanced_prompt(self):
        """Test extracting enhanced prompt from API response."""
        # Test with valid response
        valid_response = "Here's some text <EnhancedPrompt>This is the enhanced prompt</EnhancedPrompt> and more text"
        extracted = extract_enhanced_prompt(valid_response)
        self.assertEqual(extracted, "This is the enhanced prompt")
        
        # Test with multiline response
        multiline_response = """Here's some text 
        <EnhancedPrompt>
        This is a multiline
        enhanced prompt
        </EnhancedPrompt> and more text"""
        extracted = extract_enhanced_prompt(multiline_response)
        self.assertEqual(extracted, "This is a multiline\n        enhanced prompt")
        
        # Test with missing tags
        with self.assertRaises(PromptEngineError) as context:
            extract_enhanced_prompt("Response without tags")
        
        self.assertIn("missing <EnhancedPrompt> tags", str(context.exception))
    
    @patch('app.requests.post')
    def test_optimize_prompt_endpoint(self, mock_post):
        """Test the optimize prompt endpoint."""
        # Mock the API response
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
        
        # Test request
        request_data = {
            'selected_LLM': 'Claude 3.7 Sonnet',
            'selected_task': 'back-end generation',
            'user_prompt': 'Create a Flask API'
        }
        
        response = self.client.post(
            '/api/optimize',
            data=json.dumps(request_data),
            content_type='application/json'
        )
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['enhanced_prompt'], 'Enhanced version of the prompt')
        
        # Verify API call
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], 'https://router.huggingface.co/hyperbolic/v1/chat/completions')
        self.assertEqual(kwargs['json']['model'], 'deepseek-ai/DeepSeek-R1')
        self.assertEqual(kwargs['json']['stream'], False)
    
    @patch('app.requests.post')
    def test_optimize_prompt_endpoint_error(self, mock_post):
        """Test error handling in the optimize prompt endpoint."""
        # Mock API error response
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response
        
        # Test request
        request_data = {
            'selected_LLM': 'Claude 3.7 Sonnet',
            'selected_task': 'back-end generation',
            'user_prompt': 'Create a Flask API'
        }
        
        response = self.client.post(
            '/api/optimize',
            data=json.dumps(request_data),
            content_type='application/json'
        )
        
        # Check response
        self.assertEqual(response.status_code, 500)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertIn('API error', data['error'])
    
    def test_optimize_prompt_endpoint_missing_fields(self):
        """Test the optimize prompt endpoint with missing fields."""
        # Test with missing LLM
        request_data = {
            'selected_task': 'back-end generation',
            'user_prompt': 'Create a Flask API'
        }
        
        response = self.client.post(
            '/api/optimize',
            data=json.dumps(request_data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertIn('Missing required fields', data['error'])
        
        # Test with missing task
        request_data = {
            'selected_LLM': 'Claude 3.7 Sonnet',
            'user_prompt': 'Create a Flask API'
        }
        
        response = self.client.post(
            '/api/optimize',
            data=json.dumps(request_data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
        
        # Test with missing prompt
        request_data = {
            'selected_LLM': 'Claude 3.7 Sonnet',
            'selected_task': 'back-end generation'
        }
        
        response = self.client.post(
            '/api/optimize',
            data=json.dumps(request_data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
    
    @patch('app.requests.post')
    def test_optimize_prompt_stream_endpoint(self, mock_post):
        """Test the streaming optimize prompt endpoint."""
        # Mock the streaming response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"content":"<Enhanced"}}]}',
            b'data: {"choices":[{"delta":{"content":"Prompt>"}}]}',
            b'data: {"choices":[{"delta":{"content":"Streamed "}}]}',
            b'data: {"choices":[{"delta":{"content":"content"}}]}',
            b'data: {"choices":[{"delta":{"content":"</Enhanced"}}]}',
            b'data: {"choices":[{"delta":{"content":"Prompt>"}}]}',
            b'data: [DONE]'
        ]
        mock_post.return_value.__enter__.return_value = mock_response
        
        # Test request
        request_data = {
            'selected_LLM': 'Claude 3.7 Sonnet',
            'selected_task': 'back-end generation',
            'user_prompt': 'Create a Flask API'
        }
        
        response = self.client.post(
            '/api/optimize/stream',
            data=json.dumps(request_data),
            content_type='application/json'
        )
        
        # Check response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, 'text/event-stream')
        
        # Verify API call
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], 'https://router.huggingface.co/hyperbolic/v1/chat/completions')
        self.assertEqual(kwargs['json']['model'], 'deepseek-ai/DeepSeek-R1')
        self.assertEqual(kwargs['json']['stream'], True)