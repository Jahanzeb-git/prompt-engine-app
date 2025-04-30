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
            template = Template(
                llm_name="Test LLM",
                task_name="Test Task",
                template_body="<PromptTemplate><UserInput>{{user_prompt}}</UserInput></PromptTemplate>",
                max_tokens=10,  # Very low token limit for testing
                forbidden_patterns=""
            )