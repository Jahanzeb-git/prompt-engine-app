"""
Flask application for the Prompt Engine backend service.
This service optimizes user prompts for different LLMs based on templates.
"""

import re
import json
import requests
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import sqlite3
import os
from models import db, Template, init_db
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///templates.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# Huggingface API settings
HUGGINGFACE_API_URL = "https://router.huggingface.co/hyperbolic/v1/chat/completions"
HUGGINGFACE_TOKEN = os.environ.get("HF_TOKEN")
DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-R1"

# Ensure database exists
with app.app_context():
    init_db()

class PromptEngineError(Exception):
    """Custom exception for Prompt Engine errors."""
    def __init__(self, message, status_code=400):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

@app.errorhandler(PromptEngineError)
def handle_prompt_engine_error(error):
    """Error handler for PromptEngineError."""
    response = jsonify({"error": error.message})
    response.status_code = error.status_code
    return response

def extract_enhanced_prompt(response_text):
    """
    Extract the content from <EnhancedPrompt> tags in the response.
    
    Args:
        response_text: The text response from the API
        
    Returns:
        The content inside the <EnhancedPrompt> tags
        
    Raises:
        PromptEngineError: If the EnhancedPrompt tags are missing
    """
    pattern = r'<EnhancedPrompt>(.*?)</EnhancedPrompt>'
    match = re.search(pattern, response_text, re.DOTALL)
    
    if not match:
        raise PromptEngineError("API response missing <EnhancedPrompt> tags", 500)
    
    return match.group(1).strip()

def fill_template(template, user_prompt):
    """
    Fill the template with the user's prompt.
    
    Args:
        template: The template string
        user_prompt: The user's prompt text
        
    Returns:
        The filled template
    """
    # Simple slot filling (can be extended for more complex templates)
    return template.replace("{{user_prompt}}", user_prompt)

def perform_sanity_checks(template_record, filled_template, user_prompt):
    """
    Perform sanity checks on the filled template.
    
    Args:
        template_record: The Template object from the database
        filled_template: The template after slot filling
        user_prompt: The user's prompt text
        
    Raises:
        PromptEngineError: If any checks fail
    """
    # Check token length (using character count as a proxy)
    max_tokens = template_record.max_tokens
    # Rough approximation: 1 token ≈ 4 characters
    estimated_tokens = len(filled_template) / 4
    
    if estimated_tokens > max_tokens:
        raise PromptEngineError(
            f"Combined prompt exceeds token limit ({estimated_tokens:.0f} > {max_tokens})"
        )
    
    # Check for forbidden patterns
    if template_record.forbidden_patterns:
        for pattern in template_record.forbidden_patterns.split(','):
            if pattern.strip() and re.search(pattern.strip(), user_prompt):
                raise PromptEngineError(
                    f"User prompt contains forbidden pattern: {pattern}"
                )
    
    # Verify required XML sections exist
    required_sections = ["<PromptTemplate>", "</PromptTemplate>"]
    for section in required_sections:
        if section not in filled_template:
            raise PromptEngineError(f"Template missing required section: {section}")

def get_template(llm_name, task_name):
    """
    Get the template for the given LLM and task.
    
    Args:
        llm_name: The name of the LLM
        task_name: The name of the task
        
    Returns:
        The Template object from the database
        
    Raises:
        PromptEngineError: If the template is not found
    """
    template = Template.query.filter_by(
        llm_name=llm_name, 
        task_name=task_name
    ).first()
    
    if not template:
        raise PromptEngineError(
            f"No template found for LLM '{llm_name}' and task '{task_name}'", 
            404
        )
    
    return template

@app.route('/api/optimize', methods=['POST'])
def optimize_prompt():
    """
    Endpoint to optimize a user's prompt for a specific LLM and task.
    
    Request JSON:
        {
            "selected_LLM": "...",
            "selected_task": "...",
            "user_prompt": "..."
        }
        
    Response JSON:
        {
            "enhanced_prompt": "..."
        }
    """
    try:
        # Parse request data
        data = request.json
        if not data:
            raise PromptEngineError("No JSON data provided")
        
        selected_llm = data.get('selected_LLM')
        selected_task = data.get('selected_task')
        user_prompt = data.get('user_prompt')
        
        # Validate required fields
        if not all([selected_llm, selected_task, user_prompt]):
            raise PromptEngineError("Missing required fields")
        
        # Get the template
        template_record = get_template(selected_llm, selected_task)
        
        # Fill the template
        filled_template = fill_template(template_record.template_body, user_prompt)
        
        # Perform sanity checks
        perform_sanity_checks(template_record, filled_template, user_prompt)
        
        # Prepare request to Huggingface API
        system_instructions = (
            "<SystemInstructions>You are a top-tier AI prompt engineer. "
            "Your task is to enhance the given prompt according to the template "
            "instructions.</SystemInstructions>"
        )
        
        payload = {
            "model": DEEPSEEK_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": filled_template}
            ],
            "max_tokens": 1500
        }
        
        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Send request to Huggingface API
        response = requests.post(
            HUGGINGFACE_API_URL,
            json=payload,
            headers=headers
        )
        
        if response.status_code != 200:
            raise PromptEngineError(
                f"API error: {response.status_code} - {response.text}",
                500
            )
        
        # Extract response content
        response_data = response.json()
        response_text = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
        
        # Extract enhanced prompt
        enhanced_prompt = extract_enhanced_prompt(response_text)
        
        # Return the enhanced prompt
        return jsonify({"enhanced_prompt": enhanced_prompt})
    
    except PromptEngineError as e:
        # Use custom error handler
        raise e
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise PromptEngineError(f"Server error: {str(e)}", 500)

@app.route('/api/optimize/stream', methods=['POST'])
def optimize_prompt_stream():
    """
    Streaming endpoint to optimize a user's prompt.
    
    Request:
        Same as /api/optimize but returns a stream
        
    Response:
        Server-Sent Events (SSE) stream
    """
    try:
        # Parse request data
        data = request.json
        if not data:
            raise PromptEngineError("No JSON data provided")
        
        selected_llm = data.get('selected_LLM')
        selected_task = data.get('selected_task')
        user_prompt = data.get('user_prompt')
        
        # Validate required fields
        if not all([selected_llm, selected_task, user_prompt]):
            raise PromptEngineError("Missing required fields")
        
        # Get the template
        template_record = get_template(selected_llm, selected_task)
        
        # Fill the template
        filled_template = fill_template(template_record.template_body, user_prompt)
        
        # Perform sanity checks
        perform_sanity_checks(template_record, filled_template, user_prompt)
        
        # Create generator function for streaming
        def generate():
            # Prepare request to Huggingface API
            system_instructions = (
                "<SystemInstructions>You are a top-tier AI prompt engineer. "
                "Your task is to enhance the given prompt according to the template "
                "instructions.</SystemInstructions>"
            )
            
            payload = {
                "model": DEEPSEEK_MODEL,
                "stream": True,
                "messages": [
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": filled_template}
                ],
                "max_tokens": 1500
            }
            
            headers = {
                "Authorization": f"Bearer {HUGGINGFACE_TOKEN}",
                "Content-Type": "application/json"
            }
            
            # Send streaming request to Huggingface API
            with requests.post(
                HUGGINGFACE_API_URL,
                json=payload,
                headers=headers,
                stream=True
            ) as response:
                if response.status_code != 200:
                    error_msg = f"API error: {response.status_code}"
                    yield f"event: error\ndata: {json.dumps({'error': error_msg})}\n\n"
                    return
                
                # Process streaming response
                buffer = ""
                inside_enhanced_prompt = False
                enhanced_prompt_content = ""
                
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode('utf-8')
                        
                        # Skip non-data lines
                        if not line_text.startswith('data: '):
                            continue
                            
                        data_text = line_text[6:]  # Remove 'data: ' prefix
                        
                        # Check for stream end
                        if data_text == "[DONE]":
                            yield f"event: done\ndata: [STREAM_END]\n\n"
                            break
                        
                        try:
                            chunk_data = json.loads(data_text)
                            content = chunk_data.get('choices', [{}])[0].get('delta', {}).get('content', '')
                            
                            if content:
                                buffer += content
                                
                                # Check for opening tag
                                if '<EnhancedPrompt>' in buffer and not inside_enhanced_prompt:
                                    idx = buffer.find('<EnhancedPrompt>') + len('<EnhancedPrompt>')
                                    inside_enhanced_prompt = True
                                    enhanced_prompt_content = buffer[idx:]
                                    yield f"event: prompt-chunk\ndata: {json.dumps(enhanced_prompt_content)}\n\n"
                                    
                                # Check for content inside tags
                                elif inside_enhanced_prompt and '</EnhancedPrompt>' not in buffer:
                                    enhanced_prompt_content = content
                                    yield f"event: prompt-chunk\ndata: {json.dumps(enhanced_prompt_content)}\n\n"
                                    
                                # Check for closing tag
                                elif inside_enhanced_prompt and '</EnhancedPrompt>' in content:
                                    idx = content.find('</EnhancedPrompt>')
                                    if idx > 0:
                                        enhanced_prompt_content = content[:idx]
                                        yield f"event: prompt-chunk\ndata: {json.dumps(enhanced_prompt_content)}\n\n"
                                        
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse JSON: {data_text}")
                            continue
        
        # Return SSE response
        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'  # Disable buffering in Nginx
            }
        )
        
    except PromptEngineError as e:
        # For streaming, return error as SSE
        def error_stream():
            yield f"event: error\ndata: {json.dumps({'error': e.message})}\n\n"
        
        return Response(
            stream_with_context(error_stream()),
            mimetype='text/event-stream'
        )
        
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        
        def error_stream():
            yield f"event: error\ndata: {json.dumps({'error': f'Server error: {str(e)}'})}\n\n"
        
        return Response(
            stream_with_context(error_stream()),
            mimetype='text/event-stream'
        )

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "prompt-engine"})

if __name__ == '__main__':
    app.run(debug=os.environ.get('FLASK_DEBUG', 'False') == 'True', host='0.0.0.0', port=5000)