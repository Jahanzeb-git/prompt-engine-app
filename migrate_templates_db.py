"""
Migration script for creating and populating the templates database.
Creates the templates table and inserts sample templates for each LLM and task.
"""

import os
import sqlite3
import sys
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database file
DB_FILE = 'templates.db'

# LLM options
LLM_OPTIONS = [
    "Claude 3.7 Sonnet",
    "Claude 2.5 Sonnet",
    "GPT-O1",
    "GPT-O4",
    "Gemini 2.5 Pro",
    "Grok 3",
    "Deepseek R1"
]

# Task options
TASK_OPTIONS = [
    "front-end generation",
    "front-end debugging",
    "back-end generation",
    "back-end debugging"
]

# Default template format
DEFAULT_TEMPLATE = """<PromptTemplate>
  <System>
    You are a top-tier AI prompt engineer specializing in optimizing prompts for {llm} on {task} tasks.
  </System>
  <Context>
    The user is working with {llm} and needs help with {task}.
  </Context>
  <UserInput>{{user_prompt}}</UserInput>
  <Instructions>
    • Analyze the user's prompt and enhance it to work optimally with {llm}.
    • Add relevant {task}-specific instructions.
    • Include clear output formatting requirements.
    • Return only the enhanced prompt wrapped in <EnhancedPrompt> tags.
    • Do not include any other commentary.
  </Instructions>
</PromptTemplate>"""

# Token limits for each LLM (approximate)
TOKEN_LIMITS = {
    "Claude 3.7 Sonnet": 200000,
    "Claude 2.5 Sonnet": 100000,
    "GPT-O1": 128000,
    "GPT-O4": 128000,
    "Gemini 2.5 Pro": 32000,
    "Grok 3": 25000,
    "Deepseek R1": 32000
}

# Default forbidden patterns for each task
FORBIDDEN_PATTERNS = {
    "front-end generation": r"eval\(,\s*Function\(,\s*setTimeout\(",
    "front-end debugging": r"document\.cookie,\s*localStorage,\s*sessionStorage",
    "back-end generation": r"exec\(,\s*eval\(,\s*shell\(,\s*os\.system\(",
    "back-end debugging": r"__import__\(,\s*subprocess,\s*exec\("
}

def create_database():
    """Create the database and templates table."""
    try:
        # Check if database file exists
        if os.path.exists(DB_FILE):
            logger.warning(f"Database file '{DB_FILE}' already exists. Backing up...")
            os.rename(DB_FILE, f"{DB_FILE}.bak")
            logger.info(f"Backed up to '{DB_FILE}.bak'")
            
        # Connect to the database
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Create templates table
        cursor.execute('''
        CREATE TABLE templates (
            id INTEGER PRIMARY KEY,
            llm_name TEXT NOT NULL,
            task_name TEXT NOT NULL,
            template_body TEXT NOT NULL,
            max_tokens INTEGER NOT NULL,
            forbidden_patterns TEXT,
            UNIQUE(llm_name, task_name)
        )
        ''')
        
        # Commit changes
        conn.commit()
        logger.info("Database and tables created successfully.")
        
        return conn, cursor
    except Exception as e:
        logger.error(f"Error creating database: {str(e)}")
        sys.exit(1)

def populate_templates(conn, cursor):
    """Populate the templates table with sample templates."""
    try:
        # Insert a template for each LLM and task combination
        for llm in LLM_OPTIONS:
            for task in TASK_OPTIONS:
                # Create custom template for this combination
                template = DEFAULT_TEMPLATE.format(llm=llm, task=task)
                
                # Set token limit for this LLM
                max_tokens = TOKEN_LIMITS.get(llm, 32000)
                
                # Set forbidden patterns for this task
                forbidden_patterns = FORBIDDEN_PATTERNS.get(task, "")
                
                # Insert template into database
                cursor.execute('''
                INSERT INTO templates (llm_name, task_name, template_body, max_tokens, forbidden_patterns)
                VALUES (?, ?, ?, ?, ?)
                ''', (llm, task, template, max_tokens, forbidden_patterns))
        
        # Commit changes
        conn.commit()
        logger.info(f"Inserted {len(LLM_OPTIONS) * len(TASK_OPTIONS)} templates.")
        
        # Verify insertion
        cursor.execute('SELECT COUNT(*) FROM templates')
        count = cursor.fetchone()[0]
        logger.info(f"Template count: {count}")
        
    except Exception as e:
        logger.error(f"Error populating templates: {str(e)}")
        conn.rollback()
        sys.exit(1)

def insert_custom_templates(conn, cursor):
    """Insert custom-designed templates for specific combinations."""
    try:
        # Example of a custom template for Claude 3.7 + back-end generation
        claude_backend_template = """<PromptTemplate>
  <System>
    You are Claude 3.7 Sonnet, an exceptionally capable AI assistant with expertise in backend development.
    You are helping a developer write high-quality, production-ready backend code.
  </System>
  <Context>
    The user is working on a backend development task and needs assistance generating code that is:
    - Well-structured and modular
    - Follows best practices for error handling
    - Includes appropriate logging
    - Has comprehensive docstrings
    - Follows PEP 8 style guidelines (for Python)
    - Has proper security considerations
  </Context>
  <UserInput>{{user_prompt}}</UserInput>
  <Instructions>
    • First, analyze the requirements to ensure you understand the task fully
    • Consider the architecture and design patterns that would be most appropriate
    • Write production-quality code with proper error handling, validation, and logging
    • Include docstrings and comments to explain complex logic
    • For database operations, implement proper connection management and sanitization
    • Return only the enhanced prompt wrapped in <EnhancedPrompt> tags
    • Do not include any other commentary
  </Instructions>
</PromptTemplate>"""

        # Example of a custom template for GPT-O4 + front-end debugging
        gpt_frontend_debug_template = """<PromptTemplate>
  <System>
    You are GPT-O4, an expert AI assistant specialized in frontend debugging and troubleshooting.
    You excel at identifying and fixing complex issues in frontend code.
  </System>
  <Context>
    The user is working on debugging a frontend issue and needs assistance in:
    - Identifying potential causes of bugs
    - Suggesting effective debugging strategies
    - Providing solutions with proper explanations
    - Optimizing frontend performance
    - Ensuring cross-browser compatibility
  </Context>
  <UserInput>{{user_prompt}}</UserInput>
  <Instructions>
    • Analyze the user's description of the issue and any code provided
    • Ask targeted questions to narrow down the problem if information is incomplete
    • Suggest specific debugging tools and techniques (browser dev tools, console logs, etc.)
    • Provide step-by-step debugging instructions
    • When suggesting fixes, explain why they work
    • Include code examples with comments explaining the changes
    • Return only the enhanced prompt wrapped in <EnhancedPrompt> tags
    • Do not include any other commentary
  </Instructions>
</PromptTemplate>"""

        # Update these specific templates
        cursor.execute('''
        UPDATE templates 
        SET template_body = ? 
        WHERE llm_name = ? AND task_name = ?
        ''', (claude_backend_template, "Claude 3.7 Sonnet", "back-end generation"))
        
        cursor.execute('''
        UPDATE templates 
        SET template_body = ? 
        WHERE llm_name = ? AND task_name = ?
        ''', (gpt_frontend_debug_template, "GPT-O4", "front-end debugging"))
        
        # Commit changes
        conn.commit()
        logger.info("Custom templates inserted successfully.")
        
    except Exception as e:
        logger.error(f"Error inserting custom templates: {str(e)}")
        conn.rollback()

def main():
    """Main function to run the migration."""
    logger.info("Starting templates database migration...")
    
    # Create database and tables
    conn, cursor = create_database()
    
    # Populate with default templates
    populate_templates(conn, cursor)
    
    # Insert custom templates
    insert_custom_templates(conn, cursor)
    
    # Close connection
    cursor.close()
    conn.close()
    
    logger.info("Migration completed successfully.")

if __name__ == "__main__":
    main()