"""
SQLAlchemy models for the Prompt Engine service.
Defines the Template model for storing prompt templates.
"""

from flask_sqlalchemy import SQLAlchemy

# Initialize SQLAlchemy
db = SQLAlchemy()

class Template(db.Model):
    """
    Template model for storing prompt templates.
    
    Attributes:
        id (int): Primary key
        llm_name (str): Name of the LLM (e.g., "Claude 3.7 Sonnet")
        task_name (str): Name of the task (e.g., "back-end generation")
        template_body (str): XML-wrapped system + instructions + slots
        max_tokens (int): Model-specific token limit
        forbidden_patterns (str): Comma-separated regex patterns of disallowed tokens
    """
    __tablename__ = 'templates'
    
    id = db.Column(db.Integer, primary_key=True)
    llm_name = db.Column(db.Text, nullable=False)
    task_name = db.Column(db.Text, nullable=False)
    template_body = db.Column(db.Text, nullable=False)
    max_tokens = db.Column(db.Integer, nullable=False)
    forbidden_patterns = db.Column(db.Text, nullable=True)
    
    # Define a unique constraint for llm_name and task_name
    __table_args__ = (
        db.UniqueConstraint('llm_name', 'task_name', name='uix_llm_task'),
    )
    
    def __repr__(self):
        """String representation of the Template."""
        return f"<Template {self.id}: {self.llm_name} - {self.task_name}>"
    
    def to_dict(self):
        """Convert the Template to a dictionary."""
        return {
            'id': self.id,
            'llm_name': self.llm_name,
            'task_name': self.task_name,
            'template_body': self.template_body,
            'max_tokens': self.max_tokens,
            'forbidden_patterns': self.forbidden_patterns
        }

def init_db():
    """Initialize the database and create tables if they don't exist."""
    db.create_all()