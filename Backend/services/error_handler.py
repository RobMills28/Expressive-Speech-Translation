import hashlib
import time
import logging
from datetime import datetime
from typing import Tuple, Dict, Any

# Set up logging
logger = logging.getLogger(__name__)

class ErrorHandler:
    """Handles error processing and formatting for the application."""

    @staticmethod
    def handle_error(e: Exception) -> Tuple[Dict[str, Any], int]:
        """
        Process an exception and return a formatted error response
        
        Args:
            e: The exception to handle
            
        Returns:
            Tuple[Dict[str, Any], int]: (error_response, http_status_code)
        """
        # Generate a unique error ID
        error_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        
        # Create detailed error info for logging
        error_info = {
            'error_id': error_id,
            'type': type(e).__name__,
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }
        
        # Log the error with full stack trace
        logger.error(
            f"Error {error_id}: {str(e)}", 
            exc_info=True,
            extra={'error_info': error_info}
        )
        
        # Return user-friendly error message
        user_message = (
            f"An error occurred (ID: {error_id}). "
            "Please try again or contact support if the problem persists."
        )
        return {'error': user_message}, 500

    @staticmethod
    def format_validation_error(message: str) -> Tuple[Dict[str, str], int]:
        """
        Format a validation error response
        
        Args:
            message: The validation error message
            
        Returns:
            Tuple[Dict[str, str], int]: (error_response, http_status_code)
        """
        return {'error': message}, 400

    @staticmethod
    def format_resource_error(message: str) -> Tuple[Dict[str, str], int]:
        """
        Format a resource-related error response
        
        Args:
            message: The resource error message
            
        Returns:
            Tuple[Dict[str, str], int]: (error_response, http_status_code)
        """
        return {'error': message}, 503