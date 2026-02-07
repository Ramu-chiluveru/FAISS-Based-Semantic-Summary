from flask import Flask, jsonify, request
from flask_cors import CORS
from src.config.config import Config
from src.utils import setup_logger

# Initialize Logger
logger = setup_logger("app")

def create_app():
    """
    Factory function to create and configure the Flask application.
    """
    app = Flask(__name__)
    
    # Configure CORS
    CORS(app, resources={r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }})
    
    # Load configuration
    app.config.from_object(Config)
    
    # Validate critical config on startup
    try:
        Config.validate()
        logger.info("Configuration validated successfully.")
    except ValueError as e:
        logger.error(f"Configuration Error: {e}")
    
    @app.route('/status', methods=['GET'])
    def health_check():
        """
        Simple health check endpoint.
        """
        return jsonify({
            "status": "healthy",
            "llm_provider": Config.LLM_PROVIDER,
            "embedding_provider": Config.EMBEDDING_PROVIDER
        }), 200

    @app.route('/summarize', methods=['POST'])
    def summarize():
        """
        Endpoint to summarize uploaded text or file.
        """
        try:
            text = ""
            if 'file' in request.files:
                file = request.files['file']
                if file.filename == '':
                    return jsonify({"error": "No selected file"}), 400
                # Basic text reading for MVP
                try:
                    text = file.read().decode('utf-8')
                except UnicodeDecodeError:
                    return jsonify({"error": "File encoding not supported. Please upload UTF-8 text."}), 400
            elif 'text' in request.form:
                text = request.form['text']
            elif request.is_json and 'text' in request.json:
                text = request.json['text']
            else:
                return jsonify({"error": "No text or file provided"}), 400

            if not text:
                return jsonify({"error": "Empty text provided"}), 400

            # Run Pipeline
            from src.summarization.summarization import SummarizationPipeline
            pipeline = SummarizationPipeline()
            result = pipeline.run(text)
            
            return jsonify(result), 200

        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return jsonify({"error": str(e)}), 500

    logger.info("HERCULES Backend initialized.")
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=Config.PORT, debug=Config.DEBUG)
