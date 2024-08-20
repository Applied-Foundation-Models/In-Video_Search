from app import create_app

# Create Flask application using factory
app = create_app()

# Run the Flask application
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
