#!/bin/bash
set -e

# Function to wait for database
wait_for_db() {
    echo "Waiting for database to be ready..."
    while ! nc -z db 5432; do
        sleep 1
    done
    echo "Database is ready!"
}

# Function to run database migrations
run_migrations() {
    echo "Running database migrations..."
    python -c "
from app import create_app, db
from app.models import User
import os

app = create_app()
with app.app_context():
    try:
        # Create all tables
        db.create_all()
        print('Database tables created successfully')
        
        # Check if admin user exists, create if not
        admin_email = os.getenv('ADMIN_EMAIL', 'admin@hearteye.com')
        admin_password = os.getenv('ADMIN_PASSWORD', 'admin123')
        
        if not User.query.filter_by(email=admin_email).first():
            from werkzeug.security import generate_password_hash
            admin_user = User(
                username='admin',
                email=admin_email,
                password_hash=generate_password_hash(admin_password)
            )
            db.session.add(admin_user)
            db.session.commit()
            print(f'Admin user created: {admin_email}')
        else:
            print('Admin user already exists')
            
    except Exception as e:
        print(f'Database initialization error: {e}')
        exit(1)
"
}

# Main execution
echo "Starting HeartEye ECG AI Backend..."

# Wait for database
wait_for_db

# Run migrations
run_migrations

# Create necessary directories
mkdir -p /app/uploads /app/plots /app/logs

# Set proper permissions
chown -R appuser:appuser /app/uploads /app/plots /app/logs

# Switch to non-root user and execute the main command
echo "Starting application with user: appuser"
exec gosu appuser "$@" 