from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    email = db.Column(db.String(120), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    ecgs = db.relationship('ECG', backref='user', lazy=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class ECG(db.Model):
    __tablename__ = 'ecgs'
    
    id = db.Column(db.Integer, primary_key=True)
    file_id = db.Column(db.String(36), unique=True, nullable=False)  # UUID
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    patient_name = db.Column(db.String(100), nullable=True)
    age = db.Column(db.Integer, nullable=True)
    gender = db.Column(db.String(1), nullable=True)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    heart_rate = db.Column(db.Integer, nullable=True)
    p_wave_duration = db.Column(db.Integer, nullable=True)  # in ms
    pq_interval = db.Column(db.Integer, nullable=True)      # in ms
    qrs_duration = db.Column(db.Integer, nullable=True)     # in ms
    qt_interval = db.Column(db.Integer, nullable=True)      # in ms
    classification = db.Column(db.String(50), nullable=True)
    confidence = db.Column(db.Float, nullable=True)         # 0-1 confidence score
    notes = db.Column(db.Text, nullable=True)
    
    # File paths
    wfdb_path = db.Column(db.String(255), nullable=True)
    plot_path = db.Column(db.String(255), nullable=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'file_id': self.file_id,
            'patient_name': self.patient_name or '',
            'age': self.age,
            'gender': self.gender,
            'upload_date': self.upload_date.isoformat() if self.upload_date else None,
            'heart_rate': self.heart_rate,
            'intervals': {
                'p_wave_duration_ms': self.p_wave_duration,
                'pq_interval_ms': self.pq_interval,
                'qrs_duration_ms': self.qrs_duration,
                'qt_interval_ms': self.qt_interval
            },
            'classification': self.classification or '',
            'confidence': self.confidence,
            'notes': self.notes or '',
            'plot_url': f'/plots/{self.file_id}.png' if self.plot_path else None
        }