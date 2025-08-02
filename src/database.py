import sqlite3
import os
import json
from datetime import datetime
from pathlib import Path

class TrainingDatabase:
    """Database for managing training data and model metadata"""
    
    def __init__(self, db_path="training_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Training sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                status TEXT,
                epochs INTEGER,
                final_accuracy REAL,
                final_loss REAL,
                model_path TEXT,
                data_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Uploaded data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS uploaded_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                file_path TEXT,
                file_size INTEGER,
                upload_time TIMESTAMP,
                processed BOOLEAN DEFAULT FALSE,
                class_label TEXT,
                session_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES training_sessions (session_id)
            )
        ''')
        
        # Model versions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT UNIQUE,
                model_path TEXT,
                accuracy REAL,
                loss REAL,
                training_session_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (training_session_id) REFERENCES training_sessions (session_id)
            )
        ''')
        
        # Prediction history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                predicted_class TEXT,
                confidence REAL,
                actual_class TEXT,
                model_version TEXT,
                prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_training_session(self, session_id, data_path):
        """Create a new training session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO training_sessions (session_id, start_time, status, data_path)
            VALUES (?, ?, ?, ?)
        ''', (session_id, datetime.now(), 'started', data_path))
        
        conn.commit()
        conn.close()
        return session_id
    
    def update_training_session(self, session_id, **kwargs):
        """Update training session with results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        update_fields = []
        values = []
        
        for key, value in kwargs.items():
            if key in ['end_time', 'status', 'epochs', 'final_accuracy', 'final_loss', 'model_path']:
                update_fields.append(f"{key} = ?")
                values.append(value)
        
        if update_fields:
            values.append(session_id)
            query = f"UPDATE training_sessions SET {', '.join(update_fields)} WHERE session_id = ?"
            cursor.execute(query, values)
        
        conn.commit()
        conn.close()
    
    def save_uploaded_data(self, filename, file_path, file_size, session_id, class_label=None):
        """Save uploaded data information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO uploaded_data (filename, file_path, file_size, upload_time, session_id, class_label)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (filename, file_path, file_size, datetime.now(), session_id, class_label))
        
        conn.commit()
        conn.close()
    
    def get_uploaded_data_for_session(self, session_id):
        """Get all uploaded data for a training session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT filename, file_path, file_size, class_label, processed
            FROM uploaded_data
            WHERE session_id = ?
        ''', (session_id,))
        
        data = cursor.fetchall()
        conn.close()
        
        return [
            {
                'filename': row[0],
                'file_path': row[1],
                'file_size': row[2],
                'class_label': row[3],
                'processed': bool(row[4])
            }
            for row in data
        ]
    
    def save_model_version(self, version, model_path, accuracy, loss, training_session_id):
        """Save model version information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO model_versions (version, model_path, accuracy, loss, training_session_id)
            VALUES (?, ?, ?, ?, ?)
        ''', (version, model_path, accuracy, loss, training_session_id))
        
        conn.commit()
        conn.close()
    
    def save_prediction(self, filename, predicted_class, confidence, actual_class=None, model_version=None):
        """Save prediction result"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions (filename, predicted_class, confidence, actual_class, model_version)
            VALUES (?, ?, ?, ?, ?)
        ''', (filename, predicted_class, confidence, actual_class, model_version))
        
        conn.commit()
        conn.close()
    
    def get_prediction_history(self, limit=50):
        """Get recent prediction history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT filename, predicted_class, confidence, actual_class, model_version, prediction_time
            FROM predictions
            ORDER BY prediction_time DESC
            LIMIT ?
        ''', (limit,))
        
        data = cursor.fetchall()
        conn.close()
        
        return [
            {
                'filename': row[0],
                'predicted_class': row[1],
                'confidence': row[2],
                'actual_class': row[3],
                'model_version': row[4],
                'prediction_time': row[5]
            }
            for row in data
        ]
    
    def get_training_sessions(self, limit=10):
        """Get recent training sessions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT session_id, start_time, end_time, status, epochs, final_accuracy, final_loss
            FROM training_sessions
            ORDER BY created_at DESC
            LIMIT ?
        ''', (limit,))
        
        data = cursor.fetchall()
        conn.close()
        
        return [
            {
                'session_id': row[0],
                'start_time': row[1],
                'end_time': row[2],
                'status': row[3],
                'epochs': row[4],
                'final_accuracy': row[5],
                'final_loss': row[6]
            }
            for row in data
        ]

# Global database instance
db = TrainingDatabase() 