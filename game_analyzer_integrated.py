
import os
import glob
import time
import psutil
import threading
import mss
import logging
from contextlib import suppress
from datetime import datetime
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
import pyodbc
import json
from typing import Dict, List, Any, Optional

# --- Gemini Integration: Start ---
# Added imports for the machine learning model
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
# --- Gemini Integration: End ---


# Import Virtuoso components
try:
    from virtuoso_query_optimizer import VirtuosoQueryOptimizer, OptimizationLevel
    from virtuoso_query_compiler import QueryCompiler, CompilationMode
    VIRTUOSO_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Virtuoso components not available: {e}")
    VIRTUOSO_AVAILABLE = False

# --- Logging Setup ---
log_file = os.path.expanduser("~/game_analyzer.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler() # Also print to console
    ]
)

# --- Database Configuration ---
load_dotenv('/home/jpouliot/dev/game_analyzer_db.env')
load_dotenv('/home/jpouliot/dev/virtuoso_db.env')

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_DATABASE', 'game_analyzer'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres')
}

VIRTUOSO_CONFIG = {
    'driver': os.getenv('VIRTUOSO_DRIVER', 'virtuoso-odbc'),
    'host': os.getenv('VIRTUOSO_HOST', 'localhost'),
    'port': int(os.getenv('VIRTUOSO_PORT', 1111)),
    'uid': os.getenv('VIRTUOSO_USER', 'dba'),
    'pwd': os.getenv('VIRTUOSO_PASSWORD', 'dba'),
    'database': os.getenv('VIRTUOSO_DATABASE', '')
}

# AI Learning Configuration
AI_CONFIG = {
    'learning_enabled': os.getenv('AI_LEARNING_ENABLED', 'true').lower() == 'true',
    'analysis_level': os.getenv('PERFORMANCE_ANALYSIS_LEVEL', 'aggressive'),
    'auto_optimization': os.getenv('AUTO_OPTIMIZATION', 'true').lower() == 'true',
    'model_type': os.getenv('MACHINE_LEARNING_MODEL', 'adaptive'),
    'rdf_graph': os.getenv('RDF_GRAPH_URI', 'http://gameanalyzer.org/data'),
    'sparql_endpoint': os.getenv('SPARQL_ENDPOINT', 'http://localhost:8890/sparql')
}

# --- Gemini Integration: Start ---
# This new class encapsulates the machine learning model from gem2.py
class PerformanceClassifier:
    """
    A placeholder machine learning model for performance classification.
    This model is trained on the Iris dataset for demonstration purposes.
    """
    def __init__(self):
        self.model = DecisionTreeClassifier(max_depth=3, random_state=42)
        self.scaler = StandardScaler()
        self.target_names = None

    def train(self):
        """Trains the model on the Iris dataset."""
        iris = load_iris()
        X, y = iris.data, iris.target
        self.target_names = iris.target_names
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        logging.info("🤖 Performance Classifier model trained (using Iris dataset).")

    def predict(self, features: List[float]) -> str:
        """
        Predicts the performance class based on a list of features.
        NOTE: The model expects 4 features, as it's trained on the Iris dataset.
        """
        if len(features) != 4:
            logging.warning(f"Classifier expects 4 features, but got {len(features)}. Prediction may be inaccurate.")
            # Pad or truncate features to 4
            features = (features + [0, 0, 0, 0])[:4]
            
        new_sample = np.array([features])
        new_sample_scaled = self.scaler.transform(new_sample)
        prediction = self.model.predict(new_sample_scaled)
        # Map the prediction to a more descriptive class name
        predicted_class = self.target_names[prediction[0]]
        
        # We'll map the Iris classes to performance classes for this demo
        class_mapping = {
            'setosa': 'EXCELLENT',
            'versicolor': 'ACCEPTABLE',
            'virginica': 'POOR'
        }
        return class_mapping.get(predicted_class, "UNKNOWN")
# --- Gemini Integration: End ---


# --- Virtuoso Learning Manager ---

class VirtuosoLearningManager:
    """Handles Virtuoso database operations and AI learning capabilities"""
    
    def __init__(self, config):
        self.config = config
        self.conn = None
        self.optimizer = None
        self.compiler = None
        self.learning_data = []
        self.performance_patterns = {}
        self.optimization_history = {}
        
        # --- Gemini Integration: Start ---
        # Initialize and train the performance classifier
        self.performance_classifier = PerformanceClassifier()
        self.performance_classifier.train()
        # --- Gemini Integration: End ---
        
        self.connect()
    
    def connect(self):
        """Establish Virtuoso database connection"""
        if not VIRTUOSO_AVAILABLE:
            logging.warning("⚠️ Virtuoso components not available, skipping connection")
            return
        
        try:
            # Build connection string
            conn_str = (
                f"DRIVER={{{self.config['driver']}}};"
                f"HOST={self.config['host']};"
                f"PORT={self.config['port']};"
                f"UID={self.config['uid']};"
                f"PWD={self.config['pwd']}"
            )
            
            self.conn = pyodbc.connect(conn_str, autocommit=False)
            logging.info("✅ Connected to Virtuoso learning database")
            
            # Initialize optimizer and compiler
            if AI_CONFIG['learning_enabled']:
                self.optimizer = VirtuosoQueryOptimizer(
                    self.conn,
                    OptimizationLevel.AGGRESSIVE if AI_CONFIG['analysis_level'] == 'aggressive' else OptimizationLevel.MODERATE
                )
                self.compiler = QueryCompiler(self.conn, enable_caching=True)
                logging.info("🧠 AI Learning modules initialized")
            
            self._init_learning_schema()
            
        except Exception as e:
            logging.error(f"❌ Virtuoso connection failed: {e}")
            logging.info("💡 To enable learning: docker run -d --name virtuoso -p 1111:1111 -p 8890:8890 tenforce/virtuoso")
            self.conn = None
    
    def _init_learning_schema(self):
        """Initialize learning-specific tables and RDF graphs"""
        if not self.conn:
            return
        
        try:
            cursor = self.conn.cursor()
            
            # Tables are already created by init_virtuoso_schema.py
            # Just check if they exist
            cursor.execute("SELECT COUNT(*) FROM performance_patterns WHERE 1=0")
            logging.debug("performance_patterns table exists")
            
            self.conn.commit()
            logging.info("📊 Learning schema verified")
            
        except Exception as e:
            logging.debug(f"Learning schema setup: {e}")
    
    def analyze_performance_pattern(self, performance_data: List[Dict[str, Any]], game_name: str) -> Dict[str, Any]:
        """Analyze performance data to identify patterns and learning opportunities"""
        if not performance_data or not AI_CONFIG['learning_enabled']:
            return {}

        try:
            # Extract performance metrics
            fps_values = [d['fps'] for d in performance_data if 'fps' in d]
            cpu_values = [d['cpu_percent'] for d in performance_data if 'cpu_percent' in d]
            gpu_values = [d['gpu_utilization_percent'] for d in performance_data if d.get('gpu_utilization_percent', -1) != -1]
            
            if not fps_values:
                return {}
            
            # Calculate advanced metrics
            avg_fps = sum(fps_values) / len(fps_values)
            max_fps = max(fps_values) if fps_values else 1
            min_fps = min(fps_values) if fps_values else 1
            fps_stability = 1.0 - (max_fps - min_fps) / max(max_fps, 1) if max_fps > 0 else 0.0
            avg_cpu = sum(cpu_values) / len(cpu_values) if cpu_values else 0
            avg_gpu = sum(gpu_values) / len(gpu_values) if gpu_values else 0
            
            # Performance classification
            performance_class = self._classify_performance(avg_fps, fps_stability, avg_cpu, avg_gpu)
            
            # Bottleneck analysis
            bottleneck = self._analyze_bottleneck(avg_cpu, avg_gpu, fps_stability)
            
            # Generate optimization suggestions
            optimizations = self._generate_ai_optimizations(performance_class, bottleneck, avg_fps)
            
            pattern = {
                'game_name': game_name,
                'performance_class': performance_class,
                'avg_fps': avg_fps,
                'fps_stability': fps_stability,
                'avg_cpu': avg_cpu,
                'avg_gpu': avg_gpu,
                'bottleneck': bottleneck,
                'optimizations': optimizations,
                'confidence': self._calculate_confidence(len(fps_values), fps_stability),
                'timestamp': datetime.now()
            }
            
            # Store pattern for learning
            self._store_learning_pattern(pattern)
            
            return pattern
            
        except Exception as e:
            logging.error(f"Pattern analysis error: {e}")
            return {}

    # --- Gemini Integration: Start ---
    # The original _classify_performance is replaced with this one, which uses the ML model.
    def _classify_performance(self, avg_fps: float, stability: float, cpu: float, gpu: float) -> str:
        """
        Classify performance level using the integrated machine learning model.
        NOTE: This is a demonstration. The model is trained on unrelated data.
        """
        # We need to pass 4 features to the model. We'll use the available metrics.
        # The mapping is arbitrary for this demonstration.
        features = [avg_fps, stability * 100, cpu, gpu]
        return self.performance_classifier.predict(features)
    # --- Gemini Integration: End ---

    def _analyze_bottleneck(self, cpu: float, gpu: float, fps_stability: float) -> str:
        """Advanced bottleneck analysis using machine learning patterns"""
        # Enhanced bottleneck detection
        if cpu > 90 and fps_stability < 0.7:
            return "CPU_BOTTLENECK"
        elif gpu > 95 and fps_stability < 0.8:
            return "GPU_BOTTLENECK"
        elif cpu > 80 and gpu > 85:
            return "SYSTEM_BOTTLENECK"
        elif fps_stability < 0.6:
            return "FRAME_PACING_ISSUE"
        else:
            return "NONE"
    
    def _generate_ai_optimizations(self, performance_class: str, bottleneck: str, avg_fps: float) -> List[Dict[str, Any]]:
        """Generate AI-powered optimization recommendations"""
        optimizations = []
        
        # Performance-based recommendations
        if performance_class in ["POOR", "CRITICAL"]:
            optimizations.extend([
                {"type": "graphics", "action": "lower_resolution", "priority": "HIGH", "expected_gain": 15.0},
                {"type": "graphics", "action": "disable_aa", "priority": "MEDIUM", "expected_gain": 8.0},
                {"type": "graphics", "action": "lower_shadows", "priority": "MEDIUM", "expected_gain": 12.0}
            ])
        
        # Bottleneck-specific recommendations
        if bottleneck == "CPU_BOTTLENECK":
            optimizations.extend([
                {"type": "system", "action": "close_background_apps", "priority": "HIGH", "expected_gain": 10.0},
                {"type": "game", "action": "lower_draw_distance", "priority": "MEDIUM", "expected_gain": 7.0},
                {"type": "system", "action": "cpu_affinity", "priority": "LOW", "expected_gain": 3.0}
            ])
        elif bottleneck == "GPU_BOTTLENECK":
            optimizations.extend([
                {"type": "graphics", "action": "lower_texture_quality", "priority": "HIGH", "expected_gain": 12.0},
                {"type": "graphics", "action": "disable_post_processing", "priority": "MEDIUM", "expected_gain": 8.0},
                {"type": "system", "action": "gpu_overclock", "priority": "LOW", "expected_gain": 5.0}
            ])
        
        # Add confidence scores based on historical data
        for opt in optimizations:
            opt['confidence'] = self._get_optimization_confidence(opt['action'])
        
        return optimizations
    
    def _calculate_confidence(self, data_points: int, stability: float) -> float:
        """Calculate confidence score for the analysis"""
        base_confidence = min(data_points / 100.0, 1.0)  # More data = higher confidence
        stability_factor = stability  # Better stability = higher confidence
        return (base_confidence * 0.6 + stability_factor * 0.4) * 100
    
    def _store_learning_pattern(self, pattern: Dict[str, Any]):
        """Store performance pattern for machine learning"""
        if not self.conn or not AI_CONFIG['learning_enabled']:
            return
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO performance_patterns 
                (game_name, avg_fps, avg_cpu, avg_gpu, bottleneck_type, optimization_score, pattern_confidence, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, NOW(), NOW())
            """, (
                pattern['game_name'],
                pattern['avg_fps'],
                pattern['avg_cpu'],
                pattern['avg_gpu'],
                pattern['bottleneck'],
                pattern.get('optimization_score', 0.0),
                pattern['confidence']
            ))
            self.conn.commit()
            
            # Store in local learning cache
            self.learning_data.append(pattern)
            if len(self.learning_data) > 1000:  # Keep last 1000 patterns
                self.learning_data.pop(0)
                
            logging.debug(f"📚 Stored learning pattern for {pattern['game_name']}")
            
        except Exception as e:
            logging.error(f"Error storing learning pattern: {e}")
    
    def _get_optimization_confidence(self, optimization: str) -> float:
        """Get confidence score for specific optimization based on historical data"""
        # This would normally query historical optimization results
        # For now, return default confidence scores
        confidence_map = {
            "lower_resolution": 0.95,
            "disable_aa": 0.85,
            "lower_shadows": 0.90,
            "close_background_apps": 0.80,
            "lower_draw_distance": 0.75,
            "cpu_affinity": 0.60,
            "lower_texture_quality": 0.88,
            "disable_post_processing": 0.82,
            "gpu_overclock": 0.50
        }
        return confidence_map.get(optimization, 0.70)
    
    def get_learning_insights(self, game_name: str) -> Dict[str, Any]:
        """Get AI learning insights for a specific game"""
        if not self.conn or not AI_CONFIG['learning_enabled']:
            return {"insights": "Learning disabled or Virtuoso unavailable"}
        
        try:
            cursor = self.conn.cursor()
            
            # Get historical patterns
            cursor.execute("""
                SELECT AVG(avg_fps) as avg_fps, AVG(optimization_score) as avg_opt_score,
                       COUNT(*) as session_count, AVG(pattern_confidence) as avg_confidence
                FROM performance_patterns 
                WHERE game_name = ? AND created_at > DATEADD('day', -30, NOW())
            """, (game_name,))
            
            stats = cursor.fetchone()
            
            insights = {
                "game": game_name,
                "30_day_avg_fps": stats[0] if stats[0] else 0,
                "optimization_score": stats[1] if stats[1] else 0,
                "sessions_analyzed": stats[2] if stats[2] else 0,
                "confidence": stats[3] if stats[3] else 0,
                "learning_status": "active" if AI_CONFIG['learning_enabled'] else "disabled"
            }
            
            return insights
            
        except Exception as e:
            logging.error(f"Error getting learning insights: {e}")
            return {"error": str(e)}
    
    def close(self):
        """Close Virtuoso connection"""
        if self.conn:
            self.conn.close()
            logging.info("🔌 Virtuoso learning connection closed")

# --- Database Management ---

class DatabaseManager:
    """Handles all database operations for game performance data"""
    
    def __init__(self, config):
        self.config = config
        self.conn = None
        self.current_game_id = None
        self.current_session_id = None
        self.connect()
    
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**self.config)
            logging.info("✅ Connected to PostgreSQL database")
        except Exception as e:
            logging.error(f"❌ Failed to connect to database: {e}")
            self.conn = None
    
    def ensure_connection(self):
        """Ensure database connection is active"""
        if not self.conn or self.conn.closed:
            self.connect()
        return self.conn is not None
    
    def get_or_create_game(self, game_name, installation_path=None):
        """Get existing game record or create new one"""
        if not self.ensure_connection():
            return None
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Check if game exists
                cur.execute("SELECT id FROM games WHERE name = %s", (game_name,))
                result = cur.fetchone()
                
                if result:
                    self.current_game_id = result['id']
                    # Update last played time
                    cur.execute("UPDATE games SET last_played = %s WHERE id = %s", 
                              (datetime.now(), self.current_game_id))
                else:
                    # Create new game record
                    cur.execute(
                        """INSERT INTO games (name, installation_path, discovered_at, last_played) 
                           VALUES (%s, %s, %s, %s) RETURNING id""",
                        (game_name, installation_path, datetime.now(), datetime.now())
                    )
                    self.current_game_id = cur.fetchone()['id']
                    logging.info(f"📝 Created new game record: {game_name}")
                
                self.conn.commit()
                return self.current_game_id
        except Exception as e:
            logging.error(f"Database error in get_or_create_game: {e}")
            return None
    
    def start_session(self, game_id):
        """Start a new gaming session"""
        if not self.ensure_connection():
            return None
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """INSERT INTO game_sessions (game_id, started_at) 
                       VALUES (%s, %s) RETURNING id""",
                    (game_id, datetime.now())
                )
                self.current_session_id = cur.fetchone()['id']
                self.conn.commit()
                logging.info(f"🎮 Started gaming session #{self.current_session_id}")
                return self.current_session_id
        except Exception as e:
            logging.error(f"Database error in start_session: {e}")
            return None
    
    def store_performance_data(self, data_point):
        """Store a single performance data point"""
        if not self.ensure_connection() or not self.current_session_id:
            return False
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO performance_data 
                       (session_id, game_id, timestamp, fps, cpu_percent, 
                        gpu_utilization_percent, gpu_temperature_celsius) 
                       VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                    (
                        self.current_session_id,
                        self.current_game_id,
                        datetime.fromtimestamp(data_point['timestamp']),
                        data_point['fps'],
                        data_point['cpu_percent'],
                        data_point['gpu_utilization_percent'] if data_point['gpu_utilization_percent'] != -1 else None,
                        data_point['gpu_temperature_celsius'] if data_point['gpu_temperature_celsius'] != -1 else None
                    )
                )
                self.conn.commit()
                return True
        except Exception as e:
            logging.error(f"Database error storing performance data: {e}")
            return False
    
    def end_session(self, performance_data):
        """End the current gaming session and update statistics"""
        if not self.ensure_connection() or not self.current_session_id:
            return
        
        try:
            with self.conn.cursor() as cur:
                # Calculate session statistics
                if performance_data:
                    avg_fps = sum(d['fps'] for d in performance_data) / len(performance_data)
                    min_fps = min(d['fps'] for d in performance_data)
                    max_fps = max(d['fps'] for d in performance_data)
                    avg_cpu = sum(d['cpu_percent'] for d in performance_data) / len(performance_data)
                    max_cpu = max(d['cpu_percent'] for d in performance_data)
                    
                    avg_gpu = max_gpu = avg_gpu_temp = max_gpu_temp = None
                    if any(d['gpu_utilization_percent'] != -1 for d in performance_data):
                        gpu_data = [d['gpu_utilization_percent'] for d in performance_data if d['gpu_utilization_percent'] != -1]
                        temp_data = [d['gpu_temperature_celsius'] for d in performance_data if d['gpu_temperature_celsius'] != -1]
                        if gpu_data:
                            avg_gpu = sum(gpu_data) / len(gpu_data)
                            max_gpu = max(gpu_data)
                        if temp_data:
                            avg_gpu_temp = sum(temp_data) / len(temp_data)
                            max_gpu_temp = max(temp_data)
                    
                    # Detect bottlenecks
                    bottleneck = "NONE"
                    cpu_bottleneck_points = sum(1 for d in performance_data if d['cpu_percent'] > 90)
                    gpu_bottleneck_points = sum(1 for d in performance_data if d['gpu_utilization_percent'] > 95 and d['gpu_utilization_percent'] != -1)
                    
                    if cpu_bottleneck_points > len(performance_data) * 0.1:
                        bottleneck = "CPU_BOTTLENECK"
                    elif gpu_bottleneck_points > len(performance_data) * 0.1:
                        bottleneck = "GPU_BOTTLENECK"
                else:
                    avg_fps = min_fps = max_fps = avg_cpu = max_cpu = None
                    avg_gpu = max_gpu = avg_gpu_temp = max_gpu_temp = None
                    bottleneck = "NONE"
                
                # Update session record
                cur.execute("""UPDATE game_sessions SET 
                       ended_at = %s, duration_seconds = EXTRACT(EPOCH FROM (%s - started_at)),
                       average_fps = %s, min_fps = %s, max_fps = %s,
                       average_cpu_usage = %s, max_cpu_usage = %s,
                       average_gpu_usage = %s, max_gpu_usage = %s,
                       average_gpu_temp = %s, max_gpu_temp = %s,
                       data_points_collected = %s, bottleneck_detected = %s
                       WHERE id = %s""",
                    (
                        datetime.now(), datetime.now(),
                        avg_fps, min_fps, max_fps,
                        avg_cpu, max_cpu,
                        avg_gpu, max_gpu,
                        avg_gpu_temp, max_gpu_temp,
                        len(performance_data), bottleneck,
                        self.current_session_id
                    )
                )
                
                # Update game statistics
                if avg_fps is not None:
                    cur.execute(
                        """UPDATE games SET 
                           average_fps = (SELECT AVG(average_fps) FROM game_sessions WHERE game_id = %s AND average_fps IS NOT NULL),
                           average_cpu_usage = (SELECT AVG(average_cpu_usage) FROM game_sessions WHERE game_id = %s AND average_cpu_usage IS NOT NULL),
                           average_gpu_usage = (SELECT AVG(average_gpu_usage) FROM game_sessions WHERE game_id = %s AND average_gpu_usage IS NOT NULL),
                           average_gpu_temp = (SELECT AVG(average_gpu_temp) FROM game_sessions WHERE game_id = %s AND average_gpu_temp IS NOT NULL),
                           total_play_time_seconds = COALESCE((SELECT SUM(duration_seconds) FROM game_sessions WHERE game_id = %s), 0)
                           WHERE id = %s""",
                        (self.current_game_id, self.current_game_id, self.current_game_id, 
                         self.current_game_id, self.current_game_id, self.current_game_id)
                    )
                
                self.conn.commit()
                logging.info(f"📊 Session #{self.current_session_id} ended and statistics updated")
                
        except Exception as e:
            logging.error(f"Database error in end_session: {e}")
        finally:
            self.current_session_id = None
    
    def get_game_performance_history(self, game_name, limit=10):
        """Get recent performance history for a game"""
        if not self.ensure_connection():
            return []
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """SELECT gs.*, g.name as game_name 
                       FROM game_sessions gs 
                       JOIN games g ON gs.game_id = g.id 
                       WHERE g.name = %s 
                       ORDER BY gs.started_at DESC 
                       LIMIT %s""",
                    (game_name, limit)
                )
                return cur.fetchall()
        except Exception as e:
            logging.error(f"Database error getting performance history: {e}")
            return []
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logging.info("🔌 Database connection closed")

# --- Game Discovery ---

def find_steam_games():
    """
    Finds installed Steam games on Linux by checking common installation paths.
    Returns a list of discovered game names.
    """
    game_paths = []
    common_steam_paths = [
        os.path.expanduser("~/.steam/steam/steamapps/common/"),
        os.path.expanduser("~/.local/share/Steam/steamapps/common/")
    ]

    for base_path in common_steam_paths:
        if os.path.exists(base_path):
            for game_dir in os.listdir(base_path):
                full_game_path = os.path.join(base_path, game_dir)
                if os.path.isdir(full_game_path):
                    game_paths.append(game_dir)
    return sorted(list(set(game_paths)))

def is_game_running(game_names):
    """
    Checks if any of the specified games are currently running.
    Returns the name of the running game if found, otherwise None.
    """
    for proc in psutil.process_iter(['name', 'cmdline']):
        try:
            process_name = proc.info['name'].lower()
            cmdline = ' '.join(proc.info['cmdline']).lower() if proc.info['cmdline'] else ''

            for game_name in game_names:
                lower_game_name = game_name.lower().replace(" ", "")
                if lower_game_name in process_name or lower_game_name in cmdline:
                    return game_name
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return None

# --- Performance Monitoring ---

class PerformanceMonitor:
    def __init__(self, db_manager=None, virtuoso_manager=None):
        self._stop_event = threading.Event()
        self.monitoring_thread = None
        self.performance_data = []
        self.is_monitoring = False
        self.db_manager = db_manager
        self.virtuoso_manager = virtuoso_manager
        self.current_game = None
        self.learning_patterns = []
        self.ai_optimizations = []
        
        self.gpu_available = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.pynvml = pynvml
            if self.pynvml.nvmlDeviceGetCount() > 0:
                self.gpu_available = True
                logging.info("NVIDIA GPU detected. GPU monitoring enabled.")
            else:
                logging.info("No NVIDIA GPU detected. GPU monitoring disabled.")
        except Exception as e:
            logging.warning(f"NVIDIA monitoring unavailable: {e}. GPU monitoring disabled.")

        self.suggestions_kb = {
            "CPU_BOTTLENECK": [
                "Try lowering settings related to physics, world density, or population.",
                "Close other CPU-intensive applications running in the background.",
                "Check for CPU overheating, as this can cause throttling."
            ],
            "GPU_BOTTLENECK": [
                "Lower graphics settings like shadow quality, textures, anti-aliasing, and ambient occlusion.",
                "Reduce the game's screen resolution.",
                "Ensure your GPU drivers are up to date."
            ],
            "GENERAL": [
                "Ensure your system is well-ventilated to prevent thermal throttling.",
                "Check for and install any available game patches or updates.",
                "Consider upgrading your hardware if performance issues persist across multiple games."
            ]
        }

    def start(self, game_name=None):
        if self.is_monitoring:
            logging.info("Monitoring is already running.")
            return
        
        logging.info("Starting performance monitoring...")
        self._stop_event.clear()
        self.performance_data = []
        self.current_game = game_name
        
        # Initialize database session if database manager is available
        if self.db_manager and game_name:
            game_id = self.db_manager.get_or_create_game(game_name)
            if game_id:
                self.db_manager.start_session(game_id)
        
        self.monitoring_thread = threading.Thread(target=self._monitor_loop)
        self.monitoring_thread.start()
        self.is_monitoring = True

    def stop(self):
        if not self.is_monitoring:
            logging.info("Monitoring is not running.")
            return

        logging.info("Stopping performance monitoring...")
        self._stop_event.set()
        self.monitoring_thread.join()
        self.is_monitoring = False
        logging.info(f"Monitoring stopped. Collected data points: {len(self.performance_data)}")
        
        # AI Learning Analysis
        if self.virtuoso_manager and AI_CONFIG['learning_enabled'] and self.current_game:
            logging.info("🧠 Running AI learning analysis...")
            pattern = self.virtuoso_manager.analyze_performance_pattern(
                self.performance_data, 
                self.current_game
            )
            if pattern:
                self.learning_patterns.append(pattern)
                self.ai_optimizations = pattern.get('optimizations', [])
                self.show_ai_insights(pattern)
        
        # End database session if database manager is available
        if self.db_manager:
            self.db_manager.end_session(self.performance_data)
        
        self.analyze_data()
        self.show_performance_history()
        
        # Show AI recommendations
        if self.ai_optimizations:
            self.show_ai_recommendations()

    def _monitor_loop(self):
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                last_frame_time = time.time()
                frame_count = 0
                
                while not self._stop_event.is_set():
                    try:
                        sct.grab(monitor)
                    except mss.exception.ScreenShotError as e:
                        logging.warning(f"Screenshot failed: {e}. Stopping FPS monitoring for this session.")
                        break  # Exit the monitoring loop
                    
                    frame_count += 1
                    current_time = time.time()
                    delta = current_time - last_frame_time
                    
                    if delta >= 1:
                        fps = frame_count / delta
                        cpu_percent = psutil.cpu_percent()
                        
                        gpu_util = gpu_temp = -1
                        if self.gpu_available:
                            with suppress(self.pynvml.NVMLError):
                                handle = self.pynvml.nvmlDeviceGetHandleByIndex(0)
                                util = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
                                gpu_util = util.gpu
                                gpu_temp = self.pynvml.nvmlDeviceGetTemperature(handle, self.pynvml.NVML_TEMPERATURE_GPU)

                        data_point = {
                            "timestamp": current_time,
                            "fps": fps,
                            "cpu_percent": cpu_percent,
                            "gpu_utilization_percent": gpu_util,
                            "gpu_temperature_celsius": gpu_temp,
                        }
                        self.performance_data.append(data_point)
                        
                        # Store in database if available
                        if self.db_manager:
                            self.db_manager.store_performance_data(data_point)
                        
                        frame_count = 0
                        last_frame_time = current_time

                    time.sleep(0.01)
        except Exception as e:
            logging.error(f"An error occurred in the monitoring loop: {e}")
        finally:
            if self.gpu_available:
                self.pynvml.nvmlShutdown()

    def analyze_data(self):
        if not self.performance_data:
            logging.info("No performance data collected.")
            return

        avg_fps = sum(d['fps'] for d in self.performance_data) / len(self.performance_data)
        avg_cpu = sum(d['cpu_percent'] for d in self.performance_data) / len(self.performance_data)
        
        logging.info("\n--- Performance Analysis ---")
        logging.info(f"Average FPS: {avg_fps:.2f}")
        logging.info(f"Average CPU Usage: {avg_cpu:.2f}%")

        if self.gpu_available:
            avg_gpu_util = sum(d['gpu_utilization_percent'] for d in self.performance_data) / len(self.performance_data)
            avg_gpu_temp = sum(d['gpu_temperature_celsius'] for d in self.performance_data) / len(self.performance_data)
            logging.info(f"Average GPU Utilization: {avg_gpu_util:.2f}%")
            logging.info(f"Average GPU Temperature: {avg_gpu_temp:.2f}°C")
        
        issues = self._detect_issues(avg_fps)
        self._generate_suggestions(issues)

    def _detect_issues(self, avg_fps):
        issues = set()
        fps_drop_threshold = avg_fps * 0.8 

        for point in self.performance_data:
            if point['fps'] < fps_drop_threshold:
                if point['cpu_percent'] > 90:
                    issues.add("CPU_BOTTLENECK")
                if self.gpu_available and point['gpu_utilization_percent'] > 95:
                    issues.add("GPU_BOTTLENECK")
        return list(issues)

    def _generate_suggestions(self, issues):
        logging.info("\n--- Improvement Suggestions ---")
        if not issues:
            logging.info("No specific performance bottlenecks were detected, but here are some general tips:")
            issues.append("GENERAL")
        
        suggestions_made = set()
        for issue in issues:
            if issue in self.suggestions_kb:
                for suggestion in self.suggestions_kb[issue]:
                    if suggestion not in suggestions_made:
                        logging.info(f"- {suggestion}")
                        suggestions_made.add(suggestion)
        logging.info("-----------------------------\n")

    def show_performance_history(self):
        """Display recent performance history from database"""
        if not self.db_manager or not self.current_game:
            return
        
        history = self.db_manager.get_game_performance_history(self.current_game, limit=5)
        if not history:
            return
        
        logging.info("\n--- Recent Performance History ---")
        for session in history:
            duration_min = int(session['duration_seconds'] / 60) if session['duration_seconds'] else 0
            bottleneck_str = f" [{session['bottleneck_detected']}]" if session['bottleneck_detected'] != 'NONE' else ""
            
            logging.info(f"Session {session['started_at'].strftime('%Y-%m-%d %H:%M')} "
                        f"({duration_min}min): "
                        f"FPS {session['average_fps']:.1f} "
                        f"CPU {session['average_cpu_usage']:.1f}%"
                        f"{bottleneck_str}")
        logging.info("--------------------------------\n")

    def show_ai_insights(self, pattern: Dict[str, Any]):
        """Display AI learning insights"""
        logging.info("\n🧠 === AI LEARNING INSIGHTS ===")
        logging.info(f"🎮 Game: {pattern['game_name']}")
        logging.info(f"📊 Performance Class: {pattern['performance_class']}")
        logging.info(f"🎯 Average FPS: {pattern['avg_fps']:.1f}")
        logging.info(f"📈 Stability Score: {pattern['fps_stability']:.2f}")
        logging.info(f"⚠️ Bottleneck: {pattern['bottleneck']}")
        logging.info(f"🎓 AI Confidence: {pattern['confidence']:.1f}%")
        
        if self.virtuoso_manager:
            insights = self.virtuoso_manager.get_learning_insights(pattern['game_name'])
            if 'sessions_analyzed' in insights:
                logging.info(f"📚 Sessions Analyzed: {insights['sessions_analyzed']}")
                logging.info(f"📊 30-Day Avg FPS: {insights['30_day_avg_fps']:.1f}")
        
        logging.info("==============================\n")

    def show_ai_recommendations(self):
        """Display AI-powered optimization recommendations"""
        if not self.ai_optimizations:
            return
        
        logging.info("\n🤖 === AI OPTIMIZATION RECOMMENDATIONS ===")
        
        # Sort by priority and expected gain
        priority_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        sorted_opts = sorted(
            self.ai_optimizations, 
            key=lambda x: (priority_order.get(x['priority'], 0), x['expected_gain']), 
            reverse=True
        )
        
        for i, opt in enumerate(sorted_opts[:5], 1):  # Show top 5
            priority_emoji = "🔥" if opt['priority'] == "HIGH" else "⚡" if opt['priority'] == "MEDIUM" else "💡"
            logging.info(f"{i}. {priority_emoji} [{opt['priority']}] {opt['action'].replace('_', ' ').title()}")
            logging.info(f"   Category: {opt['type'].title()}")
            logging.info(f"   Expected FPS Gain: +{opt['expected_gain']:.1f}")
            logging.info(f"   AI Confidence: {opt['confidence']:.0%}")
            logging.info("")
        
        if AI_CONFIG['auto_optimization']:
            logging.info("🚀 Auto-optimization is enabled - some changes may be applied automatically")
        
        logging.info("==========================================\n")


if __name__ == "__main__":
    logging.info("🚀 Game Analyzer with AI Learning & Virtuoso Integration Starting...")
    
    # Initialize database manager
    db_manager = None
    try:
        db_manager = DatabaseManager(DB_CONFIG)
        logging.info("📊 PostgreSQL database integration enabled")
    except Exception as e:
        logging.warning(f"⚠️ PostgreSQL unavailable, running in offline mode: {e}")
    
    # Initialize Virtuoso learning manager
    virtuoso_manager = None
    if VIRTUOSO_AVAILABLE and AI_CONFIG['learning_enabled']:
        try:
            virtuoso_manager = VirtuosoLearningManager(VIRTUOSO_CONFIG)
            if virtuoso_manager.conn:
                logging.info("🧠 AI Learning with Virtuoso enabled")
            else:
                logging.info("💡 To enable AI learning: docker run -d --name virtuoso -p 1111:1111 -p 8890:8890 tenforce/virtuoso")
        except Exception as e:
            logging.warning(f"⚠️ Virtuoso learning unavailable: {e}")
    
    logging.info("Discovering Steam games...")
    games = find_steam_games()
    if not games:
        logging.info("No Steam games found in common installation paths.")
        exit()

    logging.info("Found the following Steam games:")
    for game in games:
        logging.info(f"- {game}")

    if AI_CONFIG['learning_enabled']:
        logging.info(f"🧠 AI Learning Mode: {AI_CONFIG['analysis_level'].upper()}")
        logging.info(f"🤖 Auto-optimization: {'Enabled' if AI_CONFIG['auto_optimization'] else 'Disabled'}")

    logging.info("\nMonitoring for running games (Ctrl+C to stop)...")
    
    monitor = PerformanceMonitor(db_manager, virtuoso_manager)
    currently_running_game = None

    try:
        while True:
            running_game = is_game_running(games)
            
            if running_game and not monitor.is_monitoring:
                logging.info(f"\nGame '{running_game}' detected. Starting performance monitoring.")
                currently_running_game = running_game
                monitor.start(running_game)
            elif not running_game and monitor.is_monitoring:
                logging.info(f"\nGame '{currently_running_game}' stopped. Stopping performance monitoring.")
                monitor.stop()
                currently_running_game = None

            time.sleep(5)
    except KeyboardInterrupt:
        logging.info("\nStopping application.")
        if monitor.is_monitoring:
            monitor.stop()
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        if monitor.is_monitoring:
            monitor.stop()
    finally:
        if db_manager:
            db_manager.close()
        if virtuoso_manager:
            virtuoso_manager.close()
