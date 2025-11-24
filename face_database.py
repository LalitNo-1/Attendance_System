import sqlite3
import cv2
import numpy as np
import os
import pickle
import re

class FaceDatabase:
    def __init__(self, db_path="database/faces.db"):
        self.db_path = db_path
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize LBPH recognizer
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
        
        # Maps for label <-> name and adaptive thresholds
        self.label_to_name = {}
        self.name_to_label = {}
        self.label_thresholds = {}
        
        self.init_database()
        
        # Load faces from known_faces folder if database is empty
        faces = self.get_all_faces()
        if len(faces) == 0:
            print("Database is empty. Loading faces from known_faces folder...")
            self.setup_known_faces()
        else:
            self._load_and_train_from_db()
    
    def init_database(self):
        """Initialize SQLite database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                encoding BLOB NOT NULL,
                image_path TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def add_face(self, name, image_path):
        """Add a face encoding to database using OpenCV"""
        # Normalize name
        name = self._normalize_name(name)
        
        # Load and detect the face
        image = cv2.imread(image_path)
        if image is None:
            return False, "Could not load image"
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return False, "No face found in image"
        
        # Get the first detected face
        (x, y, w, h) = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        
        # Resize and normalize
        face_roi = cv2.resize(face_roi, (100, 100))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        face_roi = clahe.apply(face_roi)
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            face_bytes = pickle.dumps(face_roi)
            cursor.execute('''
                INSERT INTO faces (name, encoding, image_path)
                VALUES (?, ?, ?)
            ''', (name, face_bytes, image_path))
            conn.commit()
            
            # Retrain recognizer
            self._load_and_train_from_db()
            return True, "Face added successfully"
        except Exception as e:
            return False, f"Error: {str(e)}"
        finally:
            conn.close()
    
    def get_all_faces(self):
        """Retrieve all face encodings and names"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT name, encoding FROM faces')
        
        faces = []
        for row in cursor.fetchall():
            name = row[0]
            encoding = pickle.loads(row[1])
            faces.append((name, encoding))
        
        conn.close()
        return faces
    
    def delete_face(self, name: str):
        """Delete a face by name"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM faces WHERE name = ?', (name,))
            conn.commit()
        finally:
            conn.close()
        
        self._load_and_train_from_db()
        return True
    
    def reset_database(self):
        """Remove all faces"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM faces')
            conn.commit()
        finally:
            conn.close()
        
        self._load_and_train_from_db()
        return True
    
    def setup_known_faces(self, faces_folder="known_faces"):
        """Setup database with images from known_faces folder"""
        if not os.path.exists(faces_folder):
            os.makedirs(faces_folder)
            print(f"Created {faces_folder} folder. Please add photos.")
            return
        
        entries = os.listdir(faces_folder)
        added_count = 0
        
        for entry in entries:
            entry_path = os.path.join(faces_folder, entry)
            
            if os.path.isdir(entry_path):
                # Person folder with multiple images
                person_name = self._normalize_name(entry)
                for file in os.listdir(entry_path):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(entry_path, file)
                        success, message = self.add_face(person_name, image_path)
                        if success:
                            added_count += 1
                        print(f"{person_name}: {message}")
            else:
                # Single image file
                if entry.lower().endswith(('.jpg', '.jpeg', '.png')):
                    base = os.path.splitext(entry)[0]
                    person_name = self._normalize_name(base)
                    success, message = self.add_face(person_name, entry_path)
                    if success:
                        added_count += 1
                    print(f"{person_name}: {message}")
        
        print(f"\n✅ Total faces added: {added_count}")
    
    def _normalize_name(self, raw_name: str) -> str:
        """Normalize filenames to canonical person name"""
        name = raw_name.strip()
        
        # Remove trailing patterns like (1), (2)
        name = re.sub(r"\s*\(\d+\)\s*$", "", name)
        
        # Remove trailing underscore/dash + digits
        name = re.sub(r"[_-]\d+\s*$", "", name)
        
        # Remove trailing space + digits
        name = re.sub(r"\s\d+\s*$", "", name)
        
        # Collapse extra spaces
        name = re.sub(r"\s+", " ", name)
        
        return name
    
    def recognize_face(self, face_image, confidence_threshold: float = 100.0):
        """Recognize a face using trained LBPH recognizer"""
        if not self.label_to_name:
            return None, float("inf")
        
        # Convert to grayscale and normalize
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image
        
        gray = cv2.resize(gray, (100, 100))
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        try:
            label, confidence = self.face_recognizer.predict(gray)
        except cv2.error:
            return None, float("inf")
        
        name = self.label_to_name.get(label)
        if name is None:
            return None, confidence
        
        # Use strict threshold to prevent false positives
        effective_threshold = confidence_threshold
        
        if confidence <= effective_threshold:
            return name, confidence
        
        return None, confidence
    
    def _load_and_train_from_db(self):
        """Load all faces from DB and train LBPH recognizer"""
        faces = self.get_all_faces()
        
        if not faces:
            self.label_to_name = {}
            self.name_to_label = {}
            self.label_thresholds = {}
            return
        
        images = []
        labels = []
        self.label_to_name = {}
        self.name_to_label = {}
        self.label_thresholds = {}
        next_label = 0
        
        for name, face_img in faces:
            if name not in self.name_to_label:
                self.name_to_label[name] = next_label
                self.label_to_name[next_label] = name
                next_label += 1
            
            # Ensure correct size and normalization
            if len(face_img.shape) == 3:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            face_img = cv2.resize(face_img, (100, 100))
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            face_img = clahe.apply(face_img)
            
            images.append(face_img)
            labels.append(self.name_to_label[name])
        
        if images and labels:
            # Train recognizer
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
            self.face_recognizer.train(images, np.array(labels, dtype=np.int32))
            
            # Calculate adaptive thresholds
            distances_per_label = {lbl: [] for lbl in set(labels)}
            
            for img, lbl in zip(images, labels):
                predicted_label, distance = self.face_recognizer.predict(img)
                if predicted_label == lbl:
                    distances_per_label[lbl].append(distance)
            
            # Strict thresholds to prevent false positives
            default_margin = 20.0  # Tighter margin for more accurate matching
            fallback_threshold = 100.0  # Stricter threshold to reduce false positives
            
            for lbl, dist_list in distances_per_label.items():
                if dist_list:
                    median_distance = float(np.median(dist_list))
                    # Use adaptive threshold with reasonable margin
                    self.label_thresholds[lbl] = min(fallback_threshold, median_distance + default_margin)
                else:
                    self.label_thresholds[lbl] = fallback_threshold
