import pandas as pd
import os
from datetime import datetime
import cv2
import numpy as np
from collections import deque, defaultdict
import time

class AttendanceManager:
    def __init__(self, face_db):
        self.face_db = face_db
        self.attendance_file = None
        self.known_faces = []
        self.known_names = []
        self.load_known_faces()
    
    def load_known_faces(self):
        """Load all known faces from database"""
        faces_data = self.face_db.get_all_faces()
        
        # De-duplicate names while keeping order
        names_in_order = [name for name, _ in faces_data]
        unique_names = list(dict.fromkeys(names_in_order))
        
        self.known_names = unique_names
        self.known_faces = [encoding for _, encoding in faces_data]
        print(f"Loaded {len(self.known_names)} known faces")
    
    def create_new_attendance_sheet(self):
        """Create a new Excel attendance sheet"""
        timestamp = datetime.now().strftime("%A_%b_%d_%Y")
        os.makedirs("attendance_records", exist_ok=True)
        
        self.attendance_file = f"attendance_records/attendance_{timestamp}.xlsx"
        
        # Initialize attendance data
        attendance_data = {
            'Name': self.known_names,
            'Status': ['Absent'] * len(self.known_names),
            'Time': [''] * len(self.known_names)
        }
        
        df = pd.DataFrame(attendance_data)
        df.to_excel(self.attendance_file, index=False)
        print(f"Created new attendance sheet: {self.attendance_file}")
        
        return self.attendance_file
    
    def capture_and_recognize(self):
        """Capture from webcam and recognize faces automatically with voting system"""
        cap = None
        
        # Try different camera indices
        for camera_index in [0, 1, 2]:
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                print(f"✅ Camera found at index {camera_index}")
                break
            cap.release()
        
        if not cap or not cap.isOpened():
            print("❌ Unable to access camera. Please check:")
            print("  - Camera is not being used by another app")
            print("  - Camera permissions are granted")
            print("  - Camera is connected and working")
            return
        
        print("=== AUTOMATIC ATTENDANCE SYSTEM ===")
        print("📸 System is continuously monitoring...")
        print("👀 Look at the camera for 10-15 seconds")
        print("✅ You'll be marked present after 3 votes")
        print("🚪 Press ESC to finish and save")
        print("=" * 40)
        
        # Voting system - balanced for accuracy
        vote_threshold = 3  # Number of votes needed to mark as present
        frame_counter = 0
        all_votes = defaultdict(int)  # Track all votes over time
        marked_present = set()  # Track who has been marked present
        frame_skip = 3  # Process every 3rd frame for performance
        
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frames periodically
            frame_counter += 1
            if frame_counter % frame_skip == 0:
                # Recognize faces - strict threshold to prevent false positives
                detected_names = self.recognize_faces(frame, confidence_threshold=100.0)
                
                # Add votes for detected faces
                for name in detected_names:
                    all_votes[name] += 1
                
                # Check if anyone reached the vote threshold
                for name, votes in all_votes.items():
                    if name not in marked_present and votes >= vote_threshold:
                        marked_present.add(name)
                        current_time = datetime.now().strftime("%H:%M:%S")
                        print(f"✅ {name} marked PRESENT ({votes} votes) at {current_time}")
                        self.mark_attendance([name])
            
            # Display current status
            display_frame = frame.copy()
            cv2.putText(display_frame, "Automatic Attendance - Look at Camera", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show votes for each person
            y_offset = 60
            cv2.putText(display_frame, "Votes:", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            for i, name in enumerate(self.known_names):
                votes = all_votes.get(name, 0)
                status = "PRESENT ✓" if name in marked_present else f"{votes}/{vote_threshold}"
                color = (0, 255, 0) if name in marked_present else (0, 255, 255)
                
                cv2.putText(display_frame, f"{name}: {status}", (10, y_offset + 25 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            cv2.imshow('Automatic Attendance System - Press ESC to finish', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
            
            # Optional: Auto-exit after 30 seconds of inactivity
            elapsed = time.time() - start_time
            if elapsed > 60:  # 1 minute timeout
                print("\n⏰ Time limit reached")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Final summary
        print("\n" + "=" * 40)
        print("📊 ATTENDANCE SUMMARY")
        print("=" * 40)
        if marked_present:
            print(f"✅ Marked Present: {', '.join(marked_present)}")
        else:
            print("⚠️  No one was marked present")
        print(f"📄 Check results in: {self.attendance_file}")
    
    def recognize_faces(self, frame, confidence_threshold: float = 100.0):
        """Recognize faces in the captured frame"""
        # Convert to grayscale with contrast enhancement
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Match training settings
        gray = clahe.apply(gray)
        
        # Detect faces with more sensitive parameters
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        face_locations = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=2,  # More sensitive
            minSize=(40, 40),  # Smaller minimum size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        present_people = []
        
        for (x, y, w, h) in face_locations:
            face_roi = gray[y:y+h, x:x+w]
            
            # Recognize the face with lenient threshold
            name, distance = self.face_db.recognize_face(face_roi, confidence_threshold=confidence_threshold)
            
            if name:
                present_people.append(name)
                
                # Draw rectangle and label with distance
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({distance:.1f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                # Show distance even for unknown faces
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, f"Unknown ({distance:.1f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        return list(set(present_people))
    
    def mark_attendance(self, present_people):
        """Update attendance in Excel sheet"""
        if not self.attendance_file:
            print("No attendance sheet created!")
            return
        
        # Read current attendance
        df = pd.read_excel(self.attendance_file)
        
        # Update attendance
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Build case-insensitive index
        name_to_index = {str(n).strip().lower(): idx for idx, n in enumerate(df['Name'].values)}
        
        for person in present_people:
            key = str(person).strip().lower()
            if key in name_to_index:
                person_index = name_to_index[key]
                df.at[person_index, 'Status'] = 'Present'
                df.at[person_index, 'Time'] = str(current_time)
        
        # Save updated attendance
        df.to_excel(self.attendance_file, index=False)
