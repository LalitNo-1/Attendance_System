#!/usr/bin/env python3

from face_database import FaceDatabase
from Manager import AttendanceManager

def main():
    print("=== 🚀 AUTOMATED ATTENDANCE SYSTEM ===")
    print("=" * 50)
    
    # Initialize components
    print("📊 Initializing system...")
    face_db = FaceDatabase()
    attendance_manager = AttendanceManager(face_db)
    
    print(f"✅ System ready! {len(attendance_manager.known_names)} faces loaded")
    print(f"👥 Known faces: {', '.join(attendance_manager.known_names)}")
    print()
    
    # Create new attendance sheet
    print("📋 Creating new attendance sheet...")
    sheet_path = attendance_manager.create_new_attendance_sheet()
    print(f"✅ Sheet created: {sheet_path}")
    print()
    
    # Start attendance capture
    print("🎥 Starting automatic attendance system...")
    print("INSTRUCTIONS:")
    print("  👀 Look at the camera")
    print("  ⏱️  System automatically recognizes and marks attendance")
    print("  ✅ You'll be marked present after 3 successful recognitions")
    print("  🚪 Press ESC to finish")
    print()
    
    # Start the capture
    attendance_manager.capture_and_recognize()
    
    print("\n🎯 Attendance session complete!")
    print(f"📊 Check your results in: {sheet_path}")

if __name__ == "__main__":
    main()

#source attendance_System_project/bin/activate
#python main.py
#python3 -c "from face_database import FaceDatabase; db = FaceDatabase(); db.reset_database(); print('Database reset - faces will reload on next run')"
