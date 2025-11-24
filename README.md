# Automatic Face Recognition Attendance System

An intelligent attendance management system that uses automatic face recognition to track attendance with a voting-based verification system.

## Features

- 🤖 **Automatic Recognition**: No need to press buttons - the system continuously monitors and recognizes faces
- 🗳️ **Voting System**: Collects multiple recognition confirmations (votes) before marking attendance
- 📊 **Excel Reports**: Automatic generation of attendance sheets with timestamps
- 🎯 **Reliable Recognition**: Uses LBPH face recognition with adaptive thresholds
- ⚡ **Real-time Feedback**: Live display of vote counts and recognition status
- 🔄 **Auto Database Loading**: Automatically loads faces from `known_faces/` folder on startup

## How It Works

1. **Setup**: The system loads known faces from your database (or from `known_faces/` folder if empty)
2. **Monitoring**: Camera continuously captures and analyzes frames
3. **Voting**: Each successful recognition adds a vote to that person
4. **Marking**: When someone receives 3+ votes, they're automatically marked present
5. **Saving**: Attendance is saved in real-time to an Excel file

## Usage

### First Time Setup

1. Add your photos to the `known_faces/` folder
2. Photos should be clear, frontal face shots
3. Filename should be the person's name (e.g., `Aayush.jpg`, `John.jpg`)

**Important**: The system will automatically load all photos from `known_faces/` when you first run it or after resetting the database.

### Running the System

```bash
python3 main.py
```

### Resetting the Database

If you want to reload all faces from the `known_faces/` folder:

```bash
python3 -c "from face_database import FaceDatabase; db = FaceDatabase(); db.reset_database(); print('Database reset - faces will reload on next run')"
```

### During Operation

1. **Look at the camera** - Face the camera directly
2. **Wait for recognition** - You'll see your votes increase (e.g., "Aayush: 2/3")
3. **Automatic marking** - Once you reach 3 votes, you'll be marked present
4. **Press ESC** - When finished, press ESC to close and save

### What You'll See

- **Green box** with name = Recognized
- **Red box** with "Unknown" = Not recognized
- **Vote counter** showing progress (X/3)
- **Distance values** showing recognition confidence

## System Requirements

- Python 3.7+
- Webcam
- Good lighting
- Clear frontal face view

## Configuration

The system uses these default settings (adjustable in `Manager.py`):

- **Vote Threshold**: 3 votes required to mark present
- **Recognition Threshold**: 150 (confidence level)
- **Frame Skip**: Process every 3rd frame for performance
- **Timeout**: 60 seconds auto-close

## Troubleshooting

### Face Not Recognized

1. **Check lighting** - Ensure good, even lighting on your face
2. **Look directly** - Face should be pointing at camera
3. **Distance** - Sit 2-3 feet from camera
4. **Photo quality** - Original photo should be clear and frontal

### No Votes Counted

- Check the distance value shown - higher = lower confidence
- Ensure your photo in the database is of good quality
- Try adjusting the confidence threshold if consistently too strict

### Camera Not Working

- Close other apps using the camera
- Check camera permissions
- Try restarting the application

### Database Issues

If the database seems empty or faces aren't loading:

```bash
# Check current faces in database
python3 -c "from face_database import FaceDatabase; db = FaceDatabase(); faces = db.get_all_faces(); print(f'Found {len(faces)} faces'); [print(f'  - {name}') for name, _ in faces]"

# Reset database to reload from known_faces folder
python3 -c "from face_database import FaceDatabase; db = FaceDatabase(); db.reset_database()"
```

## File Structure

```
attendance_System_project/
├── main.py                 # Main entry point
├── Manager.py              # Attendance management logic
├── face_database.py        # Face recognition database
├── known_faces/            # Store face photos here (auto-loaded)
│   ├── Aayush.jpg
│   ├── Divyansh.jpg
│   └── ...
├── attendance_records/     # Generated attendance sheets
├── database/               # Face database storage (auto-managed)
└── requirements.txt        # Python dependencies
```

## Dependencies

- opencv-contrib-python: Face recognition
- pandas: Data management
- openpyxl: Excel file handling
- numpy: Array operations

## License

This project is available for personal and educational use.
