# Vision Access System

A face recognition system for enterprise access control.

The project was developed as a diploma project for the Information Systems educational program. The system identifies users by face, checks face authenticity using an anti-spoofing module, and makes an access decision.

## Main Features

- real-time face recognition;
- user registration and management;
- creation and comparison of biometric face templates;
- anti-spoofing protection;
- encryption of personal and biometric data;
- administrator authentication;
- local SQLite database;
- Arduino integration for physical access control;
- administrator GUI and terminal mode;
- Windows EXE build.

## Technologies

- Python
- OpenCV
- PyTorch
- FaceNet / facenet-pytorch
- SQLite
- NumPy
- cryptography / Fernet
- Arduino
- PyInstaller

## How It Works

```text
Camera
  ↓
Face Detection
  ↓
Image Preprocessing
  ↓
FaceNet Embedding
  ↓
User Matching
  ↓
Anti-Spoofing Check
  ↓
Access Decision
  ↓
Hardware Controller
```

## Project Structure

```text
Vision-access-system/
├── app/              # main application code
├── assets/           # icons, fonts and UI resources
├── models/           # face recognition and anti-spoofing models
├── scripts/          # training, testing and evaluation scripts
├── requirements.txt  # Python dependencies
└── README.md
```

## Data Security

The system includes protection for sensitive information:

- encryption of biometric templates;
- encryption of user personal data;
- administrator password hashing;
- separate storage of encryption keys.

The public repository does not include:

- registered user database;
- face photos and datasets;
- personal FaceNet embeddings;
- encryption keys;
- local confidential data.

## Installation

Clone the repository:

```bash
git clone https://github.com/rrrim0/Vision-access-system.git
cd Vision-access-system
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python -m app.main
```

## Important Note

Some model files are not included in the public repository due to file size limitations and biometric data protection requirements.

In particular, some FaceNet weights and local biometric template files are required separately.

## Project Approval

The project was presented at the II Student Innovation Solutions Competition **InnoProject Challenge 2025/2026** at Turan University and received a **First-Degree Diploma**.

## Author

**Rustem Altyn**

Diploma project for the educational program  
**6B06101 — Information Systems**
