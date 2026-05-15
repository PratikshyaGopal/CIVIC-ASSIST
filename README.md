# CIVIC-ASSIST
### Smart Civic Complaint Management & Public Service Platform

![Python](https://img.shields.io/badge/Python-3-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-black?style=for-the-badge&logo=flask)
![Firebase](https://img.shields.io/badge/Firebase-orange?style=for-the-badge&logo=firebase)
![HTML5](https://img.shields.io/badge/HTML5-orange?style=for-the-badge&logo=html5)
![CSS3](https://img.shields.io/badge/CSS3-blue?style=for-the-badge&logo=css3)
![JavaScript](https://img.shields.io/badge/JavaScript-yellow?style=for-the-badge&logo=javascript)

---

# 📌 Overview

CIVIC-ASSIST is a civic complaint management platform designed to help citizens report public issues digitally and allow administrators to efficiently manage complaints through a centralized dashboard.

The application supports complaint registration, complaint tracking, image uploads, admin management, and AI-assisted sentiment analysis using NLP techniques.

This project demonstrates practical implementation of:
- Full-stack Flask development
- Firebase integration
- Complaint management workflows
- Secure authentication systems
- AI & NLP integration
- Civic technology solutions

---

# 🚀 Key Features

- Citizen registration and login system
- Complaint submission with image uploads
- Complaint status tracking
- Admin dashboard for complaint management
- Complaint reports and analytics
- Firebase Realtime Database integration
- AI-powered sentiment and emotion analysis
- Secure session management and authentication
- Responsive web interface
- Deployment-ready architecture

---

# 🎯 Project Objective

The main objective of CIVIC-ASSIST is to simplify and digitize the process of managing civic complaints.

The system helps:
- Citizens report issues easily
- Administrators manage complaints efficiently
- Authorities monitor complaint records digitally

The project also integrates AI-based NLP techniques for analyzing user feedback and complaint sentiment.

---

# 🛠️ Tech Stack

## Backend
- Python
- Flask

## Frontend
- HTML5
- CSS3
- JavaScript

## Database & Cloud
- Firebase Realtime Database
- Firebase Admin SDK

## AI & NLP
- TF-IDF
- NLP Embeddings
- Emotion Detection

## Deployment
- Vercel

---

# 📂 Project Structure

```text
CIVIC-ASSIST/
│
├── instance/
│   └── civic_assist.db
│
├── static/
│   ├── css/
│   │   └── style.css
│   │
│   ├── js/
│   │
│   ├── uploads/
│   │   └── uploaded complaint images
│   │
│   └── images/
│
├── templates/
│   ├── admin_complaints.html
│   ├── admin_dashboard.html
│   ├── admin_login.html
│   ├── admin_registration.html
│   ├── admin_reports.html
│   ├── admin_settings.html
│   ├── admin_users.html
│   ├── admin_workers.html
│   ├── citizen_dashboard.html
│   ├── citizen_login.html
│   ├── citizen_register.html
│   ├── complaint_detail.html
│   ├── index.html
│   └── additional HTML templates
│
├── .gitignore
├── app.py
├── emotion_nlp_camera.py
├── emotion_nlp_embedding.py
├── emotion_nlp_tfidf.py
├── requirements.txt
├── vercel.json
└── README.md
```

---

# 🧠 AI & NLP Features

The project integrates NLP-based sentiment and emotion analysis modules to improve complaint understanding and user feedback evaluation.

### NLP Functionalities
- Text preprocessing
- TF-IDF vectorization
- Word embedding analysis
- Sentiment classification
- Emotion prediction

These features help analyze complaint text and improve future complaint prioritization systems.

---

# 🔐 Security Features

- Secure authentication system
- Session management
- Password hashing
- Protected admin access
- Secure image upload handling

---

# ⚙️ Installation & Setup

## Prerequisites

- Python 3.10+
- Git
- Firebase Project Configuration

---

## Clone Repository

```bash
git clone https://github.com/bpriyanka-rao/CIVIC-ASSIST.git
cd CIVIC-ASSIST
```

---

## Create Virtual Environment

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configure Environment Variables

Create a `.env` file:

```env
FLASK_APP=app.py
FLASK_ENV=development
FLASK_SECRET_KEY=your_secret_key
```

Optional Firebase variables:

```env
FIREBASE_WEB_CONFIG_JSON=
FIREBASE_SERVICE_ACCOUNT_JSON=
FIREBASE_SERVICE_ACCOUNT_PATH=
```

---

## Run the Application

```bash
flask run
```

Open in browser:

```text
http://127.0.0.1:5000
```

---

# ☁️ Deployment

The project supports deployment using:
- Vercel
- Render
- Railway
- Heroku

Deployment configuration is included using `vercel.json`.

---

# 📈 Future Enhancements

- Real-time notifications
- AI-powered complaint categorization
- Geo-location based complaint tracking
- Mobile application support
- Advanced analytics dashboard
- Machine learning based complaint prioritization

---

# 📘 License

This project is developed for educational and demonstration purposes.

---

# 📧 Contact

**B. Priyanka**  
Final Year B.Tech Student  
AI & Full Stack Development Enthusiast