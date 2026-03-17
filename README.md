# 🎬 Movie Recommender System

<div align="center">

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-yellow?logo=huggingface)](https://priyanshuu2008-movie-recommender.hf.space)
![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-2.x-black?logo=flask)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-ML-orange?logo=scikit-learn)
![TMDb](https://img.shields.io/badge/TMDb-API-01b4e4?logo=themoviedb)
![License](https://img.shields.io/badge/License-MIT-green)

**A Machine Learning powered Movie Recommendation System with Flask backend & TMDb API integration**

🚀 **[Live Demo](https://priyanshuu2008-movie-recommender.hf.space)**

</div>

---

## 🧠 About the Project

This project is a **full-stack movie recommendation web app** built with a focus on **Machine Learning algorithms** for the backend recommendation engine.

> 💡 **ML & Python backend built by me** | Frontend UI assisted with Claude AI

The app combines two ML techniques — **Content-Based Filtering** and **Collaborative Filtering (SVD)** — to deliver personalised movie recommendations based on genre similarity and user behaviour patterns.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 Content-Based Filtering | TF-IDF + Cosine Similarity on movie genres |
| 👥 Collaborative Filtering | Matrix Factorization using SVD (scipy) |
| 🎯 Personalised Recommendations | Rate movies → ML predicts what you'll love next |
| 🔥 Trending & Popular | Live data from TMDb API |
| 🔍 Smart Search | Real-time autocomplete via TMDb |
| 🎬 Trailer Playback | YouTube trailer integration |
| 🔖 Watchlist | Save movies for later |
| ⭐ Rating System | Rate movies, stored in SQLite |
| 👤 Auth System | Sign up / Login (localStorage) |

---

## 🛠️ Tech Stack

### 🐍 Backend & ML (Built by me)
| Category | Tools |
|---|---|
| **Language** | Python 3.9+ |
| **Web Framework** | Flask |
| **ML — Content Based** | TF-IDF Vectorizer + Cosine Similarity (Scikit-learn) |
| **ML — Collaborative** | SVD Matrix Factorization (SciPy sparse) |
| **Data Processing** | Pandas, NumPy |
| **Database** | SQLite3 |
| **API Integration** | TMDb REST API, ThreadPoolExecutor (async fetching) |
| **Deployment** | Hugging Face Spaces (Docker) |

### 🎨 Frontend (AI Assisted)
| Category | Tools |
|---|---|
| **Languages** | HTML, CSS, JavaScript |
| **UI Style** | Custom CSS (no framework) |
| **Charts/UX** | Vanilla JS, CSS animations |

---

## 🤖 ML Architecture
```
User Input (Movie Title / Ratings)
        │
        ├── Content-Based Filtering
        │     ├── TF-IDF on movie genres
        │     └── Cosine Similarity Matrix
        │
        ├── Collaborative Filtering
        │     ├── User-Item Rating Matrix
        │     └── SVD (k=50 latent factors)
        │
        └── Hybrid Results → TMDb enrichment → User
```

### How it works:
- **Content-Based:** TF-IDF vectorizes movie genres → cosine similarity finds top-5 similar movies
- **Collaborative:** User-Item matrix decomposed via SVD → predicts ratings for unseen movies
- **Hybrid:** Both results merged, duplicates removed, enriched with TMDb poster/ratings

---

## 📁 Dataset

- **Source:** [MovieLens Small Dataset](https://grouplens.org/datasets/movielens/latest/)
- **Movies:** 9,742 movies
- **Ratings:** 100,836 ratings from 610 users
- **Files used:** `movies.csv`, `ratings.csv`

---

## 📂 Project Structure
```
Movie-Recommender/
├── app_flask.py              # Flask backend + ML logic
├── recommender.py            # ML recommendation engine
├── templates/
│   └── index.html            # Frontend (AI assisted)
├── ml-latest-small/
│   ├── movies.csv
│   └── ratings.csv
├── Dockerfile                # HuggingFace deployment
└── requirements.txt
```

---

## 🚀 Run Locally
```bash
git clone https://github.com/Priyanshuu2008/Movie-Recommender-System-PY-ML-HTML.git
cd Movie-Recommender-System-PY-ML-HTML
pip install -r requirements.txt
python app_flask.py
```

App opens at `http://localhost:7860` 🎉

---

## 👨‍💻 Author

**Priyanshu Tiwari**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/priyanshuu20)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/Priyanshuu2008)

> Aspiring Data Scientist | Gen AI Enthusiast | Python | ML | Deep Learning

---

⭐ **If you found this useful, please give it a star!**
