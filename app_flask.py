import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import requests
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix

app = Flask(__name__, template_folder='templates')
TMDB_API_KEY = "e7a6fccb69ec738cca12a17cb8a35344"

def init_db():
    conn = sqlite3.connect('movie_app.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS user_ratings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        movie_title TEXT,
        rating INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS watchlist (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        movie_title TEXT,
        poster TEXT,
        year TEXT,
        added_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()
    conn.close()

init_db()

print("Loading data and building models...")

movies  = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')

tfidf        = TfidfVectorizer(stop_words='english')
movies['genres'] = movies['genres'].fillna('')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim   = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices      = pd.Series(movies.index, index=movies['title']).drop_duplicates()

user_item    = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
matrix       = csr_matrix(user_item.values)
U, sigma, Vt = svds(matrix, k=50)
predicted    = np.dot(np.dot(U, np.diag(sigma)), Vt)
preds_df     = pd.DataFrame(predicted, index=user_item.index, columns=user_item.columns)

print("Models ready!")

def get_tmdb_details(title):
    try:
        year = None
        if '(' in title and ')' in title:
            try:
                year = int(title.strip()[-5:-1])
            except:
                pass
        name = title.split('(')[0].strip()
        if ', The' in name:
            name = 'The ' + name.replace(', The', '').strip()

        url  = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={name}"
        if year:
            url += f"&year={year}"
        data = requests.get(url, timeout=5).json()
        if not data.get('results') and year:
            data = requests.get(
                f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={name}",
                timeout=5).json()
        if data.get('results'):
            m = data['results'][0]
            genre_url  = f"https://api.themoviedb.org/3/movie/{m['id']}?api_key={TMDB_API_KEY}"
            genre_data = requests.get(genre_url, timeout=5).json()
            genres     = [g['name'] for g in genre_data.get('genres', [])][:3]
            return {
                'poster'  : f"https://image.tmdb.org/t/p/w342{m['poster_path']}" if m.get('poster_path') else None,
                'overview': m.get('overview', 'No description available.'),
                'rating'  : round(m.get('vote_average', 0), 1),
                'year'    : m.get('release_date', 'N/A')[:4] if m.get('release_date') else 'N/A',
                'votes'   : m.get('vote_count', 0),
                'genres'  : genres
            }
    except:
        pass
    return {'poster': None, 'overview': 'N/A', 'rating': 0, 'year': 'N/A', 'votes': 0, 'genres': []}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/suggest')
def suggest():
    query = request.args.get('q', '')
    if len(query) < 2:
        return jsonify([])
    try:
        url  = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={query}"
        data = requests.get(url, timeout=3).json()
        suggestions = []
        for m in data.get('results', [])[:6]:
            suggestions.append({
                'title' : m['title'],
                'year'  : m.get('release_date', '')[:4] if m.get('release_date') else '',
                'poster': f"https://image.tmdb.org/t/p/w92{m['poster_path']}" if m.get('poster_path') else None
            })
        return jsonify(suggestions)
    except:
        return jsonify([])

@app.route('/search')
def search():
    query = request.args.get('q', '')
    if not query:
        return jsonify([])
    try:
        url  = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={query}&page=1"
        data = requests.get(url, timeout=5).json()
        results = []
        for m in data.get('results', [])[:12]:
            results.append({
                'title'   : m.get('title', ''),
                'poster'  : f"https://image.tmdb.org/t/p/w342{m['poster_path']}" if m.get('poster_path') else None,
                'overview': m.get('overview', 'No description.')[:200],
                'rating'  : round(m.get('vote_average', 0), 1),
                'year'    : m.get('release_date', '')[:4] if m.get('release_date') else 'N/A',
                'votes'   : m.get('vote_count', 0),
                'genres'  : []
            })
        return jsonify(results)
    except:
        return jsonify([])

@app.route('/recommend')
def recommend():
    title   = request.args.get('title', '')
    user_id = int(request.args.get('user_id', 1))

    idx = indices.get(title)
    if idx is None:
        return jsonify({'error': 'Movie not found!'})

    sim_scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:6]
    content    = [movies['title'].iloc[i[0]] for i in sim_scores]

    collab = []
    if user_id in preds_df.index:
        user_row      = preds_df.loc[user_id]
        already_rated = user_item.loc[user_id][user_item.loc[user_id] > 0].index
        user_row      = user_row.drop(index=already_rated, errors='ignore')
        top_ids       = user_row.nlargest(5).index
        collab        = movies[movies['movieId'].isin(top_ids)]['title'].tolist()

    combined = list(dict.fromkeys(content + collab))[:10]

    source_map = {}
    for t in content:
        source_map[t] = 'content'
    for t in collab:
        if t not in source_map:
            source_map[t] = 'collab'

    with ThreadPoolExecutor(max_workers=10) as executor:
        details_list = list(executor.map(get_tmdb_details, combined))

    results = []
    for title_name, details in zip(combined, details_list):
        source = source_map.get(title_name, 'content')
        reason = f"Similar genre to '{title}'" if source == 'content' else "Users like you also loved this"
        results.append({
            'title'   : title_name,
            'poster'  : details['poster'],
            'overview': details['overview'][:200],
            'rating'  : details['rating'],
            'year'    : details['year'],
            'votes'   : details['votes'],
            'genres'  : details['genres'],
            'reason'  : reason,
            'source'  : source
        })
    return jsonify(results)

@app.route("/movies")
def get_movies():
    return jsonify(movies['title'].tolist())

@app.route('/rate', methods=['POST'])
def rate_movie():
    data        = request.json
    user_id     = data.get('user_id', 1)
    movie_title = data.get('movie_title', '')
    rating      = data.get('rating', 0)
    if not movie_title or not (1 <= rating <= 5):
        return jsonify({'error': 'Invalid data'}), 400
    conn = sqlite3.connect('movie_app.db')
    c    = conn.cursor()
    c.execute('SELECT id FROM user_ratings WHERE user_id=? AND movie_title=?', (user_id, movie_title))
    existing = c.fetchone()
    if existing:
        c.execute('UPDATE user_ratings SET rating=? WHERE id=?', (rating, existing[0]))
    else:
        c.execute('INSERT INTO user_ratings (user_id, movie_title, rating) VALUES (?,?,?)',
                  (user_id, movie_title, rating))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': f'Rated {rating}⭐ for {movie_title}'})

@app.route('/my_ratings')
def my_ratings():
    user_id = int(request.args.get('user_id', 1))
    conn    = sqlite3.connect('movie_app.db')
    c       = conn.cursor()
    c.execute('SELECT movie_title, rating, timestamp FROM user_ratings WHERE user_id=? ORDER BY timestamp DESC',
              (user_id,))
    rows = c.fetchall()
    conn.close()
    return jsonify([{'title': r[0], 'rating': r[1], 'timestamp': r[2]} for r in rows])

@app.route('/watchlist/add', methods=['POST'])
def add_watchlist():
    data    = request.json
    user_id = data.get('user_id', 1)
    title   = data.get('title', '')
    poster  = data.get('poster', '')
    year    = data.get('year', '')
    if not title:
        return jsonify({'error': 'No title'}), 400
    conn = sqlite3.connect('movie_app.db')
    c    = conn.cursor()
    c.execute('SELECT id FROM watchlist WHERE user_id=? AND movie_title=?', (user_id, title))
    if c.fetchone():
        conn.close()
        return jsonify({'message': 'Already in watchlist'})
    c.execute('INSERT INTO watchlist (user_id, movie_title, poster, year) VALUES (?,?,?,?)',
              (user_id, title, poster, year))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': f'Added to watchlist!'})

@app.route('/watchlist/remove', methods=['POST'])
def remove_watchlist():
    data    = request.json
    user_id = data.get('user_id', 1)
    title   = data.get('title', '')
    conn    = sqlite3.connect('movie_app.db')
    c       = conn.cursor()
    c.execute('DELETE FROM watchlist WHERE user_id=? AND movie_title=?', (user_id, title))
    conn.commit()
    conn.close()
    return jsonify({'success': True})

@app.route('/watchlist')
def get_watchlist():
    user_id = int(request.args.get('user_id', 1))
    conn    = sqlite3.connect('movie_app.db')
    c       = conn.cursor()
    c.execute('SELECT movie_title, poster, year, added_at FROM watchlist WHERE user_id=? ORDER BY added_at DESC',
              (user_id,))
    rows = c.fetchall()
    conn.close()
    return jsonify([{'title': r[0], 'poster': r[1], 'year': r[2], 'added_at': r[3]} for r in rows])

@app.route('/trending')
def trending():
    try:
        url = f"https://api.themoviedb.org/3/trending/movie/day?api_key={TMDB_API_KEY}"
        data = requests.get(url, timeout=5).json()
        results = []
        for m in data.get('results', [])[:12]:
            results.append({'title': m.get('title',''), 'poster': f"https://image.tmdb.org/t/p/w342{m['poster_path']}" if m.get('poster_path') else None, 'overview': m.get('overview','')[:200], 'rating': round(m.get('vote_average',0),1), 'year': m.get('release_date','')[:4] if m.get('release_date') else 'N/A', 'votes': m.get('vote_count',0), 'genres': []})
        return jsonify(results)
    except:
        return jsonify([])

@app.route('/popular')
def popular():
    try:
        url = f"https://api.themoviedb.org/3/movie/popular?api_key={TMDB_API_KEY}"
        data = requests.get(url, timeout=5).json()
        results = []
        for m in data.get('results', [])[:12]:
            results.append({'title': m.get('title',''), 'poster': f"https://image.tmdb.org/t/p/w342{m['poster_path']}" if m.get('poster_path') else None, 'overview': m.get('overview','')[:200], 'rating': round(m.get('vote_average',0),1), 'year': m.get('release_date','')[:4] if m.get('release_date') else 'N/A', 'votes': m.get('vote_count',0), 'genres': []})
        return jsonify(results)
    except:
        return jsonify([])

@app.route('/recommend_by_ratings', methods=['POST'])
def recommend_by_ratings():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No ratings provided'})

        matched = []
        for title, rating in data.items():
            match = movies[movies['title'].str.contains(title.split('(')[0].strip(), case=False, na=False)]
            if not match.empty:
                matched.append((match.iloc[0]['movieId'], float(rating)))

        if not matched:
            return jsonify({'error': 'None of the rated movies found in dataset'})

        new_user = np.zeros(user_item.shape[1])
        for movie_id, rating in matched:
            if movie_id in user_item.columns:
                col_idx = list(user_item.columns).index(movie_id)
                new_user[col_idx] = rating

        sigma_inv = np.diag(1.0 / sigma)
        new_user_latent = new_user @ Vt.T @ sigma_inv
        predicted_scores = new_user_latent @ Vt

        rated_ids = [m[0] for m in matched]
        movie_scores = []
        for i, col in enumerate(user_item.columns):
            if col not in rated_ids:
                movie_scores.append((col, predicted_scores[i]))

        movie_scores.sort(key=lambda x: x[1], reverse=True)
        top_ids = [m[0] for m in movie_scores[:20]]

        top_movies = movies[movies['movieId'].isin(top_ids)]['title'].tolist()

        results = []
        with ThreadPoolExecutor(max_workers=6) as ex:
            futures = {ex.submit(get_tmdb_details, t): t for t in top_movies[:12]}
            for future, title in futures.items():
                try:
                    d = future.result()
                    results.append({
                        'title': title,
                        'poster': d['poster'],
                        'overview': d['overview'],
                        'rating': d['rating'],
                        'year': d['year'],
                        'votes': d['votes'],
                        'genres': d['genres'],
                        'reason': '⭐ Based on your ratings'
                    })
                except:
                    pass

        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/trailer')
def trailer():
    title = request.args.get('title', '')
    try:
        clean = title.split('(')[0].strip()
        search = requests.get(
            f"https://api.themoviedb.org/3/search/movie",
            params={'api_key': TMDB_API_KEY, 'query': clean},
            timeout=5
        ).json()
        results = search.get('results', [])
        if not results:
            return jsonify({'key': None})
        movie_id = results[0]['id']
        videos = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}/videos",
            params={'api_key': TMDB_API_KEY},
            timeout=5
        ).json()
        for v in videos.get('results', []):
            if v['site'] == 'YouTube' and v['type'] in ['Trailer', 'Teaser']:
                return jsonify({'key': v['key']})
        return jsonify({'key': None})
    except:
        return jsonify({'key': None})

# ── THIS MUST BE AT THE END ──
if __name__ == '__main__':
   app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))