"""
========================================================
  Task 3 — Combat Online Plagiarism with AI
  Method 2: Flask Web Application
========================================================
Run:  python app.py
Open: http://127.0.0.1:5000
========================================================
"""

import os
import re
import itertools
import string
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename

# ── Optional libraries ──────────────────────────────
try:
    from rapidfuzz import fuzz as rfuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# ── Flask Setup ─────────────────────────────────────
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max
ALLOWED_EXTENSIONS = {'txt'}

# ── Thresholds ───────────────────────────────────────
THRESHOLD_HIGH   = 0.80
THRESHOLD_MEDIUM = 0.50
THRESHOLD_LOW    = 0.30

BUILTIN_STOPWORDS = {
    'i','me','my','we','our','you','your','he','him','his','she','her',
    'it','its','they','them','their','what','which','who','this','that',
    'these','those','am','is','are','was','were','be','been','being',
    'have','has','had','do','does','did','a','an','the','and','but',
    'if','or','as','of','at','by','for','with','in','out','on','off',
    'to','from','up','down','so','than','too','very','just','not','no',
    'nor','only','same','also','coz','kinda','gonna','gotta'
}

# ═══════════════════════════════════════════════════
#  NLP FUNCTIONS
# ═══════════════════════════════════════════════════

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english')) if NLTK_AVAILABLE else BUILTIN_STOPWORDS
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    if NLTK_AVAILABLE:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]
    else:
        tokens = [simple_stem(t) for t in tokens]
    return ' '.join(tokens)

def simple_stem(word):
    for suffix in ['ing','tion','ness','ment','ly','ed','er','es','s']:
        if word.endswith(suffix) and len(word) - len(suffix) > 3:
            return word[:-len(suffix)]
    return word

def get_ngrams(text, n=2):
    words = text.split()
    return set(' '.join(words[i:i+n]) for i in range(len(words)-n+1))

def ngram_similarity(t1, t2, n=2):
    ng1, ng2 = get_ngrams(t1, n), get_ngrams(t2, n)
    if not ng1 or not ng2:
        return 0.0
    return len(ng1 & ng2) / len(ng1 | ng2)

def fuzzy_sim(t1, t2):
    if RAPIDFUZZ_AVAILABLE:
        return rfuzz.token_sort_ratio(t1, t2) / 100.0
    if not t1 or not t2:
        return 0.0
    longer = max(len(t1), len(t2))
    return sum(c1 == c2 for c1, c2 in zip(t1[:longer], t2[:longer])) / longer

def tfidf_scores(corpus):
    if len(corpus) < 2:
        return {}
    vec = TfidfVectorizer(analyzer='word', ngram_range=(1,2),
                          min_df=1, sublinear_tf=True)
    mat = vec.fit_transform(corpus)
    sim = cosine_similarity(mat)
    return {(i,j): float(sim[i,j])
            for i in range(len(corpus))
            for j in range(i+1, len(corpus))}

def classify(score):
    if score >= THRESHOLD_HIGH:
        return "HIGH", "Likely Plagiarism", "#e74c3c"
    elif score >= THRESHOLD_MEDIUM:
        return "MEDIUM", "Suspicious", "#e67e22"
    elif score >= THRESHOLD_LOW:
        return "LOW", "Slight Similarity", "#f0b429"
    return "CLEAR", "No Significant Match", "#27ae60"

def run_check(docs):
    """docs = {name: text}. Returns list of result dicts."""
    names   = list(docs.keys())
    texts   = list(docs.values())
    normed  = [normalize_text(t) for t in texts]
    tf_map  = tfidf_scores(normed)
    results = []

    for i, j in itertools.combinations(range(len(names)), 2):
        tf  = tf_map.get((i,j), 0.0)
        ng  = ngram_similarity(normed[i], normed[j])
        fz  = fuzzy_sim(texts[i], texts[j])
        comp = 0.50*tf + 0.30*ng + 0.20*fz
        level, label, color = classify(comp)

        # highlighted common words
        words_a = set(normed[i].split())
        words_b = set(normed[j].split())
        common  = words_a & words_b

        results.append({
            'doc_a':     names[i],
            'doc_b':     names[j],
            'tfidf':     round(tf   * 100, 1),
            'ngram':     round(ng   * 100, 1),
            'fuzzy':     round(fz   * 100, 1),
            'composite': round(comp * 100, 1),
            'level':     level,
            'label':     label,
            'color':     color,
            'common_count': len(common),
            'snippet_a': texts[i][:250].replace('\n', ' '),
            'snippet_b': texts[j][:250].replace('\n', ' '),
        })

    results.sort(key=lambda x: x['composite'], reverse=True)
    return results

# ═══════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    docs = {}

    # ── Handle uploaded files ──
    if 'files' in request.files:
        files = request.files.getlist('files')
        for f in files:
            if f and f.filename and allowed_file(f.filename):
                name = secure_filename(f.filename)
                content = f.read().decode('utf-8', errors='ignore').strip()
                if content:
                    docs[name] = content

    # ── Handle pasted text ──
    paste_texts = request.form.getlist('paste_text[]')
    paste_names = request.form.getlist('paste_name[]')
    for name, text in zip(paste_names, paste_texts):
        name = name.strip() or f"document_{len(docs)+1}.txt"
        text = text.strip()
        if text:
            if not name.endswith('.txt'):
                name += '.txt'
            docs[name] = text

    if len(docs) < 2:
        return jsonify({'error': 'Please provide at least 2 documents to compare.'}), 400

    results  = run_check(docs)
    high     = sum(1 for r in results if r['level'] == 'HIGH')
    medium   = sum(1 for r in results if r['level'] == 'MEDIUM')
    low      = sum(1 for r in results if r['level'] == 'LOW')
    clear    = sum(1 for r in results if r['level'] == 'CLEAR')
    avg_score = round(sum(r['composite'] for r in results) / len(results), 1) if results else 0

    return jsonify({
        'results':   results,
        'summary': {
            'total':     len(results),
            'high':      high,
            'medium':    medium,
            'low':       low,
            'clear':     clear,
            'doc_count': len(docs),
            'avg_score': avg_score,
            'timestamp': datetime.now().strftime('%b %d, %Y · %H:%M'),
        }
    })

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    print("\n" + "="*55)
    print("  🔍  Plagiarism Checker Web App — Task 3")
    print("="*55)
    print("  ▶  Open in browser: http://127.0.0.1:5000")
    print("="*55 + "\n")
    app.run(debug=True, port=5000)
