from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

# Load the model safely
try:
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
except Exception as e:
    print(f"Model loading failed: {e}")
    model = None  # Safe fallback

# Define a function to calculate similarity
def calculate_similarity(text1, text2):
    emb1 = model.encode([text1])[0]
    emb2 = model.encode([text2])[0]
    similarity = cosine_similarity([emb1], [emb2])[0][0]
    return float(similarity)

# Define the API route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Validate input
    if 'text1' not in data or 'text2' not in data:
        return jsonify({"error": "Missing 'text1' or 'text2' in request body"}), 400

    text1 = data['text1']
    text2 = data['text2']

    # If model is not loaded, return error
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    similarity_score = calculate_similarity(text1, text2)

    # Return response as per DataNeuron format
    return jsonify({"similarity score": similarity_score})

# Note: No if __name__ == "__main__" block needed for Vercel
