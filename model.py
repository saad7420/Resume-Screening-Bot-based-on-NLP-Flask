from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def rank_resumes(job_desc, resume_texts):
    if not job_desc.strip():
        raise ValueError("Job description is empty.")
    if not resume_texts:
        raise ValueError("Resume list is empty.")

    documents = [job_desc] + resume_texts
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(documents)

    job_vector = vectors[0]
    resume_vectors = vectors[1:]

    scores = cosine_similarity(job_vector, resume_vectors).flatten()
    return scores.tolist()
