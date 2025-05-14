"""
from sentence_transformers import SentenceTransformer 
import numpy as np 
from numpy import dot
from numpy.linalg import norm
"""

EURO_VOC_DOMAINS = [
    "STATISTICS",
    "ENERGY",
    "POLITICS",
    "INTERNATIONAL RELATIONS",
    "LAW",
    "ECONOMICS",
    "TRADE",
    "FINANCE",
    "SOCIAL QUESTIONS",
    "EDUCATION AND COMMUNICATIONS",
    "SCIENCE",
    "BUSINESS AND COMPETITION",
    "EMPLOYMENT AND WORKING CONDITIONS",
    "TRANSPORT",
    "ENVIRONMENT"
]

"""
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
domain_embeddings = model.encode(EURO_VOC_DOMAINS, normalize_embeddings = True)

segment = "amending the representative prices and additional import duties for certain products in the sugar sector fixed by Regulation (EC) No 877/2009 for the 2009/10 marketing year"

segment_embedding = model.encode(segment, normalize_embeddings=True)

similarities = domain_embeddings @ segment_embedding.T

best_index = np.argmax(similarities)
best_domain = EURO_VOC_DOMAINS[best_index]

print("Best matching domain:", best_domain)

"""
