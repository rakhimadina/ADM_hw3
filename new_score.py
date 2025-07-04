from search_engine import preprocess_text
from search_engine import get_tfIdf
from search_engine import get_vocabulary
from search_engine import get_inverted_index
from search_engine import execute_conjunctive_query
from search_engine import cosine_similarity
from collections import defaultdict
import numpy as np
import heapq


def compute_custom_new_score(
    doc_id,
    description_scores,
    df,
    user_cuisines,
    user_facilities,
    user_price,
    max_description_score
):
    #initialize individual scores
    description_score = description_scores.get(doc_id, 0) / max_description_score  #normalize to keep it between [0,1]
    cuisine_score = 0
    facilities_score = 0
    price_score = 0

    #get restaurant data
    restaurant = df.iloc[doc_id]

    #cuisine Match
    restaurant_cuisines = [c.strip().lower() for c in restaurant['cuisineType'].split(',')]
    if any(cuisine in restaurant_cuisines for cuisine in user_cuisines):
        cuisine_score = 1

    #servicies Match
    restaurant_facilities = [f.strip().lower() for f in restaurant['facilitiesServices']]
    matching_facilities = set(user_facilities) & set(restaurant_facilities)
    facilities_score = len(matching_facilities) / len(user_facilities) if user_facilities else 0

    #price Match
    if restaurant['priceRange'] == user_price:
        price_score = 1

    #weights
    weights = {
        'description': 0.35,
        'cuisine': 0.25,
        'facilities': 0.25,
        'price': 0.15
    }

    #total score
    total_score = (
        weights['description'] * description_score +
        weights['cuisine'] * cuisine_score +
        weights['facilities'] * facilities_score +
        weights['price'] * price_score
    )

    return total_score



def get_top_k_restaurants(
    ranked_docs,
    df,
    user_cuisines,
    user_facilities,
    user_price,
    k=10
):
    heap = []
    max_description_score = max(score for (_, score) in ranked_docs) if ranked_docs else 1

    for doc_id, description_score in ranked_docs:
        # Check if doc_id is within the bounds of df
        if doc_id < 0 or doc_id >= len(df):
            print(f"Warning: Document ID {doc_id} is out of bounds for DataFrame 'df'. Skipping this document.")
            continue
        total_score = compute_custom_new_score(
            doc_id,
            dict(ranked_docs),
            df,
            user_cuisines,
            user_facilities,
            user_price,
            max_description_score
        )

        #maintain a heap of size k
        if len(heap) < k:
            heapq.heappush(heap, (total_score, doc_id))
        else:
            heapq.heappushpop(heap, (total_score, doc_id))

    #extract restaurants from heap and sort by score descending
    top_restaurants = sorted(heap, key=lambda x: x[0], reverse=True)
    return top_restaurants

def execute_ranked_query1(query_terms, inverted_index, vocabulary_df, processed_texts, top_k):
    """
    Executes a ranked query by calculating cosine similarity between a query vector (TF-IDF)
    and document vectors, using only the terms from the query that, once processed, exist in the vocabulary.

    Parameters:
    - query_terms (str): Query input as a space-separated string of terms.
    - inverted_index (dict): Dictionary with term IDs as keys and values as lists of tuples (document ID, TF-IDF score),
                             representing the inverted index for documents.
    - vocabulary_df (DataFrame): DataFrame of vocabulary terms, each with a unique term ID.
    - processed_texts (list of list of str): List of processed texts, each represented as a list of terms.
    - top_k (int): Number of top-ranked documents to return based on similarity.

    Returns:
    - ranked_results (list): List of tuples, each containing a document ID and its similarity score.
    - not_found (str): Message listing terms from the query that were not found in the vocabulary.
    """
    
    # Tokenize and clean query terms
    query_list = preprocess_text([query_terms])[0]
    
    # Filter out terms that are not in the vocabulary and store those not found
    no_matches = [term for term in query_list if term not in vocabulary_df['term'].values]
    query_list = [term for term in query_list if term in vocabulary_df['term'].values]

    # If any query terms were not found, create a message with those terms
    not_found = "No matches found for these terms: " + ', '.join(list(set(no_matches))) if no_matches else ""

    # Map query terms to their corresponding term IDs from the vocabulary
    query_term_ids = (vocabulary_df[vocabulary_df['term'].isin(query_list)]).set_index('term').loc[query_list].reset_index()['term_id'].astype(int).tolist()

    # Initialize the query vector, setting TF-IDF values for query terms
    query_vector = np.zeros(vocabulary_df.shape[0])
    for i in range(len(query_term_ids)):
        query_vector[query_term_ids[i]] = get_tfIdf(query_list[i], query_list, processed_texts)
        
    # Initialize document vectors with default zero values for each term
    document_vectors = defaultdict(lambda: np.zeros(vocabulary_df.shape[0]))
    
    # Populate document vectors with TF-IDF scores from the inverted index
    for term_id in vocabulary_df['term_id']:
        if term_id in inverted_index:
            for doc_id, tfidf_score in inverted_index[term_id]:
                document_vectors[doc_id][term_id] = tfidf_score
    
    # Compute cosine similarity for each document vector against the query vector
    scores = {doc_id: cosine_similarity(doc_vector, query_vector) for doc_id, doc_vector in document_vectors.items()}
    
    # Rank documents by similarity scores, in descending order
    ranked_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    if top_k is not None and len(ranked_results) > top_k:
        # Limit the results to the top_k documents
        ranked_results = ranked_results[:top_k]

    return ranked_results, not_found