import argparse
import sys
from typing import List, Union, Tuple
import os
import hashlib

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

import pg8000
from pgvector.pg8000 import register_vector

class BGEM3Encoder:
    def __init__(self, model_path: str = "BAAI/bge-m3", local_files_only: bool = False):
        """
        Initialize BGE-M3 encoder.
        
        Args:
            model_path: Path to the model (HuggingFace model ID or local path)
            local_files_only: If True, only use local files and disable online checks
        """
        # Load tokenizer and model with local_files_only flag to disable online checks
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=local_files_only,
            trust_remote_code=True
        )
        
        self.model = AutoModel.from_pretrained(
            model_path,
            local_files_only=local_files_only,
            trust_remote_code=True
        )
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def encode(self, texts: Union[str, List[str]], 
               batch_size: int = 32,
               max_length: int = 8192,
               normalize: bool = True) -> np.ndarray:
        """
        Encode texts into vectors.
        
        Args:
            texts: Single text or list of texts to encode
            batch_size: Batch size for encoding
            max_length: Maximum sequence length
            normalize: Whether to normalize the embeddings
            
        Returns:
            numpy array of embeddings
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                # BGE models use CLS pooling by default
                embeddings = model_output.last_hidden_state[:, 0]
            
            # Normalize if requested
            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu().numpy())
        
        # Concatenate all batches
        return np.vstack(all_embeddings)
    
    def compute_similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between two sets of embeddings.
        """
        # If embeddings are already normalized, this is just a dot product
        return np.dot(embeddings1, embeddings2.T)


def string2vector(text):
   encoder = BGEM3Encoder(model_path="/mnt/autofs/home/jl899795/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181", local_files_only=True)
   return encoder.encode(text)[0]

def get_db_connection(args):
    """Create and return a database connection"""
    conn = pg8000.connect(
        host=args.host,
        port=args.port,
        database=args.db,
        user=args.user,
        password=os.getenv("PG_PASSWORD", "")
    )
    register_vector(conn)
    return conn

################################################################
# set
def create_tables_if_not_exist(conn):
    """Create necessary tables and indexes if they don't exist"""
    cursor = conn.cursor()
    
    # Enable pgvector extension
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
    
    # Create content table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS content (
            id SERIAL PRIMARY KEY,
            filepath VARCHAR(1024) UNIQUE NOT NULL
        )
    """)
    
    # Create question table with vector column
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS question (
            id SERIAL PRIMARY KEY,
            question VARCHAR(2048) UNIQUE NOT NULL,
            qv vector(1024)  -- Adjust dimension based on your string2vector output
        )
    """)
    
    # Create many-to-many relationship table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS content_question (
            cid INTEGER REFERENCES content(id) ON DELETE CASCADE,
            qid INTEGER REFERENCES question(id) ON DELETE CASCADE,
            PRIMARY KEY (cid, qid)
        )
    """)
    
    # Create HNSW index for vector similarity search
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS question_qv_hnsw_idx 
        ON question 
        USING hnsw (qv vector_cosine_ops)
    """)
    
    conn.commit()

def insert_content(conn, filepath: str) -> int:
    """Insert content record and return its ID"""
    cursor = conn.cursor()
    
    # Check if content already exists
    cursor.execute("SELECT id FROM content WHERE filepath = %s", (filepath,))
    result = cursor.fetchone()
    
    if result:
        return result[0]
    
    # Insert new content
    cursor.execute(
        "INSERT INTO content (filepath) VALUES (%s) RETURNING id",
        (filepath,)
    )
    content_id = cursor.fetchone()[0]
    conn.commit()
    
    return content_id

def insert_question(conn, question_text: str) -> int:
    """Insert question with its vector and return its ID"""
    cursor = conn.cursor()
    
    # Check if question already exists
    cursor.execute("SELECT id FROM question WHERE question = %s", (question_text,))
    result = cursor.fetchone()
    
    if result:
        return result[0]
    
    # Generate vector for the question
    question_vector = string2vector(question_text)
    
    # Insert new question with vector
    cursor.execute(
        "INSERT INTO question (question, qv) VALUES (%s, %s) RETURNING id",
        (question_text, question_vector)
    )
    question_id = cursor.fetchone()[0]
    conn.commit()
    
    return question_id

def link_content_question(conn, content_id: int, question_id: int):
    """Create relationship between content and question"""
    cursor = conn.cursor()
    
    # Check if relationship already exists
    cursor.execute(
        "SELECT 1 FROM content_question WHERE cid = %s AND qid = %s",
        (content_id, question_id)
    )
    
    if not cursor.fetchone():
        cursor.execute(
            "INSERT INTO content_question (cid, qid) VALUES (%s, %s)",
            (content_id, question_id)
        )
        conn.commit()

# do insert
def do_set(args):
    filepath = args.filepath
    questions = args.questions
    conn = None
    try:
        # Connect to database
        conn = get_db_connection(args)
        
        # Create tables if needed
        create_tables_if_not_exist(conn)
        
        # Insert content
        content_id = insert_content(conn, filepath)
        print(f"Content inserted/found with ID: {content_id}")
        
        # Insert questions and create relationships
        for question_text in questions:
            question_id = insert_question(conn, question_text)
            link_content_question(conn, content_id, question_id)
            print(f"Question '{question_text}' linked to content")
        
        print("All operations completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        if conn:
            conn.rollback()
        sys.exit(1)
    finally:
        if conn:
            conn.close()
################################################################

################################################################
# do search
def search_similar_questions(conn, query_vector: np.ndarray, limit: int = 10) -> List[Tuple[str, int, float]]:
    """
    Search for similar questions using vector similarity
    Returns list of (filepath, question_id, similarity_score)
    """
    cursor = conn.cursor()
    
    # Perform vector similarity search with cosine distance
    # Note: cosine distance returns values between 0 and 2, where 0 is most similar
    # We'll convert to similarity score (1 - distance/2) for better interpretation
    cursor.execute("""
        SELECT DISTINCT c.filepath, q.id, q.question,
               1 - (q.qv <=> %s) as similarity_score
        FROM question q
        JOIN content_question cq ON q.id = cq.qid
        JOIN content c ON cq.cid = c.id
        ORDER BY similarity_score DESC
        LIMIT %s
    """, (query_vector, limit))
    
    results = cursor.fetchall()
    return results

def do_search(args):
    query_question = args.question
    conn = None
    try:
        # Connect to database
        conn = get_db_connection(args)
        
        # Generate vector for the query
        print(f"Searching for: {query_question}")
        query_vector = string2vector(query_question)
        
        # Search for similar questions
        results = search_similar_questions(conn, query_vector)
        
        if not results:
            print("No matching content found.")
        else:
            print("\nMatching content (ranked by similarity):")
            print("-" * 60)
            
            seen_filepaths = set()
            for filepath, qid, question, score in results:
                if filepath not in seen_filepaths:
                    seen_filepaths.add(filepath)
                    print(f"Score: {score:.4f} | File: {filepath}")
                    print(f"  Matched question: {question}")
                    print()
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        if conn:
            conn.close()
################################################################

def main():
    parser = argparse.ArgumentParser(
        prog="kg.py",
        description="Simple knowledge tool for vector search",
        epilog="File-Question"
    )
    subparsers = parser.add_subparsers(
        title="Available Commands",
        dest="command",  # This attribute will store the name of the chosen subcommand
        required=True,   # In Python 3.7+, this makes a subcommand mandatory
        help="Sub-command help"
    )
    parser.add_argument(
        "--host", "-H",
        type=str,
        default="127.0.0.1",
        help="The database host"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=5432,
        help="The database port"
    )
    parser.add_argument(
        "--user", "-U",
        type=str,
        default="postgres",
        help="The database username"
    )
    parser.add_argument(
        "--db", "-w",
        type=str,
        default="kg",
        help="The database db name"
    )
    parser_set = subparsers.add_parser(
        "set",
        help="Update related questions for a given knowledge piece",
        description="Update related questions for a given knowledge piece"
    )
    parser_set.add_argument(
        "filepath",
        type=str,
        help="The filepath of the given knowledge piece"
    )
    parser_set.add_argument(
        "questions",
        type=str,
        nargs='+',
        help="The questions related to the given knowledge piece"
    )
    parser_set.set_defaults(func=do_set)
    parser_search = subparsers.add_parser(
        "search",
        help="Perform a simple search according to query question.",
        description="Performs a simple search according to query question"
    )
    parser_search.add_argument(
        "question",
        type=str,
        help="The query question"
    )
    parser_search.set_defaults(func=do_search)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
