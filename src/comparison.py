import numpy as np
from typing import List, Set
import mmh3  # MurmurHash3 for hashing

def sketch(sequence: bytes, kmer_size: int, sketch_size: int) -> Set[int]:
    """
    Generate k-mer hashes using FracMinHash sketching.
    
    Args:
        sequence: Byte representation of the DNA sequence
        kmer_size: Size of k-mers
        sketch_size: Size of the sketch
    
    Returns:
        Set of hash values representing the sketch
    """
    # Generate all k-mers and hash them
    all_hashes = []
    for i in range(len(sequence) - kmer_size + 1):
        kmer = sequence[i:i+kmer_size]
        hash_value = mmh3.hash(kmer, signed=False)  # Using MurmurHash3
        all_hashes.append(hash_value)
    
    # Sort hashes and take the smallest 'sketch_size' hashes
    all_hashes.sort()
    return set(all_hashes[:min(sketch_size, len(all_hashes))])

def encode_hash_hd(kmer_hashes: Set[int], hd_dim: int) -> np.ndarray:
    """
    Encode k-mer hashes into a hypervector.
    
    Args:
        kmer_hashes: Set of hash values
        hd_dim: Dimensionality of the hypervector
    
    Returns:
        Hypervector representation
    """
    # Initialize hypervector with zeros
    hypervector = np.zeros(hd_dim, dtype=np.float32)
    
    # For each hash, set specific dimensions to 1
    for hash_val in kmer_hashes:
        # Use the hash to determine positions in the hypervector
        np.random.seed(hash_val)
        positions = np.random.choice(hd_dim, size=int(hd_dim * 0.1), replace=False)
        hypervector[positions] = 1.0
    
    return hypervector

def encode_sequence_to_hypervector(sequence: str, kmer_size: int, sketch_size: int, hd_dim: int) -> np.ndarray:
    """
    Encode a DNA sequence into a hypervector.
    
    Args:
        sequence: DNA sequence string
        kmer_size: Size of k-mers
        sketch_size: Size of the sketch
        hd_dim: Dimensionality of the hypervector
    
    Returns:
        Hypervector representation of the sequence
    """
    # Generate k-mer hashes using FracMinHash sketching
    kmer_hashes = sketch(sequence.encode(), kmer_size, sketch_size)
    
    # Encode k-mer hashes into a hypervector
    return encode_hash_hd(kmer_hashes, hd_dim)

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
    
    Returns:
        Cosine similarity value
    """
    dot_product = np.dot(vec1, vec2)
    magnitude1 = np.sqrt(np.sum(vec1 * vec1))
    magnitude2 = np.sqrt(np.sum(vec2 * vec2))
    return dot_product / (magnitude1 * magnitude2)

def main():
    # Hardcoded DNA sequences
    sequence1 = "ACGTACGTACGTACGT"
    sequence2 = "ACGTACGTACGTACGA"
    
    # Parameters
    kmer_size = 4  # Size of k-mers
    sketch_size = 100  # Size of the sketch
    hd_dim = 10000  # Dimensionality of the hypervector
    
    # Encode sequences into hypervectors
    hypervector1 = encode_sequence_to_hypervector(sequence1, kmer_size, sketch_size, hd_dim)
    hypervector2 = encode_sequence_to_hypervector(sequence2, kmer_size, sketch_size, hd_dim)
    
    # Compute cosine similarity between the two hypervectors
    similarity = cosine_similarity(hypervector1, hypervector2)
    
    print(f"Cosine Similarity: {similarity:.4f}")

if __name__ == "__main__":
    main()