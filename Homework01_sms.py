import hashlib
import random
from typing import Set, List, Dict, Tuple
from collections import defaultdict
import time

class Shingling:
    """
    constructs k-shingles from documents and represents them as sets of integers.
    uses dictionary encoding to map shingles to unique IDs.
    """
    def __init__(self, k: int = 10):
        #initialize the Shingling class.

        self.k = k
        self.shingle_to_id = {}  #dictionary encoding: shingle -> unique ID
        self.next_id = 0
    
    def generate_shingles(self, document: str) -> Set[int]:
        """
        generate k-shingles from a document and return as a set of integer IDs.
        """
        #remove extra whitespace and convert to lowercase
        doc = ' '.join(document.lower().split()) 
        
        shingles = set()
        
        #create k-shingles
        for i in range(len(doc) - self.k + 1):
            shingle = doc[i:i + self.k]
            
            #dictionary encoding: assign unique ID to each unique shingle
            if shingle not in self.shingle_to_id:
                self.shingle_to_id[shingle] = self.next_id
                self.next_id += 1
            
            shingles.add(self.shingle_to_id[shingle])
        
        return shingles
    
    def get_vocabulary_size(self) -> int:        
        return len(self.shingle_to_id) #return total number of unique shingles 


class CompareSets:
    
    @staticmethod
    def jaccard_similarity(set1: Set[int], set2: Set[int]) -> float:
        """
        calculate Jaccard similarity between two sets of integers and then
        returns jaccard similarity
        """
        if len(set1) == 0 and len(set2) == 0:
            return 1.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0


class MinHashing:
    """
    builds MinHash signatures for sets of integers using multiple hash functions
    """
    def __init__(self, num_hashes: int = 100): #initializes MinHashing with n number of hash functions
        self.num_hashes = num_hashes #num_hashes is number of hash functions (signature length)
        self.hash_functions = self._generate_hash_functions()
    
    def _generate_hash_functions(self) -> List[Tuple[int, int]]:
        #generates parameters for hash functions of the form h(x) = (a*x + b) mod p

        random.seed(42)
        
        hash_funcs = []
        for _ in range(self.num_hashes):
            a = random.randint(1, 10**9)
            b = random.randint(0, 10**9)
            hash_funcs.append((a, b))
        
        return hash_funcs #list of (a, b) tuples for each hash function
    
    def compute_signature(self, shingle_set: Set[int]) -> List[int]:
        #computes MinHash signature for a set of shingles.

        signature = []
        p = 2**31 - 1  
        
        for a, b in self.hash_functions:
            min_hash = float('inf')
            
            for shingle_id in shingle_set:
                #computes hash value (a * shingle_id + b) mod p
                hash_value = (a * shingle_id + b) % p
                min_hash = min(min_hash, hash_value)
            
            signature.append(min_hash if min_hash != float('inf') else 0)
        
        return signature #MinHash signature as a list of integers


class CompareSignatures: #estimates similarity between MinHash signatures
    
    @staticmethod
    def similarity(sig1: List[int], sig2: List[int]) -> float:
        if len(sig1) != len(sig2):
            raise ValueError("Signatures must have the same length")
        
        if len(sig1) == 0:
            return 0.0
        
        matches = sum(1 for i in range(len(sig1)) if sig1[i] == sig2[i])
        return matches / len(sig1)


class LSH:
    """
    implements LSH to find candidate similar pairs.
    uses banding technique to hash signatures into buckets.
    """
    
    def __init__(self, num_bands: int, rows_per_band: int):
        
        #initialize LSH with banding parameters
  
        self.num_bands = num_bands
        self.rows_per_band = rows_per_band
        self.signature_length = num_bands * rows_per_band
    
    def find_candidate_pairs(self, signatures: Dict[str, List[int]], 
                            threshold: float = 0.8) -> Set[Tuple[str, str]]:
        #find candidate pairs of documents that may be similar.
        
        candidate_pairs = set()
        
        #for each band
        for band_idx in range(self.num_bands):
            #hash table for this band: band_hash -> list of doc_ids
            buckets = defaultdict(list)
            
            start_row = band_idx * self.rows_per_band
            end_row = start_row + self.rows_per_band
            
            #hash each document's band
            for doc_id, signature in signatures.items():
                if len(signature) < self.signature_length:
                    continue
                
                #extract band portion of signature
                band = tuple(signature[start_row:end_row])
                
                #hash the band 
                band_hash = hash(band)
                buckets[band_hash].append(doc_id)
            
            #dind pairs in same bucket
            for bucket_docs in buckets.values():
                if len(bucket_docs) > 1:
                    
                    #all pairs in this bucket are candidates
                    for i in range(len(bucket_docs)):
                        for j in range(i + 1, len(bucket_docs)):
                            doc1, doc2 = bucket_docs[i], bucket_docs[j]

                            pair = tuple(sorted([doc1, doc2]))
                            candidate_pairs.add(pair)
        
        return candidate_pairs


def load_sms_spam_collection(filename='SMSSpamCollection', num_docs=None):
    """
    Load the SMS Spam Collection dataset.
    """
    documents = {}
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if num_docs and idx >= num_docs:
                    break
        
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    label, message = parts
                    doc_id = f"sms_{idx+1:04d}_{label}"
                    documents[doc_id] = message
    except FileNotFoundError:
        print(f"Error: Could not find file '{filename}'")
        print("Please make sure the SMSSpamCollection file is in the current directory.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    return documents

def collect_similar_pairs(exact_similarities, documents, threshold):
    """
    convert exact_similarities dict into a list of
    (docA, textA, jaccard, docB, textB) tuples.
    this function is for when to print most similar pairs in main.
    """
    result = []

    for (docA, docB), sim in exact_similarities.items():
        if sim >= threshold:
            result.append((
                docA,
                documents[docA],
                sim,
                docB,
                documents[docB]
            ))

    #sort by similarity (highest first)
    result.sort(key=lambda x: x[2], reverse=True)
    return result


def main():
    """
    main function to demonstrate the document similarity detection system.
    """
    #load SMS spam collection dataset
    documents = load_sms_spam_collection('SMSSpamCollection', num_docs=500) #num_docs corpus size parameter
    
    if documents is None:
        print("Failed to load dataset. Exiting.")
        return
    
    print("=" * 80)
    print("DOCUMENT SIMILARITY DETECTION SYSTEM - SMS SPAM COLLECTION")
    print("=" * 80)
    print(f"\nCorpus size: {len(documents)} SMS messages")
    
    #show sample of loaded documents
    print("\nSample documents:")
    for i, (doc_id, text) in enumerate(list(documents.items())[:3]):
        label = "SPAM" if "spam" in doc_id else "HAM"
        preview = text[:60] + "..." if len(text) > 60 else text
        print(f"  {doc_id} [{label}]: {preview}")
    
    #parameters
    k = 5  #shingle length
    num_hashes = 100  #signature length
    similarity_threshold = 0.6
    
    print(f"\nParameters:")
    print(f"  - Shingle length (k): {k}")
    print(f"  - MinHash signature length: {num_hashes}")
    print(f"  - Similarity threshold: {similarity_threshold}")
    
    #shingling
    print("\n" + "-" * 80)
    print("STEP 1: SHINGLING")
    print("-" * 80)
    
    start_time = time.time()
    shingling = Shingling(k=k)
    shingle_sets = {}
    
    for doc_id, doc_text in documents.items():
        shingle_sets[doc_id] = shingling.generate_shingles(doc_text)
    
    shingling_time = time.time() - start_time
    print(f"Shingling completed in {shingling_time:.4f} seconds")
    print(f"Total unique shingles: {shingling.get_vocabulary_size()}")
    print(f"\nShingle set sizes:")
    for doc_id, shingles in shingle_sets.items():
        print(f"  {doc_id}: {len(shingles)} shingles")
    
    #jaccard similarity
    print("\n" + "-" * 80)
    print("STEP 2: EXACT JACCARD SIMILARITY (Ground Truth)")
    print("-" * 80)
    
    start_time = time.time()
    compare_sets = CompareSets()
    doc_ids = list(documents.keys())
    exact_similarities = {}
    
    print(f"\nSimilar pairs (Jaccard >= {similarity_threshold}):")
    for i in range(len(doc_ids)):
        for j in range(i + 1, len(doc_ids)):
            doc1, doc2 = doc_ids[i], doc_ids[j]
            similarity = compare_sets.jaccard_similarity(
                shingle_sets[doc1], 
                shingle_sets[doc2]
            )
            exact_similarities[(doc1, doc2)] = similarity
            
            if similarity >= similarity_threshold:
                print(f"  {doc1} - {doc2}: {similarity:.4f}")
    
    jaccard_time = time.time() - start_time
    print(f"\nJaccard computation completed in {jaccard_time:.4f} seconds")
    
    #MinHashing
    print("\n" + "-" * 80)
    print("STEP 3: MINHASHING")
    print("-" * 80)
    
    start_time = time.time()
    minhashing = MinHashing(num_hashes=num_hashes)
    signatures = {}
    
    for doc_id, shingle_set in shingle_sets.items():
        signatures[doc_id] = minhashing.compute_signature(shingle_set)
    
    minhash_time = time.time() - start_time
    print(f"MinHash signatures computed in {minhash_time:.4f} seconds")
    
    #compare signatures
    print("\n" + "-" * 80)
    print("STEP 4: SIGNATURE SIMILARITY ESTIMATION")
    print("-" * 80)
    
    start_time = time.time()
    compare_sigs = CompareSignatures()
    
    print(f"\nEstimated similar pairs (signature similarity >= {similarity_threshold}):")
    print(f"{'Pair':<20} {'Exact':<12} {'Estimated':<12} {'Error':<10}")
    print("-" * 60)
    
    for i in range(len(doc_ids)):
        for j in range(i + 1, len(doc_ids)):
            doc1, doc2 = doc_ids[i], doc_ids[j]
            estimated_sim = compare_sigs.similarity(signatures[doc1], signatures[doc2])
            exact_sim = exact_similarities[(doc1, doc2)]
            error = abs(exact_sim - estimated_sim)
            
            if estimated_sim >= similarity_threshold:
                print(f"{doc1}-{doc2:<16} {exact_sim:<12.4f} {estimated_sim:<12.4f} {error:<10.4f}")
    
    signature_time = time.time() - start_time
    print(f"\nSignature comparison completed in {signature_time:.4f} seconds")
    
    #LSH
    print("\n" + "-" * 80)
    print("STEP 5: LOCALITY-SENSITIVE HASHING (LSH)")
    print("-" * 80)
    
    num_bands = 20
    rows_per_band = num_hashes // num_bands
    
    print(f"\nLSH Parameters:")
    print(f"  - Number of bands: {num_bands}")
    print(f"  - Rows per band: {rows_per_band}")
    print(f"  - Probability threshold ≈ (1/{num_bands})^(1/{rows_per_band}) = {(1/num_bands)**(1/rows_per_band):.4f}")
    
    start_time = time.time()
    lsh = LSH(num_bands=num_bands, rows_per_band=rows_per_band)
    candidate_pairs = lsh.find_candidate_pairs(signatures, threshold=similarity_threshold)
    lsh_time = time.time() - start_time
    
    print(f"\nLSH found {len(candidate_pairs)} candidate pairs in {lsh_time:.4f} seconds")
    print(f"\nCandidate pairs:")
    for doc1, doc2 in sorted(candidate_pairs):
        exact_sim = exact_similarities.get((doc1, doc2), 
                                          exact_similarities.get((doc2, doc1), 0.0))
        estimated_sim = compare_sigs.similarity(signatures[doc1], signatures[doc2])
        print(f"  {doc1} - {doc2}: Exact={exact_sim:.4f}, Estimated={estimated_sim:.4f}")
    
    #summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nExecution times:")
    print(f"  - Shingling: {shingling_time:.4f}s")
    print(f"  - Exact Jaccard: {jaccard_time:.4f}s")
    print(f"  - MinHashing: {minhash_time:.4f}s")
    print(f"  - Signature comparison: {signature_time:.4f}s")
    print(f"  - LSH: {lsh_time:.4f}s")
    print(f"  - Total: {shingling_time + jaccard_time + minhash_time + signature_time + lsh_time:.4f}s")
    
    print(f"\nAccuracy:")
    total_pairs = len(exact_similarities)
    true_similar = sum(1 for sim in exact_similarities.values() if sim >= similarity_threshold)
    lsh_candidates = len(candidate_pairs)
    
    print(f"  - Total pairs: {total_pairs}")
    print(f"  - True similar pairs (Jaccard >= {similarity_threshold}): {true_similar}")
    print(f"  - LSH candidate pairs: {lsh_candidates}")


    #prints most similar pairs of messages
    similar_pairs = collect_similar_pairs(exact_similarities, documents, similarity_threshold)
    print("\n========================================================")
    print(" MOST SIMILAR SMS MESSAGE PAIRS (Jaccard ≥ threshold)")
    print("========================================================")

    for i, (a, text_a, jaccard, b, text_b) in enumerate(similar_pairs, start=1):
        print(f"\n#{i}")
        print(f"Document A ({a}):")
        print(f"  \"{text_a}\"")
        print(f"Document B ({b}):")
        print(f"  \"{text_b}\"")
        print(f"Jaccard similarity: {jaccard:.4f}")
        print("--------------------------------------------------------")




if __name__ == "__main__":
    main()
