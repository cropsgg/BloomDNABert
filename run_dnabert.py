"""
DNABERT-2-117M Model Runner
Downloads and runs the DNABERT-2-117M model from Hugging Face
"""

import torch
from transformers import AutoTokenizer, AutoModel

def main():
    print("Loading DNABERT-2-117M model and tokenizer...")
    print("This may take a few minutes on first run as the model will be downloaded...")
    
    # Load tokenizer and model
    model_name = "zhihan1996/DNABERT-2-117M"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        print("✓ Model and tokenizer loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Set model to evaluation mode
    model.eval()
    
    # Example DNA sequences
    dna_sequences = [
        "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC",
        "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
        "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA"
    ]
    
    print("\n" + "="*60)
    print("Running inference on DNA sequences...")
    print("="*60)
    
    for i, dna_seq in enumerate(dna_sequences, 1):
        print(f"\nSequence {i}: {dna_seq[:50]}..." if len(dna_seq) > 50 else f"\nSequence {i}: {dna_seq}")
        
        # Tokenize the sequence
        inputs = tokenizer(dna_seq, return_tensors="pt")["input_ids"]
        
        # Get model outputs
        with torch.no_grad():
            hidden_states = model(inputs)[0]  # shape: [1, sequence_length, hidden_size]
        
        # Compute embeddings using mean pooling
        embedding_mean = torch.mean(hidden_states[0], dim=0)
        embedding_max = torch.max(hidden_states[0], dim=0)[0]
        
        print(f"  Input shape: {inputs.shape}")
        print(f"  Hidden states shape: {hidden_states.shape}")
        print(f"  Mean embedding shape: {embedding_mean.shape}")
        print(f"  Max embedding shape: {embedding_max.shape}")
        print(f"  Mean embedding (first 10 values): {embedding_mean[:10].tolist()}")
    
    print("\n" + "="*60)
    print("Model inference completed successfully!")
    print("="*60)
    
    # Additional test with a longer sequence
    print("\nTesting with a longer sequence...")
    long_sequence = "ATCG" * 50  # 200 nucleotides
    inputs_long = tokenizer(long_sequence, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        hidden_states_long = model(inputs_long)[0]
    embedding_long = torch.mean(hidden_states_long[0], dim=0)
    print(f"  Long sequence length: {len(long_sequence)}")
    print(f"  Embedding shape: {embedding_long.shape}")
    
    print("\n" + "="*60)
    print("All tests passed! Model is ready to use.")
    print("="*60)
    print("\nTo use the model in your code:")
    print("  from transformers import AutoTokenizer, AutoModel")
    print("  tokenizer = AutoTokenizer.from_pretrained('zhihan1996/DNABERT-2-117M', trust_remote_code=True)")
    print("  model = AutoModel.from_pretrained('zhihan1996/DNABERT-2-117M', trust_remote_code=True)")

if __name__ == "__main__":
    main()

