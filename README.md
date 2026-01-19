# 🧬 Bloom-Enhanced DNABERT for Sickle Cell Variant Classification

A novel hybrid system combining **Bloom filters** for fast pathogenic k-mer detection with **DNABERT-2** embeddings for variant classification, featuring an interactive web dashboard with attention-based heatmap visualization.

## 🌟 Key Features

- **Multi-scale Bloom Filters**: Fast O(1) lookup of known pathogenic k-mers at k=6, 8, 10
- **DNABERT-2 Integration**: 117M parameter transformer for deep sequence understanding
- **Hybrid Classifier**: Combines pattern matching with deep learning
- **Interpretable Visualizations**: Attention heatmaps showing model focus
- **Sickle Cell Focus**: Specifically trained for HBB gene variants (E6V mutation)
- **Interactive Web Dashboard**: Gradio-based UI for real-time analysis

## 🏗️ Architecture

```
DNA Sequence → [Bloom Filters (k=6,8,10)] → Bloom Features (18-dim)
            ↘                                              ↘
              [DNABERT-2 Encoder] → Embeddings (768-dim) → [Hybrid MLP] → Pathogenic/Benign
                     ↓                                            ↓
              Attention Weights → Heatmap Visualization    Confidence Score
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Windows Users**: A triton stub is automatically created for Windows compatibility. The model will use PyTorch attention instead of Triton-accelerated attention.

### 2. Launch Web Dashboard

```bash
python app.py
```

The dashboard will be available at `http://localhost:7860`

### 3. Using the Dashboard

1. **Train the Model** (optional but recommended):
   - Go to the "Train Model" tab
   - Adjust epochs (default: 30)
   - Click "Train Model" and wait 5-10 minutes
   - View training metrics and test performance

2. **Analyze Sequences**:
   - Go to "Analyze Sequence" tab
   - Enter a DNA sequence or load an example
   - Click "Analyze Sequence"
   - View prediction, attention heatmap, and Bloom filter hits

3. **Example Sequences Included**:
   - Normal HBB (Wild-type)
   - Sickle Cell (HbS E6V mutation)
   - HbC Disease (E6K mutation)
   - Random Benign Variant

## 📊 Novel Contributions

### 1. Multi-scale Bloom Filter Pre-screening
Uses Bloom filters at three scales (k=6, 8, 10) populated with known pathogenic k-mers from the HBB gene region. Provides:
- Fast O(1) membership testing
- ~0.1% false positive rate
- Feature extraction (hit counts, ratios, positions)

### 2. Hybrid Feature Fusion
Concatenates:
- Bloom filter features (18 dimensions): hit statistics at multiple scales
- DNABERT-2 embeddings (768 dimensions): contextual sequence representation
- Total: 786-dimensional input to classifier

### 3. Interpretable Attention Heatmaps
- Extracts attention weights from DNABERT-2 transformer layers
- Aggregates across attention heads
- Maps to nucleotide-level importance
- Overlays Bloom filter hit positions
- Interactive Plotly visualizations

### 4. Sickle Cell Disease Detection
Specifically targets the HBB gene E6V mutation:
- Position: Codon 6 (nucleotide 20 in CDS)
- Normal: GAG (Glutamic acid)
- Mutant: GTG (Valine)
- Effect: Sickle cell disease (abnormal hemoglobin)

## 📁 Project Structure

```
DNABERT/
├── app.py                      # Gradio web dashboard
├── run_dnabert.py             # Original DNABERT test script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── bloom_dnabert/
│   ├── __init__.py
│   ├── bloom_filter.py        # Multi-scale Bloom filter module
│   ├── dnabert_wrapper.py     # DNABERT with attention extraction
│   ├── classifier.py          # Hybrid classifier model
│   ├── data_loader.py         # ClinVar data fetcher + synthetic data
│   └── visualizer.py          # Attention heatmap generation
└── data/
    └── hbb_variants.csv       # Generated variant dataset
```

## 🧪 Model Performance

Expected performance on synthetic test set:
- **Accuracy**: ~85-95% (depending on training)
- **Precision**: High for pathogenic variants
- **Recall**: Good detection of sickle cell mutations
- **AUC-ROC**: >0.90
- **Inference Speed**: <100ms per sequence on CPU

## 🔬 Scientific Background

### Sickle Cell Disease
- Most common inherited blood disorder
- Caused by HBB gene mutation (chr11:5227002 T>A)
- Point mutation at codon 6: Glu → Val
- Results in abnormal hemoglobin (HbS)
- Affects millions worldwide, especially in Africa, Mediterranean, Middle East

### Why This Approach is Novel
1. **Speed**: Bloom filters pre-screen in microseconds
2. **Accuracy**: Deep learning captures complex patterns
3. **Interpretability**: Attention heatmaps explain predictions
4. **Scalability**: Can be extended to other genes/diseases

## 🛠️ Advanced Usage

### Training on Custom Data

```python
from bloom_dnabert import MultiScaleBloomFilter, DNABERTWrapper
from bloom_dnabert.classifier import HybridClassifierPipeline

# Initialize components
bloom_filter = MultiScaleBloomFilter()
bloom_filter.load_hbb_pathogenic_variants()

dnabert = DNABERTWrapper()

# Create pipeline
pipeline = HybridClassifierPipeline(bloom_filter, dnabert)

# Train
pipeline.train(train_sequences, train_labels, epochs=50)

# Predict
result = pipeline.predict("CACGTGGTCTACCCCTGAGGAG...")
print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']:.3f}")
```

### Extracting Attention Weights

```python
from bloom_dnabert import DNABERTWrapper, AttentionVisualizer

dnabert = DNABERTWrapper()
visualizer = AttentionVisualizer(dnabert, bloom_filter)

# Create attention heatmap
fig = visualizer.create_attention_heatmap(sequence)
fig.show()

# Get nucleotide importance
importance_plot = visualizer.create_nucleotide_importance_plot(sequence)
importance_plot.show()
```

## 📈 Expected Results

The system demonstrates:
- **Clinical Relevance**: Correctly identifies known sickle cell mutations
- **Fast Screening**: Bloom filters identify pathogenic k-mers instantly
- **Interpretability**: Attention focuses on mutation sites (codon 6)
- **Promising Accuracy**: High performance on synthetic dataset

## ⚠️ Limitations

- Model trained on synthetic data (for demonstration)
- DNABERT-2 max sequence length: ~2048 tokens (BPE)
- Bloom filters have false positives (tunable, currently 0.1%)
- Initial version doesn't include fine-tuning on real clinical data
- Windows: Uses PyTorch attention (slightly slower than Triton)

## 🔮 Future Work

- Fine-tune DNABERT-2 on real ClinVar data
- Add support for more genes (BRCA1, BRCA2, TP53, etc.)
- Implement counting Bloom filters for frequency tracking
- Add multi-variant analysis (compound heterozygotes)
- Deploy as clinical decision support tool
- Extend to other hemoglobinopathies

## 📚 References

- **DNABERT-2**: Zhou et al. (2023) - [https://huggingface.co/zhihan1996/DNABERT-2-117M](https://huggingface.co/zhihan1996/DNABERT-2-117M)
- **Bloom Filters in Bioinformatics**: Solomon & Kingsford (2016), BMC Bioinformatics
- **ClinVar**: Landrum et al. (2018), Nucleic Acids Research
- **Sickle Cell Disease**: Piel et al. (2017), NEJM

## 🤝 Citation

If you use this system in your research, please cite:

```bibtex
@software{bloom_dnabert_2026,
  title={Bloom-Enhanced DNABERT for Sickle Cell Variant Classification},
  author={Your Name},
  year={2026},
  note={A novel hybrid system combining Bloom filters and transformer models for pathogenic variant detection}
}
```

## ⚖️ Disclaimer

**This is a research prototype for educational and research purposes only.**  
This system should NOT be used for clinical diagnosis or treatment decisions.  
Always consult with qualified healthcare professionals and genetic counselors for medical interpretation of genetic variants.

## 📝 License

This project is provided as-is for research and educational purposes.

---

**Built with**: PyTorch, Transformers, Gradio, Plotly, and ❤️ for genomics research

