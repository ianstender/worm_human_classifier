
# Species-Specific DNA Classification Using Machine Learning

This project uses convolutional neural networks (CNNs) to classify DNA sequences as belonging to either humans or worms. By analyzing 200 bp-long sequences, the model identifies species-specific patterns and motifs, showcasing the power of deep learning in bioinformatics.

## Table of Contents
- [Background](#background)
- [Dataset](#dataset)
- [Model Architectures](#model-architectures)
  - [DeepBind CNN](#deepbind-cnn)
  - [SmallCNN](#smallcnn)
  - [IanCNN](#iancnn)
- [Model Performance](#model-performance)
- [Interpretability](#interpretability)
  - [Attribution Analysis](#attribution-analysis)
  - [Global Importance Analysis](#global-importance-analysis)
  - [Filter Interpretation](#filter-interpretation)
  - [Motif Discovery](#motif-discovery)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---

## Background
Understanding species-specific DNA sequences has applications in:
- **Phylogenetics**: Exploring evolutionary histories and constructing phylogenetic trees.
- **Molecular Evolution**: Detecting conserved motifs under selective evolutionary pressure.
- **Forensics**: DNA-based species identification in complex scenarios.

This project aims to leverage machine learning to identify distinguishing DNA patterns and motifs.

---

## Dataset
The dataset contains:
- **Human DNA**: 3,250 sequences (200 bp each)
- **Worm DNA**: 3,250 sequences (200 bp each)

**Training/Validation Split**: 80% training, 20% validation  
**Target Variable**:  
- `1`: Human  
- `0`: Worm  

---

## Model Architectures

### DeepBind CNN
A simplified version of the DeepBind model was adapted for binary classification of 200 bp sequences. Features include:
- Single convolutional layer
- RELU activation
- Max pooling
- Fully connected layer
- Binary classifier

**Performance**:  
- Training Accuracy: 60%  
- Validation Accuracy: ~90% (overfitting suspected)

### SmallCNN
A smaller CNN with fewer layers was tested to address overfitting.  
**Performance**:  
- Training Accuracy: 63%  
- Validation Accuracy: 65%  

### IanCNN
The final model introduced:
- Two convolutional blocks (128 filters each)
- Global average pooling
- Dropout layers to reduce overfitting
- Fully connected layers

**Performance**:  
- Validation Accuracy: ~94%  

---

## Model Performance
The IanCNN model achieved high validation accuracy, successfully identifying species-specific features in DNA sequences.

---

## Interpretability

### Attribution Analysis
Highlights the contribution of individual nucleotides to the model's predictions for specific sequences.

### Global Importance Analysis
Evaluates the model's response to inserting specific motifs (e.g., TATA box) at various positions in a sequence. Demonstrates the learned importance of motifs.

### Filter Interpretation
Analyzes positional nucleotide preferences captured by convolutional filters, uncovering potential motifs.

### Motif Discovery
Utilized the **TomTom Motif Discovery Tool** with outputs compared against the **JASPAR** database, identifying significant motifs like **EBF1**, involved in human early B-cell development.

---

## Usage
1. Clone this repository:  
   ```bash
   git clone https://github.com/yourusername/species-dna-classifier.git
   ```
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare the dataset (ensure sequences are in the specified format).
4. Train the model:  
   ```bash
   python train_model.py
   ```
5. Visualize motifs:  
   ```bash
   python interpret_model.py
   ```

---

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

---

## License
This project is licensed under the [MIT License](LICENSE).

---
