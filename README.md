# 💃 DanceGen — Deep Learning Based Dance Motion Generation

DanceGen is a deep learning project focused on learning motion patterns from dance videos and generating expressive dance sequences. The project provides a modular pipeline for video preprocessing, training, and experimentation with generative modeling techniques.

---

## 🚀 Overview

DanceGen explores how neural networks can learn temporal and spatial motion dynamics from dance data. The repository is designed to support reproducible experiments and rapid prototyping for motion-based generative models.

The project includes:

* Video preprocessing and data preparation pipeline
* Training scripts for generative motion modeling
* Experiment tracking and output organization
* Clean modular architecture for extensibility

---

## 📂 Repository Structure

```
DanceGen/
│
├── scripts/        # Training, preprocessing, and utility scripts
├── runs/           # Experiment outputs (ignored for large artifacts)
├── .gitignore      # Large file exclusion rules
├── .gitattributes  # Git LFS configuration
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Abhishekreddy3236/DanceGen.git
cd DanceGen
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run training:

```bash
python scripts/train.py
```

Run preprocessing:

```bash
python scripts/preprocess.py
```

---

## 📦 Dataset

Due to size constraints, datasets are not included in this repository.

You can place your video dataset inside the project directory following the structure expected by the preprocessing scripts.

---

## 🧪 Experiments

The `runs/` directory stores experiment artifacts such as:

* Generated outputs
* Logs
* Checkpoints
* Visualizations

Large files are excluded from version control.

---

## 🎯 Future Work

* Integration with diffusion-based motion models
* Multi-modal conditioning (music + motion)
* Real-time dance generation
* Improved evaluation metrics for motion realism

---

## 👨‍💻 Author

**Abhishek Reddy**
AI & Deep Learning Enthusiast

---

⭐ If you find this project useful, consider starring the repository.
