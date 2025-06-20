# Web Wizard: Translation Model Evaluation + Web Interface

This project combines an interactive web interface with robust backend evaluation of machine translation models from English to Spanish. It supports running evaluations, visualizing results, and viewing translation output — all via a Flask + React web app in Docker.

## Translation Models Evaluated

| Model | Description |
|-------|-------------|
| `Helsinki-NLP/opus-mt-en-es` | MarianNMT-based specialized English → Spanish model |
| `facebook/m2m100_418M` | Facebook's multilingual M2M100 model |
| `facebook/nllb-200-distilled-600M` | Meta's No Language Left Behind (NLLB) |

## Evaluation Metrics

We evaluate translations using the following metrics:

- **BLEU** – N-gram precision with brevity penalty  
- **CHRF** – Character n-gram F-score  
- **METEOR** – Word-level metric using synonym matching and ordering  
- **BERTScore** – Semantic similarity using BERT embeddings  

## Quickstart (Using Docker)

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed
- Optional: [Git](https://git-scm.com/)

### 1. Clone the Repository

```bash
git clone https://github.com/mustaphouni04/synthesis-II.git
cd synthesis-II
```

### 2. Build and Run the Docker Container

```bash
docker build -t translation-evaluator .
docker run -d -p 5000:5000 --name evaluator-container translation-evaluator
```

The web app will be available at: [http://localhost:5000](http://localhost:5000)

## Backend CLI Evaluation (Optional)

If you want to run the evaluation without Docker, you can also do it using the CLI:

### 1. Set up Environment

```bash
cd ml_server
pip install -r requirements.txt
```

### 2. Run Evaluation

```bash
python translation_model_evaluation.py --csv path/to/translations.csv --samples 1000 --batch-size 16
```

### 3. Visualize Results

```bash
python visualize_results.py --csv translation_evaluation_results.csv
```

## Input CSV Format

| Column | Description                     |
|--------|---------------------------------|
| `source` | English sentence               |
| `target` | Reference Spanish translation |

## Output

The evaluation will:

- Print comparison results in terminal
- Save a CSV of detailed scores per model
- Display sample translations
- Create charts (if visualizing) for:
  - Bar charts per metric
  - Radar chart comparing models
  - Grammatical vs semantic breakdowns

## Web Interface Features

- Upload a CSV for batch evaluation
- See per-model scores and ranking
- View sample translations
- Download results

## Developer Notes

- React frontend is built using [Vite](https://vitejs.dev/)
- Flask backend serves static frontend and handles model inference
- Dockerfile uses multi-stage build to optimize size and performance

## Contributing

Contributions welcome. Feel free to:

- Suggest new metrics or models
- Improve UI/UX of the web interface
- Report bugs

## License

MIT – see `LICENSE` file.
