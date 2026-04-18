#Team details
K.Aniketh
A.Vigneshwar reddy
CH.Ramu

# HERCULES: Hierarchical Embedding-based Recursive Clustering Using LLMs for Efficient Summarization

HERCULES is an advanced document summarization system that uses semantic clustering to break down large texts into distinct topic groups before generating a comprehensive hierarchical summary.

## 🚀 Features

- **Semantic Chunking**: Intelligently splits text based on semantic meaning rather than just character counts.
- **Hierarchical Clustering**: Uses FAISS and embedding models to group related chunks of text.
- **Recursive Summarization**: Summarizes each cluster individually before generating a final executive summary.
- **Modern UI**: A sleek, black-and-white, full-screen React interface with tabbed navigation.
- **Dockerized**: Fully containerized for easy deployment.

## 🛠️ Tech Stack

- **Backend**: Python, Flask, LangChain, FAISS, Google Gemini (or OpenAI/Anthropic).
- **Frontend**: React, Vite, Material UI (MUI).
- **Containerization**: Docker, Docker Compose.

## 🏁 Getting Started

### Prerequisites

- Docker and Docker Compose installed.
- A Google Gemini API Key (or OpenAI/Anthropic key).

### Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd MadScientist
    ```

2.  **Configure Environment**:
    Copy `.env.example` to `.env` and add your API key:
    ```bash
    cp .env.example .env
    ```
    Edit `.env`:
    ```env
    GEMINI_API_KEY=your_actual_api_key_here
    ```

3.  **Run with Docker**:
    ```bash
    docker-compose up --build
    ```
    The application will be available at `http://localhost:5000` (Backend) and `http://localhost:5173` (Frontend, if running separately) or served together depending on the setup.

    *Note: In the current dev setup, the frontend runs on port 5174 via `npm run dev`.*

### Local Development (Without Docker)

1.  **Backend**:
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    export PYTHONPATH=.
    python src/app.py
    ```

2.  **Frontend**:
    ```bash
    cd frontend
    npm install
    npm run dev
    ```

## 📂 Project Structure

```
MadScientist/
├── src/                  # Backend Source Code
│   ├── app.py            # Flask Application Entry Point
│   ├── clustering/       # FAISS Clustering Logic
│   ├── config/           # Configuration Management
│   ├── embeddings/       # Embedding Providers
│   ├── llm/              # LLM Integration (Gemini, OpenAI)
│   ├── summarization/    # Core Pipeline Logic
│   └── utils.py          # Utility Functions
├── frontend/             # React Frontend
│   ├── src/
│   │   └── App.jsx       # Main UI Component
│   └── ...
├── Dockerfile            # Docker Build Instructions
├── docker-compose.yml    # Container Orchestration
└── requirements.txt      # Python Dependencies
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.
