# ⚡ Cloudflare Workers AI Explorer

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-orange)

> A sleek, fully interactive Streamlit app to explore Cloudflare Workers AI using the official Python SDK. This app enables text generation, image synthesis, speech recognition, translation, summarization, and classification directly from your browser — powered by Cloudflare’s distributed AI models.

---

## 🚀 Features

- 🔐 Securely input your Cloudflare API Token and Account ID
- 🧠 Choose from supported **Workers AI models**
- 📊 Automatically calculate and monitor **neuron usage** (10,000 free/day)
- 💬 **Text Generation** with Llama 3.1
- 🎨 **Text-to-Image** with DreamShaper 8
- 🧾 **Image Captioning** with LLaVA
- 🎙️ **Speech Recognition** using Whisper
- 🌐 **Language Translation** (M2M100)
- 🧠 **Text & Image Classification**
- ✂️ **Text Summarization**
- ⚡ 100% Streamlit-native, with **modern sidebar navigation** and no custom CSS

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/88plug/Cloudflare-Workers-AI-Explorer.git
cd Cloudflare-Workers-AI-Explorer
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install streamlit cloudflare requests
```

---

## 🔧 Configuration

You need two secrets:

- `CLOUDFLARE_API_TOKEN`: From [Cloudflare Dashboard → API Tokens](https://dash.cloudflare.com/profile/api-tokens)
- `CLOUDFLARE_ACCOUNT_ID`: From your [Cloudflare Workers AI dashboard](https://dash.cloudflare.com)

You can provide them in either of these ways:

### Option 1: Set environment variables

```bash
export CLOUDFLARE_API_TOKEN="your-token"
export CLOUDFLARE_ACCOUNT_ID="your-account-id"
```

Or use a `.env` file with [`python-dotenv`](https://pypi.org/project/python-dotenv/):

```ini
CLOUDFLARE_API_TOKEN=your-token
CLOUDFLARE_ACCOUNT_ID=your-account-id
```

### Option 2: Enter credentials via sidebar

When the app loads, use the sidebar input fields to securely provide credentials.

---

## ▶️ Usage

To run the app locally:

```bash
streamlit run app.py
```

Then open your browser to http://localhost:8501

---

## 🧪 Supported Models

| Feature              | Model ID                                |
|----------------------|------------------------------------------|
| Text Generation      | `@cf/meta/llama-3.1-8b-instruct`         |
| Text-to-Image        | `@cf/lykon/dreamshaper-8-lcm`            |
| Image Captioning     | `@cf/llava-hf/llava-1.5-7b-hf`           |
| Speech Recognition   | `@cf/openai/whisper`                     |
| Translations         | `@cf/meta/m2m100-1.2b`                   |
| Text Classification  | `@cf/huggingface/distilbert-sst-2-int8` |
| Image Classification | `@cf/microsoft/resnet-50`               |
| Summarization        | `@cf/facebook/bart-large-cnn`           |

---


