import os
import re
import streamlit as st
import requests
from io import BytesIO
from getpass import getpass

# Import the Cloudflare API client; ensure you have installed it:
#     pip install cloudflare
try:
    from cloudflare import Cloudflare
except ImportError:
    st.error("Cloudflare package not installed. Run 'pip install cloudflare' to install it.")
    st.stop()

# Optionally load environment variables from a .env file
# from dotenv import load_dotenv
# load_dotenv()

# --- Utility: Slugify function for anchors ---
def slugify(text):
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9]+', '-', text)
    text = re.sub(r'-+', '-', text)
    return text.strip('-')

# --- Page Configuration & Theming ---
st.set_page_config(
    page_title="Cloudflare Workers AI Explorer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Caching ---
@st.cache_resource(show_spinner=False)
def init_cloudflare_client(api_token: str, account_id: str):
    return Cloudflare(api_token=api_token)

@st.cache_data(ttl=60)
def get_neuron_usage():
    # In production, implement actual API call to monitor neuron consumption.
    return 3456  # Dummy value for simulation

# --- Helper Functions ---
def run_model(client, model, account_id, **kwargs):
    """
    Call an AI model (JSON response). For now, automatically use the model's full context window
    as max_tokens.
    """
    # Set max_tokens to the model's full context window if available,
    # otherwise, default to 256.
    context_window = kwargs.pop("context_window", 256)
    kwargs["max_tokens"] = context_window
    return client.ai.run(model, account_id=account_id, **kwargs)

def run_model_raw(client, model, account_id, **kwargs):
    """
    Call an AI model returning a raw response (e.g. image bytes).
    """
    return client.ai.with_raw_response.run(model, account_id=account_id, **kwargs)

# --- Sidebar: Credentials, Neuron Usage, and Global Settings ---
st.sidebar.header("Cloudflare Credentials")
api_token = os.environ.get("CLOUDFLARE_API_TOKEN", "")
account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
if not api_token:
    api_token = st.sidebar.text_input("Enter your Cloudflare API Token", type="password")
if not account_id:
    account_id = st.sidebar.text_input("Enter your Cloudflare Account ID")
if not (api_token and account_id):
    st.sidebar.warning("Both API Token and Account ID are required.")
    st.stop()

# Initialize Cloudflare client.
client = init_cloudflare_client(api_token, account_id)

# Display neuron usage.
neurons_used = get_neuron_usage()
st.sidebar.info(f"Neuron usage: {neurons_used} / 10,000")

st.sidebar.divider()
# Sidebar navigation for features.
feature_to_tab = {
    "Text Generation (Chat)": "Text Generation",
    "Text to Image": "Text to Image",
    "Image to Text": "Image to Text",
    "Speech Recognition": "Speech Recognition",
    "Translations": "Translations",
    "Text Classification": "Text Classification",
    "Image Classification": "Image Classification",
    "Summarization": "Summarization"
}
selected_feature = st.sidebar.selectbox("Choose a feature to explore:", list(feature_to_tab.keys()))
selected_tab_slug = slugify(feature_to_tab[selected_feature])
st.experimental_set_query_params(tab=selected_tab_slug)

# --- Global AI Model Settings for Text Generation ---
# For this simplified version, we'll allow model selection only.
with st.sidebar.expander("AI Model Settings (Text Generation)", expanded=True):
    # A dictionary mapping display names to their model details.
    text_models = {
        "Llama 3.1 Instruct (context: 7,968 tokens)": {
            "id": "@cf/meta/llama-3.1-8b-instruct",
            "context_window": 7968
        },
        "Llama 3.1 Instruct Fast (context: 128,000 tokens)": {
            "id": "@cf/meta/llama-3.1-8b-instruct-fast",
            "context_window": 128000
        }
        # Additional models can be added here.
    }
    selected_text_model_name = st.selectbox("Choose Text Generation Model", list(text_models.keys()))
    text_model = text_models[selected_text_model_name]
    st.markdown(f"Using full context window: **{text_model['context_window']} tokens**")
    # Note: Other generation parameters (temperature, top_p, etc.) are not exposed now.

# --- Main App Header ---
st.title("⚡ Cloudflare Workers AI Explorer")
st.markdown(
    """
    This app demonstrates how to run AI models on Cloudflare’s network using the official Python SDK.
    
    The free allocation provides up to 10,000 Neurons daily (neurons measure GPU compute consumption).
    Use the sidebar to enter credentials, check neuron usage, select a text model (which uses its full context window),
    and navigate to specific features like text generation, image processing, and more.
    """
)
st.toast("Welcome to the AI Explorer!", icon="🚀")

# --- Auto-scroll to Selected Feature ---
query_params = st.experimental_get_query_params()
if "tab" in query_params:
    target_tab = query_params["tab"][0]
    scroll_js = f"""
    <script>
      var target = document.getElementById("{target_tab}");
      if(target){{
          target.scrollIntoView({{ behavior: 'smooth' }});
      }}
    </script>
    """
    st.components.v1.html(scroll_js, height=0, scrolling=False)

# --- Main Layout with Tabs ---
tab_titles = [
    "Text Generation",
    "Text to Image",
    "Image to Text",
    "Speech Recognition",
    "Translations",
    "Text Classification",
    "Image Classification",
    "Summarization"
]
tabs = st.tabs(tab_titles)

# === 1. Text Generation (Chat) Tab ===
with tabs[0]:
    st.markdown(f'<div id="{slugify("Text Generation")}"></div>', unsafe_allow_html=True)
    st.header("Text Generation (@cf/meta/llama-3.1-8b-instruct)")
    st.markdown("Chat with the assistant. Your message will be processed using the selected model with its full context capacity.")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    for entry in st.session_state.chat_history:
        if entry["role"] == "user":
            with st.chat_message("user"):
                st.write(entry["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(entry["content"])
    user_input = st.chat_input("Enter your message")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("Waiting for response..."):
            messages = [{
                "role": "system",
                "content": "You are a friendly assistant for Jupyter Notebook users. Respond in Markdown."
            }] + st.session_state.chat_history
            try:
                result = run_model(
                    client,
                    text_model["id"],
                    account_id,
                    messages=messages,
                    context_window=text_model["context_window"]
                )
                assistant_response = result.get("response", "No response received.")
            except Exception as e:
                assistant_response = f"Error: {e}"
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
        st.rerun()

# === 2. Text to Image Tab ===
with tabs[1]:
    st.markdown(f'<div id="{slugify("Text to Image")}"></div>', unsafe_allow_html=True)
    st.header("Text to Image (@cf/lykon/dreamshaper-8-lcm)")
    with st.form("text_to_image_form"):
        prompt_img = st.text_input("Enter a text prompt", "A software developer excited about AI, smiling wide")
        submit_img = st.form_submit_button("Generate Image")
    if submit_img:
        with st.spinner("Generating image..."):
            try:
                response = run_model_raw(client, "@cf/lykon/dreamshaper-8-lcm", account_id, prompt=prompt_img)
                image_bytes = response.read()
                st.image(image_bytes, caption="Generated Image", use_column_width=True)
            except Exception as e:
                st.error(f"Image generation failed: {e}")

# === 3. Image to Text Tab ===
with tabs[2]:
    st.markdown(f'<div id="{slugify("Image to Text")}"></div>', unsafe_allow_html=True)
    st.header("Image to Text (@cf/llava-hf/llava-1.5-7b-hf)")
    st.markdown("Submit an image URL or upload an image to get a descriptive caption.")
    col1, col2 = st.columns(2)
    image_data = None
    with col1:
        source_choice = st.radio("Select image source", ["URL", "Upload"], key="img2txt_source")
    with col2:
        if source_choice == "URL":
            image_url = st.text_input("Image URL", "https://blog.cloudflare.com/content/images/2017/11/lava-lamps.jpg")
            if image_url:
                try:
                    resp = requests.get(image_url, allow_redirects=True)
                    image_data = resp.content
                    st.image(image_data, caption="Input Image", use_column_width=True)
                except Exception as e:
                    st.error(f"Failed to load image: {e}")
        else:
            uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="img2txt_upload")
            if uploaded_img:
                image_data = uploaded_img.read()
                st.image(image_data, caption="Uploaded Image", use_column_width=True)
    if image_data and st.button("Describe Image"):
        with st.spinner("Describing image..."):
            try:
                result = run_model(
                    client,
                    "@cf/llava-hf/llava-1.5-7b-hf",
                    account_id,
                    image=image_data,
                    prompt="Describe this photo"
                )
                st.markdown(result.get("description", "No description available."))
            except Exception as e:
                st.error(f"Error describing image: {e}")

# === 4. Speech Recognition Tab ===
with tabs[3]:
    st.markdown(f'<div id="{slugify("Speech Recognition")}"></div>', unsafe_allow_html=True)
    st.header("Speech Recognition (@cf/openai/whisper)")
    with st.form("speech_recognition_form"):
        audio_url = st.text_input("Enter audio URL", "https://raw.githubusercontent.com/craigsdennis/notebooks-cloudflare-workers-ai/main/assets/craig-rambling.mp3")
        submit_audio = st.form_submit_button("Transcribe Audio")
    if submit_audio:
        with st.spinner("Transcribing audio..."):
            try:
                audio_resp = requests.get(audio_url)
                st.audio(audio_resp.content)
                result = run_model(client, "@cf/openai/whisper", account_id, audio=audio_resp.content)
                st.subheader("Transcription:")
                st.write(result.get("text", "No transcription available."))
            except Exception as e:
                st.error(f"Audio transcription failed: {e}")

# === 5. Translations Tab ===
with tabs[4]:
    st.markdown(f'<div id="{slugify("Translations")}"></div>', unsafe_allow_html=True)
    st.header("Translations (@cf/meta/m2m100-1.2b)")
    with st.form("translations_form"):
        source_text = st.text_area("Enter English text", "Artificial intelligence is pretty impressive these days. It is a bonkers time to be a builder", height=100)
        target_language = st.text_input("Target language", "spanish")
        submit_trans = st.form_submit_button("Translate")
    if submit_trans:
        with st.spinner("Translating..."):
            try:
                result = run_model(
                    client,
                    "@cf/meta/m2m100-1.2b",
                    account_id,
                    text=source_text,
                    source_lang="english",
                    target_lang=target_language
                )
                st.subheader("Translated Text:")
                st.write(result.get("translated_text", "No translation available."))
            except Exception as e:
                st.error(f"Translation error: {e}")

# === 6. Text Classification Tab ===
with tabs[5]:
    st.markdown(f'<div id="{slugify("Text Classification")}"></div>', unsafe_allow_html=True)
    st.header("Text Classification (@cf/huggingface/distilbert-sst-2-int8)")
    with st.form("text_classification_form"):
        text_for_classification = st.text_input("Enter text for classification", "This taco is delicious")
        submit_class = st.form_submit_button("Classify Text")
    if submit_class:
        with st.spinner("Classifying text..."):
            try:
                result = run_model(
                    client,
                    "@cf/huggingface/distilbert-sst-2-int8",
                    account_id,
                    text=text_for_classification
                )
                st.subheader("Classification Result:")
                st.json(result)
            except Exception as e:
                st.error(f"Classification error: {e}")

# === 7. Image Classification Tab ===
with tabs[6]:
    st.markdown(f'<div id="{slugify("Image Classification")}"></div>', unsafe_allow_html=True)
    st.header("Image Classification (@cf/microsoft/resnet-50)")
    st.markdown("Provide an image URL or upload an image to classify its content.")
    col_a, col_b = st.columns(2)
    image_data_class = None
    with col_a:
        source_class = st.radio("Select image source", ["URL", "Upload"], key="imgclass_source")
    with col_b:
        if source_class == "URL":
            image_url_class = st.text_input("Image URL", "https://raw.githubusercontent.com/craigsdennis/notebooks-cloudflare-workers-ai/main/assets/craig-and-a-burrito.jpg")
            if image_url_class:
                try:
                    resp_class = requests.get(image_url_class, allow_redirects=True)
                    image_data_class = resp_class.content
                    st.image(image_data_class, caption="Input Image", use_column_width=True)
                except Exception as e:
                    st.error(f"Error loading image: {e}")
        else:
            uploaded_class = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="imgclass_upload")
            if uploaded_class:
                image_data_class = uploaded_class.read()
                st.image(image_data_class, caption="Uploaded Image", use_column_width=True)
    if image_data_class and st.button("Classify Image"):
        with st.spinner("Classifying image..."):
            try:
                result = run_model(
                    client,
                    "@cf/microsoft/resnet-50",
                    account_id,
                    image=image_data_class
                )
                st.subheader("Classification Result:")
                st.json(result)
            except Exception as e:
                st.error(f"Error classifying image: {e}")

# === 8. Summarization Tab ===
with tabs[7]:
    st.markdown(f'<div id="{slugify("Summarization")}"></div>', unsafe_allow_html=True)
    st.header("Summarization (@cf/facebook/bart-large-cnn)")
    with st.form("summarization_form"):
        long_text = st.text_area("Enter long text to summarize", height=200,
                                 value="In Congress, July 4, 1776... [Text truncated for brevity]")
        submit_sum = st.form_submit_button("Summarize")
    if submit_sum:
        with st.spinner("Summarizing text..."):
            try:
                result = run_model(
                    client,
                    "@cf/facebook/bart-large-cnn",
                    account_id,
                    input_text=long_text
                )
                st.subheader("Summary:")
                st.write(result.get("summary", "No summary available."))
            except Exception as e:
                st.error(f"Summarization failed: {e}")

# --- Additional Documentation Section ---
with st.expander("Explore the Workers AI API using Python", expanded=False):
    st.markdown(
        """
        **Overview:**
        
        Workers AI lets you run machine learning models on Cloudflare’s network—from Workers or anywhere via a REST API.
        
        **Official Python SDK Example:**
        
        ```python
        import os
        from getpass import getpass
        from cloudflare import Cloudflare
        from IPython.display import display, Image, Markdown, Audio
        import requests
        
        # Load .env variables if available
        %load_ext dotenv
        %dotenv
        
        if "CLOUDFLARE_API_TOKEN" in os.environ:
            api_token = os.environ["CLOUDFLARE_API_TOKEN"]
        else:
            api_token = getpass("Enter your Cloudflare API Token")
        
        if "CLOUDFLARE_ACCOUNT_ID" in os.environ:
            account_id = os.environ["CLOUDFLARE_ACCOUNT_ID"]
        else:
            account_id = getpass("Enter your Cloudflare Account ID")
        
        # Initialize the Cloudflare client
        client = Cloudflare(api_token=api_token)
        
        # Now you can use client.ai.run() to invoke AI models.
        ```
        
        This snippet demonstrates configuring the environment and initializing the client.
        """
    )

# --- Final Status ---
st.success("All tasks completed. Enjoy exploring Cloudflare Workers AI!")
