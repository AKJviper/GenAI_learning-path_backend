import os
from chromadb.config import Settings
from langchain.document_loaders import UnstructuredPDFLoader
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
Category = "PYTHON"
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/SOURCE/{Category}"

PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB/{Category}"

MODELS_PATH = "./models"

INGEST_THREADS = os.cpu_count() or 8

CHROMA_SETTINGS = Settings(
        persist_directory=PERSIST_DIRECTORY,
        anonymized_telemetry=False,
        is_persistent=True,
)

CONTEXT_WINDOW_SIZE = 4096
MAX_NEW_TOKENS = CONTEXT_WINDOW_SIZE/4 # int(CONTEXT_WINDOW_SIZE/4)


N_GPU_LAYERS = 100  # Llama-2-70B has 83 layers
N_BATCH = 512
DOCUMENT_MAP = {
    ".pdf": UnstructuredPDFLoader,
}
# Default Instructor Model
# EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"  # Uses 1.5 GB of VRAM (High Accuracy with lower VRAM usage)



# EMBEDDING_MODEL_NAME = "hkunlp/instructor-xl" # Uses 5 GB of VRAM (Most Accurate of all models)
# EMBEDDING_MODEL_NAME = "intfloat/e5-large-v2" # Uses 1.5 GB of VRAM (A little less accurate than instructor-large)
# EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2" # Uses 0.5 GB of VRAM (A good model for lower VRAM GPUs)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # Uses 0.2 GB of VRAM (Less accurate but fastest - only requires 150mb of vram)

####
#### MULTILINGUAL EMBEDDING MODELS
####

# EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large" # Uses 2.5 GB of VRAM
# EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base" # Uses 1.2 GB of VRAM

# MODEL_ID = "TheBloke/Llama-2-7B-Chat-GGML"
# MODEL_BASENAME = "llama-2-7b-chat.ggmlv3.q4_0.bin"


# MODEL_ID = "TheBloke/Llama-2-13b-Chat-GGUF"
# MODEL_BASENAME = "llama-2-13b-chat.Q4_K_M.gguf"

MODEL_ID = "TheBloke/Llama-2-7b-Chat-GGUF"
MODEL_BASENAME = "llama-2-7b-chat.Q4_K_M.gguf"
# MODEL_BASENAME = "llama-2-7b-chat.Q2_K.gguf"
# MODEL_ID = "TheBloke/Llama-2-7B-Chat-GGML"
# MODEL_BASENAME = "llama-2-7b-chat.ggmlv3.q3_K_M.bin	"

# MODEL_ID="TheBloke/Llama-2-7b-Chat-GGUF"
# MODEL_BASENAME="llama-2-7b-chat.Q2_K.gguf"

# MODEL_ID = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
# MODEL_BASENAME = "mistral-7b-instruct-v0.1.Q8_0.gguf"

# MODEL_ID = "TheBloke/Vigogne-2-7B-Chat-GGUF"
# MODEL_BASENAME = "vigogne-2-7b-chat.Q4_K_M.gguf"

# MODEL_ID = "TheBloke/Xwin-LM-7B-V0.2-GGUF"
# MODEL_BASENAME = "xwin-lm-7b-v0.2.Q4_K_M.gguf"

# MODEL_ID = "TheBloke/Llama-2-70b-Chat-GGUF"
# MODEL_BASENAME = "llama-2-70b-chat.Q4_K_M.gguf"

### 7b GPTQ Models for 8GB GPUs
# MODEL_ID = "TheBloke/Wizard-Vicuna-7B-Uncensored-GPTQ"
# MODEL_BASENAME = "Wizard-Vicuna-7B-Uncensored-GPTQ-4bit-128g.no-act.order.safetensors"
# MODEL_ID = "TheBloke/WizardLM-7B-uncensored-GPTQ"
# MODEL_BASENAME = "WizardLM-7B-uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors"
# MODEL_ID = "TheBloke/wizardLM-7B-GPTQ"
# MODEL_BASENAME = "wizardLM-7B-GPTQ-4bit.compat.no-act-order.safetensors"