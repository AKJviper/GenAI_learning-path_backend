import logging
import os
import subprocess
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from langchain.chains import RetrievalQA
from langchain.embeddings import  HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from template import get_prompt_template
from langchain.vectorstores import Chroma
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
from constants import (
    MODEL_ID,
    MODEL_BASENAME,
    CHROMA_SETTINGS,
    MODEL_ID, MODEL_BASENAME, MAX_NEW_TOKENS
)
from load_models import (
    load_quantized_model_gguf_ggml,
    load_full_model,
)
from transformers import (
    GenerationConfig,
    pipeline,
)

DEVICE_TYPE = "cpu"
SHOW_SOURCES = True
DOCUMENT_CAT = "JAVA"
EMBEDDINGS_MODEL_NAME = "all-MiniLM-L6-v2"
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB/{DOCUMENT_CAT}"
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
prompt, memory = get_prompt_template(promptTemplate_type="llama", history=False)

# load the llm pipeline

device_type ="cpu"
model_type = "llama"
def load_model(device_type, model_id, model_basename=None, LOGGING=logging):
    if model_basename is not None:
        if ".gguf" in model_basename.lower():
            llm = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
            return llm
        elif ".ggml" in model_basename.lower():
            model, tokenizer = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
    else:
        model, tokenizer = load_full_model(model_id, model_basename, device_type, LOGGING)
    generation_config = GenerationConfig.from_pretrained(model_id)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=MAX_NEW_TOKENS,
        temperature=0.3,
        # top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)

    return local_llm
llm = load_model(device_type, model_id=MODEL_ID, model_basename=MODEL_BASENAME, LOGGING=logging)
def retrieve(cat="JAVA"):
    PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB/{cat}"
    db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
            client_settings=CHROMA_SETTINGS
        )
    retriever = db.as_retriever()
    global qar
    qar = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
        retriever=retriever,
        # return_source_documents=True,  # verbose=True,
        callbacks=callback_manager,
        chain_type_kwargs={
            "prompt": prompt,
        },
    )
    return qar
qar = retrieve(DOCUMENT_CAT)
app = Flask(__name__) 
CORS(app, support_credentials=True)
# @app.route("/api/run_ingestion", methods=["GET"])
# def run_ingest_route():
#     global DB
#     global RETRIEVER
#     global QA
#     try:
#         run_langest_commands = ["python", "ingestion.py"]
#         result = subprocess.run(run_langest_commands, capture_output=True)
#         if result.returncode != 0:
#             return "Script execution failed: {}".format(result.stderr.decode("utf-8")), 500
#         # load the vectorstore
#         DB = Chroma(
#             persist_directory=PERSIST_DIRECTORY,
#             embedding_function=embeddings,
#             client_settings=CHROMA_SETTINGS,
#         )
#         RETRIEVER = DB.as_retriever()
#         prompt, memory = get_prompt_template(promptTemplate_type="llama", history=False)

#         QA = RetrievalQA.from_chain_type(
#             llm=llm,
#             chain_type="stuff",
#             retriever=RETRIEVER,
#             return_source_documents=SHOW_SOURCES,
#             chain_type_kwargs={
#                 "prompt": prompt,
#             },
#         )
#         return "Script executed successfully: {}".format(result.stdout.decode("utf-8")), 200
#     except Exception as e:
#         return f"Error occurred: {str(e)}", 500


    
@app.route('/startqna', methods = ['GET']) 
@cross_origin(supports_credentials=True)
def home(): 
    if(request.method == 'GET'): 
        subprocess.run(["streamlit","run","qna.py"])
        
@app.route("/api/assign_doc", methods=["POST"])
@cross_origin(supports_credentials=True)
def assign_doc():
    if(request.method == 'POST'):
        global DOCUMENT_CAT
        DOCUMENT_CAT = request.json['userData']['category']
        print(DOCUMENT_CAT)
        global qar
        qar = retrieve(DOCUMENT_CAT)
        return "Loaded successfully", 200
    return "Error", 500


@app.route("/api/prompt_route", methods=['POST'])
@cross_origin(supports_credentials=True)
def prompt_route():
    if(request.method== 'POST'):
        learning_goal = request.json['userData']['learning_goal']
        technical_interest = request.json['userData']['technical_interest']
        user_experience = request.json['userData']['user_experience']
        if learning_goal:
            question = 'What is the learning path for ' + learning_goal+' as I have an experience of ' + user_experience + 'year in ' + technical_interest 
            res = qar(question)
            
            answer= res["result"]
            prompt_response_dict = {
                "Prompt": question,
                "Answer": answer
            }

            # prompt_response_dict["Sources"] = []
            # for document in docs:
            #     prompt_response_dict["Sources"].append(
            #         (os.path.basename(str(document.metadata["source"])), str(document.page_content))
            #     )

            return jsonify(prompt_response_dict), 200
        else:
            return "No user prompt received", 400
    return "Error", 500


@app.route("/api/qna", methods=["POST"])
@cross_origin(supports_credentials=True)
def qna_api():
    if(request.method== 'POST'):
        user_prompt = request.json['userData']['question']
        if user_prompt:
            user_prompt = user_prompt + ' answer in less than 100 words'
            res = qar(user_prompt)
            answer = res["result"]

            prompt_response_dict = {
                "Prompt": user_prompt,
                "Answer": answer,
            }
            return jsonify(prompt_response_dict), 200
        else:
            return "No user prompt received", 400
if __name__ == "__main__":
    app.run(debug=False, port=5110)

# what is the learning path for JAVA as I have an experience of 4 year in JAVA developement