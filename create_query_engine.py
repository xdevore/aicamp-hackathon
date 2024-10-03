"""
Text files used to train the model have been removed, as they are not public files
"""

import nltk
import ssl
import os
from dotenv import load_dotenv
import openai
from langchain.llms.openai import OpenAIChat
from llama_index import (
    GPTVectorStoreIndex,
    GPTSimpleKeywordTableIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext,
    StorageContext
)
from llama_index.vector_stores import MilvusVectorStore
from milvus import default_server
from llama_index.indices.composability import ComposableGraph
from llama_index.indices.query.query_transform.base import DecomposeQueryTransform
from llama_index.query_engine.transform_query_engine import TransformQueryEngine


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def start_server():

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download("stopwords")

    default_server.start()

    vector_store = MilvusVectorStore(
        host = "127.0.0.1",
        port = default_server.listen_port
    )

    return vector_store


def load_file_docs(txt_file_title_list):

    # Load all wiki documents
    insurance_file_docs = {}
    for txt_file_title in txt_file_title_list:
        insurance_file_docs[txt_file_title] = SimpleDirectoryReader(input_files=[f"./AutoInsuranceTxt/{txt_file_title}"]).load_data()
  
    return insurance_file_docs


def use_llm():
    llm_predictor_chatgpt = LLMPredictor(llm=OpenAIChat(temperature=0, model_name="gpt-3.5-turbo"))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor_chatgpt)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    output = [llm_predictor_chatgpt, service_context, storage_context]

    return output


def build_file_document_index(txt_file_title_list):
    insurance_file_indices = {}
    index_summaries = {}
    for txt_file_title in txt_file_title_list:
        insurance_file_indices[txt_file_title] = GPTVectorStoreIndex.from_documents(insurance_file_docs[txt_file_title], service_context=service_context, storage_context=storage_context)
        # set summary text for file
        index_summaries[txt_file_title] = f"Insurance information about {txt_file_title}"
    
    return insurance_file_indices, index_summaries


def create_graph(insurance_file_indices, index_summaries, llm_predictor_chatgpt):
    graph = ComposableGraph.from_indices(
        GPTSimpleKeywordTableIndex,
        [index for _, index in insurance_file_indices.items()],
        [summary for _, summary in index_summaries.items()],
        max_keywords_per_chunk=50
    )

    decompose_transform = DecomposeQueryTransform(
        llm_predictor_chatgpt, verbose=True
    )

    return graph, decompose_transform


def create_query_engine(insurance_file_indices, service_context, decompose_transform, graph):
    custom_query_engines = {}
    for index in insurance_file_indices.values():
        query_engine = index.as_query_engine(service_context=service_context)
        transform_extra_info = {'index_summary': index.index_struct.summary}
        tranformed_query_engine = TransformQueryEngine(query_engine, decompose_transform, transform_extra_info=transform_extra_info)
        custom_query_engines[index.index_id] = tranformed_query_engine

    custom_query_engines[graph.root_index.index_id] = graph.root_index.as_query_engine(
        retriever_mode='simple',
        response_mode='tree_summarize',
        service_context=service_context
    )

    query_engine_decompose = graph.as_query_engine(
        custom_query_engines=custom_query_engines,)
    
    return query_engine, query_engine_decompose


vector_store = start_server()
txt_file_title_list = os.listdir("./AutoInsuranceTxt")

insurance_file_docs = load_file_docs(txt_file_title_list)
print(insurance_file_docs)

output = use_llm()
llm_predictor_chatgpt = output[0]
service_context = output[1]
storage_context = output[2]

insurance_file_indices, index_summaries = build_file_document_index(txt_file_title_list)

graph, decompose_transform = create_graph(insurance_file_indices=insurance_file_indices, index_summaries=index_summaries, llm_predictor_chatgpt=llm_predictor_chatgpt)

query_engine, query_engine_decompose = create_query_engine(insurance_file_indices=insurance_file_indices, service_context=service_context, decompose_transform=decompose_transform, graph=graph)

def take_prompt(prompt):
    print(prompt)
    response_chatgpt = query_engine.query(prompt)
    return str(response_chatgpt)