


import argparse
import asyncio
import json
import time
import nest_asyncio
from llama_index.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator,
    BatchEvalRunner,
    DatasetGenerator,
    RelevancyEvaluator
)
from llama_index import ServiceContext, SimpleDirectoryReader, StorageContext, load_index_from_storage
from images.rag.llms.llm_loader import LLMLoader

# Constant
PRODUCT_INDEX = "product"
PRODUCT_DOCS_PERSIST_DIR = "./vector-db/ocp-product-docs"
SUMMARY_INDEX = "summary"
SUMMARY_DOCS_PERSIST_DIR = "./vector-db/summary-docs"

def get_eval_results(key, eval_results):
    results = eval_results[key]
    correct = 0
    for result in results:
        if result.passing:
            correct += 1
    score = correct / len(results)
    print(f"{key} Score: {score}")
    return score

async def main(): 
    start_time = time.time()
    # collect args 
    parser = argparse.ArgumentParser(
        description="evaluation cli for task execution"
    )
    parser.add_argument("-p", "--provider",   default="bam", help="LLM provider supported value: bam, openai")
    parser.add_argument("-m", "--model",   default="local:BAAI/bge-base-en", help="the valid models are:\
                                                                    - ibm/granite-13b-chat-v1, ibm/granite-13b-chat-v2, ibm/granite-20b-code-instruct-v1 for bam \
                                                                    - gpt-3.5-turbo-1106, gpt-3.5-turbo for openai"
                        )

    
    parser.add_argument("-a", "--auth", help="Authentication headers per vectorDB requirements")    
    parser.add_argument("-n", "--collection-name", help="Collection name in vector DB")
    parser.add_argument("-f", "--folder", help="Plain text folder path")
    parser.add_argument("-e", "--include-evaluation",   default="True", help="preform evaluation [True/False]")
    parser.add_argument("-q", "--question-folder",   default="", help="docs folder for questions gen")

    parser.add_argument("-o", "--output", help="persist folder")


    # execute 
    args = parser.parse_args() 
    
    PERSIST_FOLDER = args.output
    
    print("** settings params")
        
    embed_model = "local:BAAI/bge-base-en"    
    bare_llm = LLMLoader(args.provider, args.model).llm
    
    service_context = ServiceContext.from_defaults(
            chunk_size=1024, llm=bare_llm, embed_model=embed_model
        )
    
    print("** settings context")
    storage_context = StorageContext.from_defaults(
    persist_dir=PRODUCT_DOCS_PERSIST_DIR) 

    print("** load embeddings evaluating")

    index = load_index_from_storage(
        storage_context=storage_context,
        index_id=PRODUCT_INDEX,
        service_context=service_context       
    )
    
    print("** starting model evaluating")
    nest_asyncio.apply()
    
    print("*** generating questions ")        
    question_folder = args.folder if args.question_folder is None else args.question_folder
    
    reader = SimpleDirectoryReader(question_folder)
    question = reader.load_data()
    data_generator = DatasetGenerator.from_documents(question)
    eval_questions = data_generator.generate_questions_from_nodes(num=5)
    # engine = index.as_query_engine(similarity_top_k=1, service_context=service_context)
    
    print( eval_questions)
    
    print("*** start evaluation")
    faithfulness = FaithfulnessEvaluator(service_context=service_context)
    relevancy = RelevancyEvaluator(service_context=service_context)
    correctness = CorrectnessEvaluator(service_context=service_context)

    runner = BatchEvalRunner(
        {"faithfulness": faithfulness, "relevancy": relevancy , "correctness": correctness },
        workers=10, show_progress=True
    )
    
    eval_results = await runner.aevaluate_queries( index.as_query_engine(similarity_top_k=1, \
                                                                        service_context=service_context), \
                                                                        queries=eval_questions ) 
    
    evaluation_results = {}
    evaluation_results["faithfulness"] = get_eval_results("faithfulness", eval_results)
    evaluation_results["relevancy"] = get_eval_results("relevancy", eval_results)
    evaluation_results["correctness"] = get_eval_results("correctness", eval_results)

    end_time = time.time()
    execution_time_seconds = end_time - start_time
    
    print(f"** Total execution time in seconds: {execution_time_seconds}")
    
    # creating metadata folder 
    metadata = {} 
    
    metadata["execution-time"] = execution_time_seconds
    metadata["llm"] = args.provider
    metadata["embedding-model"] = args.model 
    metadata["index_id"] = PRODUCT_INDEX

    metadata["eval_questions"] = eval_questions
    metadata["evaluation_results"] = evaluation_results
    json_metadata = json.dumps(metadata)

    # Write the JSON data to a file
    file_path = f"{PERSIST_FOLDER}/{args.provider}_metadata.json"
    with open(file_path, 'w') as file:
        file.write(json_metadata)
    
    # Convert JSON data to markdown 
    markdown_content = "```markdown\n"
    for key, value in metadata.items():
        markdown_content += f"- {key}: {value}\n"
    markdown_content += "```" 
    
    file_path = f"{PERSIST_FOLDER}/{args.provider}_metadata.md"
    with open(file_path, 'w') as file:
        file.write(markdown_content)


    return "Completed"
    
asyncio.run(main())