


import argparse
import asyncio
import json
import os
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
from llms.llm_loader import LLMLoader


def list_all_files(folder):
    all_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files

def dirs_all_files(folder):
    all_dirs = []
    for root, dirs, files in os.walk(folder):
        for dir in dirs:
            if len(files) > 0: 
                all_dirs.append(os.path.join(root,dir))
    return all_dirs


def eval_parser(eval_response: str) :
    """
    Default parser function for evaluation response.

    Args:
        eval_response (str): The response string from the evaluation.

    Returns:
        Tuple[float, str]: A tuple containing the score as a float and the reasoning as a string.
    """
    
    print("eval_response:", eval_response)
    
    eval_response_parsed = eval_response.split("\n")
    eval_len =len (eval_response_parsed)
    
    if eval_len == 0 :
        return 0, eval_response_parsed
    if eval_len == 1 :
        return 0, eval_response_parsed 
    if eval_len == 2 : 
        score_str =  eval_response_parsed[0]
        score = float(score_str) if score_str else 0 
        reasoning = eval_response_parsed[1]
        return score, reasoning
        
    if eval_len == 3 : 
        score_str, reasoning_str =  eval_response_parsed[1], eval_response_parsed[2]    
        score = float(score_str) if score_str else 0 
        reasoning = reasoning_str.lstrip("\n")
        return score, reasoning
    if eval_len > 3 : 
        return 0, eval_response

    


def get_eval_results(key, eval_results):
    results = eval_results[key]
    correct = 0
    for result in results:
        if result.passing:
            correct += 1
    score = correct / len(results)
    print(f"{key} Score: {score}")
    return score

def get_eval_breakdown(key, eval_results):
    results = eval_results[key] 
    response = [] 
    for result in results:       
        res = {} 
        res["query"] = None if result.query is None else result.query
        res["response"] = result.response
        res["score"] = result.score
        res["passing"] = result.passing
        res["invalid_reason"] = result.invalid_reason
        res["feedback"] = result.feedback
        response.append(res)
    return response
    
    


#   python evaluation.py --provider ${PROVIDER} \
                            # ---input-persist-dir ${input-persist-dir}
                            # --auth ${HEADERS} \
                            # --model ${MODEL_NAME} \
                            # --question-folder ${question-folder} \
                            # --collection-name ${COLLECTION_NAME} 
                            # -o ${output} -e ${INCLUDE_EVALUATION} --question-folder ${QUESTION_FOLDER}


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
    
    parser.add_argument("-x", "--product-index" ,default="product" , help="storage product index")
    parser.add_argument("-i", "--input-persist-dir" , help="path to persist file dir")
    
    
    parser.add_argument("-q", "--question-main-folder",   default="", help="docs folder for questions gen")
    parser.add_argument("-n", "--number-of-questions" , default="5" , help="number of questions per file for evaluation")
    parser.add_argument("-s", "--similarity" , default="5" , help="similarity_top_k")

    parser.add_argument("-c", "--chunk",   default="1024", help="chunk size for embedding")
    parser.add_argument("-l", "--overlap",   default="20", help="chunk overlap for embedding")

    
    parser.add_argument("-o", "--output", help="persist folder")
    # parser.add_argument("-n", "--collection-name" , help="Collection name in vector DB")

    # execute 
    args = parser.parse_args() 
    
    PERSIST_FOLDER = args.output
    PRODUCT_INDEX=args.product_index
    PRODUCT_DOCS_PERSIST_DIR = args.input_persist_dir
    NUM_OF_QUESTIONS=int(args.number_of_questions)
    SIMILARITY=int(args.similarity)
    CHUNK_SIZE=int(args.chunk)
    CHUNK_OVERLAP=int(args.overlap)
    
    print("** settings params")
        
    embed_model = "local:BAAI/bge-base-en"    
    bare_llm = LLMLoader(args.provider, args.model).llm
    
    service_context = ServiceContext.from_defaults(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, llm=bare_llm, embed_model=embed_model
        )
    
    print("** settings context")
    storage_context = StorageContext.from_defaults( persist_dir=PRODUCT_DOCS_PERSIST_DIR) 

    print("** load embeddings evaluating")

    index = load_index_from_storage(
        storage_context=storage_context,
        index_id=PRODUCT_INDEX,
        service_context=service_context       
    )
    

    nest_asyncio.apply()
    
    print("*** generating questions ")        
    question_folder = args.folder if args.question_main_folder is None else args.question_main_folder
    
    dir_list = dirs_all_files(question_folder)
    
    print("*** starting question iteration ")
    
    full_results = []
    for dir in dir_list:         
        print(f"gen questions for: {dir}") 

        results = {}
        reader = SimpleDirectoryReader(dir)
        question = reader.load_data()
        data_generator = DatasetGenerator.from_documents(question)
        eval_questions = data_generator.generate_questions_from_nodes(num=NUM_OF_QUESTIONS)
        
        results["dir_name"] = dir
        results["questions"] = eval_questions
    
        print( eval_questions)
    
        print("*** start evaluation")
        faithfulness = FaithfulnessEvaluator(service_context=service_context)
        relevancy = RelevancyEvaluator(service_context=service_context)
        correctness = CorrectnessEvaluator(service_context=service_context,score_threshold=2.0 ,parser_function=eval_parser)
    
    


        runner = BatchEvalRunner(
        { "faithfulness": faithfulness, "relevancy": relevancy ,
            
        },
            workers=100, show_progress=True
        )
    
        eval_results = await runner.aevaluate_queries( index.as_query_engine(similarity_top_k=SIMILARITY, \
                                                                            service_context=service_context), \
                                                                        queries=eval_questions ) 
    
         
        results["faithfulness"] = get_eval_breakdown("faithfulness", eval_results)
        results["relevancy"] = get_eval_breakdown("relevancy", eval_results) 
        
        end_time = time.time()
        execution_time_seconds = end_time - start_time    
        print(f"** completed faithfulness,relevancy evaluation: execution time in min: {execution_time_seconds/60}")
    
        # correcntess
    
        engine = index.as_query_engine(similarity_top_k=SIMILARITY, service_context=service_context)
        
        res_table = []
        for query in eval_questions: 
            
            res_row ={}
                    
            summary = engine.query(query)
            referenced_documents = "\n".join(
                [
                    source_node.node.metadata["file_name"]
                    for source_node in summary.source_nodes
                ]
            )
            
            result =correctness.evaluate(
                query=query,
                response=summary.response,
                reference=summary.source_nodes[0].text,
                )
            
            res_row["query"] = query
            res_row["response"] = summary.response
            res_row["ref"] = referenced_documents 
            res_row["ref_doc"] = summary.source_nodes[0].text
            res_row["ref_doc_score"] = summary.source_nodes[0].score
            res_row["passing"] = result.passing
            res_row["feedback"] = result.feedback
            res_row["score"] = result.score

            res_table.append(res_row)

            results["correctness"] = res_table
            
            end_time = time.time()
            execution_time_seconds = end_time - start_time
            print(f"*** Completed correctness evaluation: execution time in min: {execution_time_seconds/60}")

            full_results.append(results)
            
        # break
            
    end_time = time.time()
    execution_time_seconds = end_time - start_time
    print(f"** Total execution time in min: {execution_time_seconds/60}")
    
    # creating metadata folder 
    metadata = {} 
    
    metadata["execution-time-MIN"] = execution_time_seconds
    metadata["llm"] = args.provider
    metadata["model"] = args.model 
    metadata["index-id"] = PRODUCT_INDEX
    metadata["evaluation-results"] = full_results
    json_metadata = json.dumps(metadata)
    
    if not os.path.exists(PERSIST_FOLDER):
        os.makedirs(PERSIST_FOLDER) 

    model_name_formatted = args.model.replace("/","_")

    # Write the JSON data to a file
    file_path = f"{PERSIST_FOLDER}/{args.provider}-{model_name_formatted}_metadata.json"
    with open(file_path, 'w') as file:
        file.write(json_metadata)
        
    
    full_results_markdown_content = "\n***\n"
    for res in full_results: 
        for key, value in res.items():
            if isinstance(value, list) and type(value[0]) == dict :
                for d in value : 
                    new ="\n"
                    for k,v in d.items():
                        new += f"   - {k}: {v}\n"
                value = new
            full_results_markdown_content += f"- {key}: {value}\n"
    full_results_markdown_content += "***"     
    
    metadata["evaluation-results"] = full_results_markdown_content

    
    # Convert JSON data to markdown 
    markdown_content = "```markdown\n"
    for key, value in metadata.items():
        markdown_content += f"- {key}: {value}\n"
    markdown_content += "```" 
    

    
    
    file_path = f"{PERSIST_FOLDER}/{args.provider}-{model_name_formatted}_metadata.md"
    with open(file_path, 'w') as file:
        file.write(markdown_content)


    return "Completed"
    
asyncio.run(main())