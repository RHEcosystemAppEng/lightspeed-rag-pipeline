from kfp import dsl
import kfp
import subprocess
from kfp_tekton import tekton



@dsl.pipeline(
    name="script_python_op"
)
def pipeline():
  tekton_task = TektonTask(
        name='example-tekton-task',
        task_ref='example-task',
        params={'param1': 'value1', 'param2': 'value2'}
    )
    
  run_script_op = dsl.ContainerOp(
        name='evaluation',
        image='quay.io/redhat_emp1/rag-pipeline-tasks',
        command=["python", "evaluation.py"],
        arguments= [    "--provider" , "bam",
                     "--input-persist-dir", "/Users/ilanpinto/dev/ai/rag-pipeline/persist-dir",
                     "--model", "ibm/granite-13b-chat-v2",
                     "-s", "2",
                     "-n", "5",
                     "-qq","/Users/ilanpinto/dev/ai/rag-pipeline/output/openai-gpt-3.5-turbo_no_eval.json"
                    ]
  )
