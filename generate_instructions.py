### FUNCTION TO PROMPT GENERATOR MODEL TO GENERATE INSTRUCTIONS BASED ON PREDEFINED INSTRUCTION GENERATION TEMPLATES
from transformers import pipeline
import torch
import os
from datetime import datetime
import pdb
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
import random
import argparse
from multiprocessing import Pool
import numpy as np
from itertools import combinations
from tqdm import tqdm
import re
from ast import literal_eval
import pandas as pd 
from vllm import LLM, SamplingParams
import time

# Login to HF to access Meta LLAMA model
from huggingface_hub import login
login("")

SAMPLING_PARAMS = SamplingParams() # use default params, match OpenAI params
CHOICES = [2,3,4]

# function to add system prompt to generated instruction template
def llama_ify(prompt):
    res = """<<SYS>>
You are a helpful assistant.
<</SYS>>
[INST]
User:
%s
[/INST]
Assistant:"""%(prompt)
    return res

######################################################################
###### Define methods to sample instruction generation prompts #######
######################################################################
def template_A_0(snippet1, snippet2):
    prompt = """You are a search engine. A person queried something in detail and the most relevant snippets about the query are as follows.
Query: X
Snippets: '%s', '%s'
What could the detailed query X be? Answer with a plausible question or instruction.
X:"""%(snippet1, snippet2)
    return prompt

def template_A_1_0(doc1, doc2, snippet1, snippet2):
    prompt = """'%s', '%s'
You are a search engine. A person queried something in detail about the documents above and the most relevant snippets about the query are as follows.
Query: X
Snippets: '%s', '%s'

What could the detailed query X be? Answer with a plausible question or instruction.
X:"""%(doc1, doc2, snippet1, snippet2)
    return prompt

def template_B_0_3(snippet1, snippet2):
    prompt = """Instruction: X
Snippets: '%s', '%s'

What kind of instruction could these two snippets be the answer to? Your answer should be a specific question that can only be answered by utilizing information in both snippets. Say "Not sure" if you can't come up with a high-quality question. Format your answer as plain text. Before answering, ask yourself if the question you produce covers information in both snippets. If not, regenerate before providing your answer.
X:"""%(snippet1, snippet2)
    return prompt

def template_B_1_3(doc1, doc2, snippet1, snippet2):
    prompt = """'%s', '%s'

Instruction: X
Snippets: '%s', '%s'

What kind of instruction could these two snippets be the answer to? Your answer should be a specific question that can ONLY be answered by utilizing information in BOTH snippets. Say "Not sure" if you can't come up with a high-quality question. Format your answer as plain text.
Read the question again: What kind of instruction could these two snippets be the answer to? Your answer should be a specific question that can ONLY be answered by utilizing information in BOTH snippets. Say "Not sure" if you can't come up with a high-quality question. Format your answer as plain text.
X:"""%(doc1, doc2, snippet1, snippet2)
    return prompt

def template_B_1_4(doc1, doc2, snippet1, snippet2):
    prompt = """'%s', '%s'

Instruction: X
Snippets: '%s', '%s'

What kind of instruction could these two snippets be the answer to? Your answer should be a specific question that can ONLY be answered by utilizing information in BOTH snippets. Say "Not sure" if you can't come up with a high-quality question. Format your answer as plain text.
X:"""%(doc1, doc2, snippet1, snippet2)
    return prompt

def template_D_3(doc1, doc2):
    prompt = """Below are two documents. Select 3 sentences that are most pertinent to the content of the documents and generate a single question or instruction that can only be answered or responded to using ALL 3 sentences.
'%s', '%s'

Your answer should be a single instruction or question that can only be answered using ALL of the snippets you identify. Before providing your proposal, pause and check that EACH snippet is critical to answering the question/instruction. If not, regenerate and check it again. Format your proposal as:

Question/Instruction: 
Snippet 1:
Snippet 2: 
Snippet 3:"""%(doc1, doc2)
    return prompt

def template_D_4(doc1, doc2):
    prompt = """Below are two documents. Select 4 sentences that are most pertinent to the content of the documents and generate a single question or instruction that can only be answered or responded to using ALL 4 sentences.
'%s', '%s'

Your answer should be a single instruction or question that can only be answered using ALL of the snippets you identify. Before providing your proposal, pause and check that EACH snippet is critical to answering the question/instruction. If not, regenerate and check it again. Format your proposal as:

Question/Instruction: 
Snippet 1:
Snippet 2: 
Snippet 3:
Snippet 4:"""%(doc1, doc2)
    return prompt

def template_E_4(cluster):
    prompt = """The documents below are ordered by relevance to a given query (a question or instruction), with the first document being most relevant."""
    for doc in cluster:
        prompt += '\n\n'
        prompt += '\''
        prompt += doc
        prompt += '\''

    prompt += "\n\nGiven the order of the documents from most to least useful to answering the query, what could be the query X?\nX:"
    return prompt

def template_E_6(*args):
    prompt = ""
    for doc in args:
        prompt += '\''
        prompt += doc
        prompt += "\'\n\n"
    prompt += """Select two sentences from the above documents. Generate a query to either compare or contrast the information identified. Format your answer as:

Sentence 1:
Sentence 2:
Query:"""
    return prompt

# save directly
def template_E_5(cluster):
    prompt = "Documents:\n"
    n = random.choice(CHOICES)
    sents = [sent for doc in cluster for sent in doc.split('\n')]
    snips = random.sample(sents, min(n, len(sents)))
    for doc in cluster:
        prompt += '\''
        prompt += doc
        prompt += '\'\n\n'
        prompt += ''

    prompt += "Snippets:\n"
    for snip in snips:
        prompt += '\''
        prompt += snip
        prompt += '\'\n'
    prompt = prompt + "\nAbove is a series of snippets extracted from several  documents given above. How many of the above documents do these snippets come from? Provide your answer as a single number." # -2 to remove last comma and space
    return prompt, len(snips)

# save directly
def template_E_2(doc1, doc2):
    doc1_split = doc1.split('\n')
    doc2_split = doc2.split('\n')
    try:
        doc1_snips = random.sample(doc1_split, random.choice(range(2,min(4,len(doc1_split)))))
    except: 
        doc1_snips = doc1_split[0]
    try:
        doc2_snips = random.sample(doc2_split, random.choice(range(2,min(4,len(doc2_split)))))
    except: 
        doc2_snips = doc2_split[0]
    
    # modify doc1
    for i, pos in enumerate(random.sample(range(len(doc1_split)), min(len(doc2_snips), len(doc1_split)))):
        doc1_split.insert(pos, doc2_snips[i])
    prompt1 = "In the document above, there are 2-4 snippets that do not belong. Which are they?\n\n" + "\n".join(doc1_split)

    # modify doc2
    for i, pos in enumerate(random.sample(range(len(doc2_split)), min(len(doc1_snips), len(doc2_split)))):
        doc2_split.insert(pos, doc1_snips[i])
    prompt2 = "In the document above, there are 2-4 snippets that do not belong. Which are they?\n\n" + "\n".join(doc2_split)
    
    return prompt1, prompt2, len(doc2_snips), len(doc1_snips)


# Function to generate and save instruction tuning prompts/examples
def generate_and_save_instructions(
    input_dir,
    output_dir,
    model_name,
    instruction_format,
    use_existing_prompts,
    given_prompt_dir,
    start_instr_file_idx
):
    # select specified generator model
    if model_name=="llama2-chat-7b":
        tokenizer_kwargs = {"truncation": True, "max_length":4096}
        generator_model = pipeline(
            "text-generation", 
            model="meta-llama/Llama-2-7b-chat-hf",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            temperature=1, 
            top_p=1,
            max_new_tokens=512,
            # truncate=True,
            )
    
    # ensure output dir exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Access all the newshead file addresses (train/valid/test split .pt files)
    all_files = [
        f
        for f in os.listdir(input_dir)
        if f.endswith(".pt")
    ]
    all_files = all_files[::-1] # reverse order of files 

    # For each data split (train/valid/test)
    for file_idx, file_name in enumerate(tqdm(all_files)):
        print(file_name)
        num_prompts=0

        # prepare directories to save outputs
        all_prompts = []
        source_docs = []
        num_snips = []
        prompt_dir = os.path.join(output_dir, "prompts")
        if not os.path.exists(prompt_dir):
            os.makedirs(prompt_dir)
        docs_dir = os.path.join(output_dir, "source_docs")
        if not os.path.exists(docs_dir):
            os.makedirs(docs_dir)
        
        # set up for saving answers if needed
        if instruction_format=="E_2" or instruction_format=="E_5":
            num_snips_dir = os.path.join(output_dir, "num_snips")
            if not os.path.exists(num_snips_dir):
                os.makedirs(num_snips_dir)

        # if prompts haven't been created yet
        if not use_existing_prompts:
            if instruction_format in {"A_0", "A_1_0", "B_0_3", "B_1_3", "B_1_4"}:
                base_snippet_pairs_path = "./base_snippet_pairs/" + file_name.replace('.pt', '.csv')
                selected_sent_pairs = pd.read_csv(base_snippet_pairs_path, index_col=None, converters={"articles": literal_eval,"answer": literal_eval})

                all_source_docs = selected_sent_pairs['articles'].tolist()
                all_answers = selected_sent_pairs['answer'].tolist()
                file_idx = 0

                # for each document pair and selected snippet pair, process according to the specified instruction template(s)
                for docs, snippets in zip(all_source_docs, all_answers):
                    if instruction_format=="A_0":
                        prompt = template_A_0(snippets[0], snippets[1])
                        source_docs.append([docs[0], docs[1]])
                    elif instruction_format=="A_1_0":
                        prompt = template_A_1_0(docs[0], docs[1], snippets[0], snippets[1])
                    elif instruction_format=="B_0_3":
                        prompt = template_B_0_3(snippets[0], snippets[1])
                        source_docs.append((docs[0], docs[1]))
                    elif instruction_format=="B_1_3":
                        prompt = template_B_1_3(docs[0], docs[1], snippets[0], snippets[1])
                    elif instruction_format=="B_1_4":
                        prompt = template_B_1_3(docs[0], docs[1], snippets[0], snippets[1])
                    all_prompts.append(llama_ify(prompt))
                    num_prompts += 1
                    
                    # if too many prompts have been processed, save and dump to avoid overloading memory
                    if num_prompts > 4:
                        prompt_file_name = file_name[:-2]+"%d.pt"%(file_idx)
                        torch.save(all_prompts, os.path.join(prompt_dir, prompt_file_name))
                        print("Finished saving prompt slice %d at %s"%(file_idx, prompt_dir+"/"+prompt_file_name))

                        if instruction_format=="A_0" or instruction_format=="B_0_3":
                            source_docs_file_name = file_name[:-2]+"%d.pt"%(file_idx)
                            torch.save(source_docs, os.path.join(docs_dir, source_docs_file_name))
                            print("Finished saving source documents slice %d at %s"%(file_idx, docs_dir+"/"+source_docs_file_name))
                            source_docs = []

                        print("TOTAL NUMBER OF PROMPTS:",num_prompts)
                        num_prompts = 0
                        all_prompts = []
                        file_idx += 1

            elif instruction_format in {"D_3", "D_4", "E_4", "E_5", "E_6"}:
                all_clusters = torch.load(os.path.join(input_dir, file_name))

                count = 0
                file_idx = 0
                for cluster in all_clusters:
                    print("On cluster",count)

                    if instruction_format=="E_4":
                        prompt = template_E_4(cluster)
                        all_prompts.append(llama_ify(prompt))
                        num_prompts += 1
                    elif instruction_format=="E_5":
                        prompt, snip_count = template_E_5(cluster)
                        all_prompts.append(llama_ify(prompt))
                        num_snips.append(snip_count)
                        num_prompts += 1
                    else:
                        doc_idx_combos = list(combinations(range(len(cluster)), 2))

                        for idx1, idx2 in doc_idx_combos:
                            if instruction_format=="D_3":
                                prompt = template_D_3(cluster[idx1], cluster[idx2])
                            elif instruction_format=="D_4":
                                prompt = template_D_4(cluster[idx1], cluster[idx2])
                            elif instruction_format=="E_6":
                                prompt = template_E_6(cluster[idx1], cluster[idx2])
                            all_prompts.append(llama_ify(prompt))
                        num_prompts += len(doc_idx_combos)
                    
                    # if too many prompts have been processed, save and dump to avoid overloading memory
                    if num_prompts > 4:
                        prompt_file_name = file_name[:-2]+"%d.pt"%(file_idx)
                        torch.save(all_prompts, os.path.join(prompt_dir, prompt_file_name))
                        print("Finished saving prompt slice %d at %s"%(file_idx, prompt_dir+"/"+prompt_file_name))

                        if instruction_format=="E_5":
                            num_snips_file_name = file_name[:-2]+"%d.pt"%(file_idx)
                            torch.save(num_snips, os.path.join(num_snips_dir, num_snips_file_name))
                            print("Finished saving snippet counts slice %d at %s"%(file_idx, num_snips_dir+"/"+num_snips_file_name))

                        print("TOTAL NUMBER OF PROMPTS:",num_prompts)
                        num_prompts = 0
                        all_prompts = []
                        file_idx += 1
                    
                    count += 1

            elif instruction_format=="E_2":
                # Take two documents from different clusters. In the longer one, every few sentences, place a sentence from the other document. Ask model: which sentences do not belong?
                all_clusters = torch.load(os.path.join(input_dir, file_name))

                doc_pairs = list(set((doc1, doc2) for cluster1, cluster2 in zip(all_clusters, all_clusters[1:]) for doc1 in cluster1 for doc2 in cluster2))

                file_id = 0
                for doc1, doc2 in doc_pairs:
                    prompt1, prompt2, snip_count1, snip_count2 = template_E_2(doc1, doc2)
                    all_prompts.append(llama_ify(prompt1))
                    all_prompts.append(llama_ify(prompt2))
                    num_snips.append(snip_count1)
                    num_snips.append(snip_count2)
                    num_prompts += 2

                    # if too many prompts have been processed, save and dump to avoid overloading memory
                    if num_prompts > 1999:
                        prompt_file_name = file_name[:-2]+"%d.pt"%(file_id)
                        torch.save(all_prompts, os.path.join(prompt_dir, prompt_file_name))
                        print("Finished saving prompt slice %d at %s"%(file_id, prompt_dir+"/"+prompt_file_name))

                        num_snips_file_name = prompt_file_name
                        torch.save(num_snips, os.path.join(num_snips_dir, num_snips_file_name))
                        print("Finished saving snippet counts slice %d at %s"%(file_id, num_snips_dir+"/"+num_snips_file_name))

                        num_prompts = 0
                        all_prompts = []
                        file_id += 1
                
            torch.save(all_prompts, os.path.join(prompt_dir, file_name))
            print("Finished saving remaining prompts at %s"%(prompt_dir+"/"+file_name))

            if instruction_format=="A_0" or instruction_format=="B_0_3":
                torch.save(source_docs, os.path.join(docs_dir, file_name))
                print("Finished saving remaining source documents at %s"%(docs_dir+"/"+file_name))

            if instruction_format=="E_5" or instruction_format=="E_2":
                torch.save(num_snips, os.path.join(num_snips_dir, file_name))
                print("Finished saving remaining snippet counts at %s"%(num_snips_dir+"/"+file_name))

            print("TOTAL NUMBER OF PROMPTS:",num_prompts)
        
        ##################################################
        ########## Generate LLM-Created Prompts ##########
        ##################################################
        if instruction_format!="E_2" and instruction_format!="E_5":
            
            if use_existing_prompts:
                prompt_dir = given_prompt_dir

            # ensure output directory exists
            instr_dir = os.path.join(output_dir, "instructions")
            if not os.path.exists(instr_dir):
                os.makedirs(instr_dir)
        
            all_files = sorted([
                f
                for f in os.listdir(prompt_dir)
                if f.endswith(".pt")
            ])

            # For each data split (train/valid/test)
            for file_idx, pt_file in enumerate(tqdm(all_files)):
                                  
                if start_instr_file_idx!=-1 and file_idx < start_instr_file_idx:
                    pass
                
                elif (start_instr_file_idx==-1) or (start_instr_file_idx!=-1 and file_idx >= start_instr_file_idx): 
                    ## USING PIPELINE
                    print("On slice", file_idx)
                    cleaned_instructions = []
                    prompt_slice = torch.load(os.path.join(prompt_dir, pt_file))
                    instructions = generator_model(prompt_slice,tokenizer_kwargs=tokenizer_kwargs)
                    cleaned_instructions = [instruction[0]['generated_text'].removeprefix(prompt) for instruction, prompt in zip(instructions, prompt_slice)]

                    instr_file_name = pt_file#[:-2]+"%d.pt"%(i)

                    torch.save(cleaned_instructions, os.path.join(instr_dir, instr_file_name))
                    print("Finished saving cleaned instructions part %d at %s"%(file_idx,  instr_dir+"/"+instr_file_name))

        print("done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./base_articles",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generated_instructions/style_",
    )
    parser.add_argument(
        '--use_existing_prompts', 
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "--given_prompt_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--start_instr_file_idx",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--instruction_format",
        choices=[
            "A_0",
            "A_1_0",
            "B_0_3",
            "B_1_3",
            "B_1_4",
            "D_3",
            "D_4",
            "E_2",
            "E_3",
            "E_4",
            "E_5",
            "E_6",
        ],
        default="A_0",
        type=str,
    )
    parser.add_argument(
        "--model_name",
        choices=[
            "llama2-chat-7b",
            "llama2-chat-13b",
        ],
        default="llama2-chat-7b",
        type=str,
    )
    args = parser.parse_args()
    print(args)

    current_date_time = str(datetime.now().strftime("%Y-%m-%d_hr%H-min%M"))
    output_dir = args.output_dir+args.instruction_format+"/model_"+args.model_name+"/"+current_date_time
    
    generate_and_save_instructions(
        args.input_dir,
        output_dir,
        args.model_name,
        args.instruction_format,
        args.use_existing_prompts,
        args.given_prompt_dir,
        args.start_instr_file_idx,
    )

