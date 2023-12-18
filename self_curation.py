### CURATES THE DATA: For each of train/test/val splits, saves the data for each as a single json file with one json object per line.
### NOTE: NO NEED TO RUN THIS FOR TEMPLATES E_2 and E_5

import argparse
from transformers import pipeline
from tqdm import tqdm
import pandas as pd
import re
import random
import os
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModel
import json

# Login to HF to access Meta LLAMA model
from huggingface_hub import login
login("")

# add system prompt for llama if needed
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

# define string templates to be used in self-curation
CURATION_PREFIX = """Below is an Instruction from an user and a candidate Response. Evaluate whether or not the Response is a good example of how AI Assistant should respond to the user's instruction. Please assign a score using the following 5-point scale:
1: It means the Response is incomplete, vague, off-topic, controversial, or not exactly what the user asked for. For example, some content seems missing, numbered list does not start from the beginning, the opening sentence repeats user's question. Or the response is from another person's perspective with their personal experience (e.g. taken from blog posts), or looks like an answer from a forum. Or it contains promotional text, navigation text, or other irrelevant information.
2: It means the Response addresses most of the asks from the user. It does not directly address the user's question. For example, it only provides a high-level methodology instead of the exact solution to user's question.
3: It means the Response is helpful but not written by an AI Assistant. It addresses all the basic asks from the user. It is complete and self contained with the drawback that the response is not written from an AI assistant's perspective, but from other people's perspective. The content looks like an excerpt from a blog post, web page, or web search results. For example, it contains personal experience or opinion, mentions comments section, or share on social media, etc.
4: It means the Response is written from an AI assistant's perspective with a clear focus of addressing the instruction. It provide a complete, clear, and comprehensive response to user's question or instruction without missing or irrelevant information. It is well organized, self-contained, and written in a helpful tone. It has minor room for improvement, e.g. more concise and focused.
5: It means it is a perfect Response from an AI Assistant. It has a clear focus on being a helpful AI Assistant, where the response looks like intentionally written to address the user's question or instruction without any irrelevant sentences. The Response provides high quality content, demonstrating expert knowledge in the area, is very well written, logical, easy-to-follow, engaging and insightful. 

Please first provide a brief reasoning you used to derive the rating score, and then write "Score: <rating>" in the last line.

"""
REGEX = re.compile(r"[Ss]core:\s*(\d+)")

LENGTH_ENHANCEMENTS = ["\nRespond in 2 sentences.", "\nRespond briefly."]

# IF GENERATOR MODEL WAS LLAMA
SYS_PREFIX = "<<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n[INST]\nUser:\n"
SYS_SUFFIX = "\n[/INST]\nAssistant:"

# UPDATE SYS PREFIX/SUFFIX AS NEEDED IF GENERATOR IS ****NOT**** LLAMA

# extract the generated instruction/query from the generator model's response
def get_query(instr_style: str,
              instr: str,
              ):

    if instr_style=="A_0" or instr_style=="A_1_0" or instr_style=="B_0_3" or instr_style=="B_1_3" or instr_style=="B_1_4":
        instr = instr.replace('"X"',"")
        pattern = r'"([^"]*?[^X"]|\S*?\?)"'
        try:
            query = re.findall(pattern, instr)[0]
        except:
            pattern = r'^([^!?.]*\?[^\S]*)'
            try:
                query = re.search(pattern, instr).group(1)
            except:
                pattern = r'\n\n(.*\?\n\n)'
                try:
                    query = re.search(pattern, instr).group(1)
                except: 
                    query = ""
        return query.strip()
    
    elif instr_style=="D_3" or instr_style=="D_4":
        snippets = []
        pattern = r'"([^"]*)"'
        for part in instr.split("\n"):
            if len(part)>0 and part[-1]==':':
                continue
            if "Question/Instruction:" in part:
                query = part.strip("Question/Instruction:")
            elif "uestion" in part or "nstruction" in part:
                try:
                    query = part[part.index(": ")+2:]
                except:
                    try: 
                        query = part[part.index(":")+1:]
                    except:
                        pass
            elif "?" in part:
                pattern = r'^([^!?.]*\?[^\S]*)'
                try:
                    query = re.search(pattern, part).group(1)
                except:
                    pass
            elif "1: " in part:
                try:
                    snip = part[part.index("1: ")+3:].strip()
                    try:
                        if snip[0]=='"':
                            snip = re.findall(pattern, snip)[0]
                    except:
                        pass
                    snippets.append(snip)
                except:
                    pass
            elif "2: " in part:
                try:
                    snip = part[part.index("2: ")+3:].strip()
                    try:
                        if snip[0]=='"':
                            snip = re.findall(pattern, snip)[0]
                    except:
                        pass
                    snippets.append(snip)
                except:
                    pass
            elif "3: " in part:
                try:
                    snip = part[part.index("3: ")+3:].strip()
                    try:
                        if snip[0]=='"':
                            snip = re.findall(pattern, snip)[0]
                    except:
                        pass
                    snippets.append(snip)
                except:
                    pass
            elif "4: " in part:
                try:
                    snip = part[part.index("4: ")+3:].strip()
                    try:
                        if snip[0]=='"':
                            snip = re.findall(pattern, snip)[0]
                    except:
                        pass
                    snippets.append(snip)
                except:
                    pass
        return query.strip(), snippets
    
    elif instr_style=="E_4":
        instr = instr.replace('"X"',"")
        pattern = r'"([^"]*?[^X"]|\S*?\?)"'
        try:
            query = re.findall(pattern, instr)[-1]
        except:
            pass
            pattern = r'^([^!?.]*\?[^\S]*)'
            try:
                query = re.search(pattern, instr).group(1)
            except:
                pass
        return query.strip()
    elif instr_style=="E_6":
        snippets = []
        pattern = r'"([^"]*)"'
        for part in instr.split("\n"):
            if len(part)>0 and part[-1]==':':
                continue
            if "Query:" in part:
                query = part.strip("Query:")
            elif "uery" in part or "nstruction" in part:
                try:
                    query = part[part.index(": ")+2:]
                except:
                    try: 
                        query = part[part.index(":")+1:]
                    except:
                        pass
            elif "?" in part:
                pattern = r'^([^!?.]*\?[^\S]*)'
                try:
                    query = re.search(pattern, part).group(1)
                except:
                    pass
            elif "1: " in part:
                try:
                    snip = part[part.index("1: ")+3:].strip()
                    try:
                        if snip[0]=='"':
                            snip = re.findall(pattern, snip)[0]
                    except:
                        pass
                    snippets.append(snip)
                except:
                    pass
            elif "2: " in part:
                try:
                    snip = part[part.index("2: ")+3:].strip()
                    try:
                        if snip[0]=='"':
                            snip = re.findall(pattern, snip)[0]
                    except:
                        pass
                    snippets.append(snip)
                except:
                    pass
        return query.strip(), snippets

# convert given query, documents, and answer into appropriate format for saving as instruction tuning data
def finalize_instr(instr_style: str,
                  prompt: str,
                  query: str, # pre-stripped when passed in!
                  docs_or_snippets=None # list of source docs
                  ):
    
    # remove prefix and suffix from instruction generation prompt (LLAMA)
    prompt = prompt.removeprefix(SYS_PREFIX)
    prompt = prompt.removesuffix(SYS_SUFFIX)

    ###########################################
    ######## Obtain Documents & Answers #######
    ###########################################
    if instr_style=="A_0":
        docs = list(docs_or_snippets) 
        prompt_prefix = """You are a search engine. A person queried something in detail and the most relevant snippets about the query are as follows.\nQuery: X\nSnippets: """
        prompt_suffix = """\nWhat could the detailed query X be? Answer with a plausible question or instruction.\nX:"""
        isolated_snippets_str = prompt.removeprefix(prompt_prefix).removesuffix(prompt_suffix)
        snippets = isolated_snippets_str[1:-1].split("', '")

    elif instr_style=="A_1_0":
        # Access documents
        prompt_midfix = "\nYou are a search engine. A person queried something in detail about the documents above and the most relevant snippets about the query are as follows.\nQuery: X\nSnippets: "
        isolated_docs_str, snippets_str = prompt.split(prompt_midfix)
        docs = isolated_docs_str[1:-1].split("', '") # a list; use [1:-1] to remove start and end single-quote mark '

        prompt_suffix = "\n\nWhat could the detailed query X be? Answer with a plausible question or instruction.\nX:"
        isolated_snippets_str = snippets_str.removesuffix(prompt_suffix)
        snippets = isolated_snippets_str[1:-1].split("', '") 

    elif instr_style=="B_0_3":
        docs = list(docs_or_snippets) 
        prompt_prefix = """Instruction: X\nSnippets: """
        prompt_suffix = """\n\nWhat kind of instruction could these two snippets be the answer to? Your answer should be a specific question that can only be answered by utilizing information in both snippets. Say "Not sure" if you can\'t come up with a high-quality question. Format your answer as plain text. Before answering, ask yourself if the question you produce covers information in both snippets. If not, regenerate before providing your answer.\nX:"""
        isolated_snippets_str = prompt.removeprefix(prompt_prefix).removesuffix(prompt_suffix)
        snippets = isolated_snippets_str[1:-1].split("', '")

    elif instr_style=="B_1_3":
        isolated_docs_str, snippets_str = prompt.split("\n\nInstruction: X\nSnippets: ")
        docs = isolated_docs_str[1:-1].split("', '")

        prompt_suffix = """\n\nWhat kind of instruction could these two snippets be the answer to? Your answer should be a specific question that can ONLY be answered by utilizing information in BOTH snippets. Say "Not sure" if you can\'t come up with a high-quality question. Format your answer as plain text.\nRead the question again: What kind of instruction could these two snippets be the answer to? Your answer should be a specific question that can ONLY be answered by utilizing information in BOTH snippets. Say "Not sure" if you can\'t come up with a high-quality question. Format your answer as plain text.\nX:"""
        isolated_snippets_str = snippets_str.removesuffix(prompt_suffix)
        snippets = isolated_snippets_str[1:-1].split("', '")

    elif instr_style=="B_1_4":
        isolated_docs_str, snippets_str = prompt.split("\n\nInstruction: X\nSnippets: ")
        docs = isolated_docs_str[1:-1].split("', '")

        prompt_suffix = """\n\nWhat kind of instruction could these two snippets be the answer to? Your answer should be a specific question that can ONLY be answered by utilizing information in BOTH snippets. Say "Not sure" if you can\'t come up with a high-quality question. Format your answer as plain text.\nX:"""
        isolated_snippets_str = snippets_str.removesuffix(prompt_suffix)
        snippets = isolated_snippets_str[1:-1].split("', '")

    elif instr_style=="D_3":
        prompt_prefix = """Below are two documents. Select 3 sentences that are most pertinent to the content of the documents and generate a single question or instruction that can only be answered or responded to using ALL 3 sentences.\n"""
        prompt_suffix = """\n\nYour answer should be a single instruction or question that can only be answered using ALL of the snippets you identify. Before providing your proposal, pause and check that EACH snippet is critical to answering the question/instruction. If not, regenerate and check it again. Format your proposal as:\n\nQuestion/Instruction: \nSnippet 1:\nSnippet 2: \nSnippet 3:"""

        isolated_docs_str = prompt.removeprefix(prompt_prefix).removesuffix(prompt_suffix)
        docs = isolated_docs_str[1:-1].split("', '")
        snippets = docs_or_snippets

    elif instr_style=="D_4":
        prompt_prefix = """Below are two documents. Select 4 sentences that are most pertinent to the content of the documents and generate a single question or instruction that can only be answered or responded to using ALL 4 sentences.\n"""
        prompt_suffix = """\n\nYour answer should be a single instruction or question that can only be answered using ALL of the snippets you identify. Before providing your proposal, pause and check that EACH snippet is critical to answering the question/instruction. If not, regenerate and check it again. Format your proposal as:\n\nQuestion/Instruction: \nSnippet 1:\nSnippet 2: \nSnippet 3: \nSnippet 4:"""

        isolated_docs_str = prompt.removeprefix(prompt_prefix).removesuffix(prompt_suffix)
        docs = isolated_docs_str[1:-1].split("', '")
        snippets = docs_or_snippets

    elif instr_style=="E_4":
        prompt_prefix = """The documents below are ordered by relevance to a given query (a question or instruction), with the first document being most relevant.\n\n"""
        prompt_suffix = """\n\nGiven the order of the documents from most to least useful to answering the query, what could be the query X?\nX:"""

        isolated_docs_str = prompt.removeprefix(prompt_prefix).removesuffix(prompt_suffix)
        docs = isolated_docs_str[1:-1].split("\'\n\n\'")

    elif instr_style=="E_6":
        prompt_suffix = """\n\nSelect two sentences from the above documents. Generate a query to either compare or contrast the information identified. Format your answer as:\n\nSentence 1:\nSentence 2:\nQuery:"""
        isolated_docs_str = prompt.removesuffix(prompt_suffix)
        docs = isolated_docs_str[1:-1].split("\'\n\n\'")
        snippets = docs_or_snippets

    if instr_style!="E_4":
        finalized_instr = "'"
        finalized_instr += docs[0]
        finalized_instr += "'\n'"
        finalized_instr += docs[1]
        finalized_instr += "'\n\n"
        finalized_instr += query

        finalized_answer = " ".join(snippets)

    elif instr_style=="E_4":
        finalized_instr = "Query: "+query 
        for idx, doc in enumerate(docs):
            finalized_instr += "\n\n"
            finalized_instr += str(idx+1)
            finalized_instr += ": '"
            finalized_instr += doc 
            finalized_instr += "'"
        finalized_instr += "\n\nEach document above is identified by an ID number. Order the document IDs according to relevance to the given query above, such that the first ID corresponds to the most relevant document. Your answer should be an ordered list of ID numbers."
    
    return finalized_instr, finalized_answer


# not changed for processing multiple thresholds at one since no scoring needed  
def process_E2_E5(args):

    # output dir not needed since no scoring but use to standardize data format
    json_dir = os.path.join(args.input_dir, "data_jsons")
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)

    scored_by = "scorer_"+args.model_name
    threshold_descrip = "thresh_0" # use 0 since no scoring invovled
    output_dir = os.path.join(json_dir,scored_by,threshold_descrip)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # selected_instr_dir = os.path.join(output_dir, "selected_instructions")
    # if not os.path.exists(selected_instr_dir):
    #     os.makedirs(selected_instr_dir)

    # selected_answr_dir = os.path.join(output_dir, "selected_answers")
    # if not os.path.exists(selected_answr_dir):
    #     os.makedirs(selected_answr_dir)

    # access instruction generation prompts and outputs (i.e., the generated instructions)
    # prompt = instructions
    prompt_dir = os.path.join(args.input_dir,"prompts") # instructions
    num_snips_dir = os.path.join(args.input_dir,"num_snips") # target answers

    # sort alphanumerically so that names match in order of reference 
    all_prompt_files_train = sorted(sorted([
                f
                for f in os.listdir(prompt_dir)
                if f.endswith(".pt") and "train" in f
            ], key=str.lower), key=len)
    all_prompt_files_valid = sorted(sorted([
                f
                for f in os.listdir(prompt_dir)
                if f.endswith(".pt") and "valid" in f
            ], key=str.lower), key=len)
    all_prompt_files_test = sorted(sorted([
                f
                for f in os.listdir(prompt_dir)
                if f.endswith(".pt") and "test" in f
            ], key=str.lower), key=len)
    all_prompt_files = [all_prompt_files_train, all_prompt_files_valid, all_prompt_files_test]

    all_ans_files_train = sorted(sorted([
                f
                for f in os.listdir(num_snips_dir)
                if f.endswith(".pt") and "train" in f
            ], key=str.lower), key=len)
    all_ans_files_valid = sorted(sorted([
                f
                for f in os.listdir(num_snips_dir)
                if f.endswith(".pt") and "valid" in f
            ], key=str.lower), key=len)
    all_ans_files_test = sorted(sorted([
                f
                for f in os.listdir(num_snips_dir)
                if f.endswith(".pt") and "test" in f
            ], key=str.lower), key=len)
    all_ans_files = [all_ans_files_train, all_ans_files_valid,all_ans_files_test]

    # iterate over train/val/test splits
    for split_idx, (prompt_split_files, ans_split_files) in enumerate(tqdm(zip(all_prompt_files, all_ans_files))):

        # Create json file to save data for current split
        one_prompt_pt = prompt_split_files[0] # e.g., train.1.pt
        split_json_file_name = one_prompt_pt[:one_prompt_pt.index(".")] + ".json" # e.g., train.1.pt -> train.json
        data_json_path = os.path.join(output_dir, split_json_file_name)

        with open(data_json_path, "a") as json_file:
            print("########################")
            print("Working on",split_json_file_name)

            # score all the instructons for current split
            for file_idx, (prompt_pt, ans_pt) in enumerate(tqdm(zip(prompt_split_files, ans_split_files))):

                prompt_slice = torch.load(os.path.join(prompt_dir, prompt_pt))
                ans_slice = torch.load(os.path.join(num_snips_dir, ans_pt))

                for example_idx, (prompt, answer) in enumerate(zip(prompt_slice, ans_slice)):
                    finalized_instr = prompt

                    # finalized_instr = enhance(finalized_instr, args)
                    # selected_instr_slice.append(finalized_instr)

                    # Save data to json for HF dataset creation
                    data = {"instruction": finalized_instr, "answer": answer, "score": -1}
                    json.dump(data, json_file)
                    json_file.write('\n')

                print("Finished writing selected datapoints part %d to %s"%(file_idx, data_json_path))

        print("Finished saving selected instructions for %s!"%(split_json_file_name))
        

def main(args):

    # ensure output dir exists
    # enhancement_descrip = get_enhancement_descrip(args)
    json_dir = os.path.join(args.input_dir, "data_jsons")
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)

    scored_by = "scorer_"+args.model_name
    threshold_descrip_3 = "thresh_3"
    threshold_descrip_4 = "thresh_4"
    threshold_descrip_5 = "thresh_5" # however threshold 4.5 per paper spec
    output_dir_3 = os.path.join(json_dir,scored_by,threshold_descrip_3)
    output_dir_4 = os.path.join(json_dir,scored_by,threshold_descrip_4)
    output_dir_5 = os.path.join(json_dir,scored_by,threshold_descrip_5)
    if not os.path.exists(output_dir_3):
        os.makedirs(output_dir_3)
    if not os.path.exists(output_dir_4):
        os.makedirs(output_dir_4)
    if not os.path.exists(output_dir_5):
        os.makedirs(output_dir_5)

    # load selector model
    if args.model_name=="llama2-chat-7b":
        tokenizer_kwargs = {"truncation": True, "max_length": 8192} # EDIT IF NEEDED
        selector_model = pipeline(
            "text-generation", 
            model="meta-llama/Llama-2-7b-chat-hf",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            temperature=1, 
            top_p=1,
            max_new_tokens=512,
            )
    elif args.model_name=="chatglm2-6b":
        tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
        model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True, device='cuda')
        selector_model = model.eval()

    # access instruction generation prompts and outputs (i.e., the generated instructions)
    # prompt = instruction generation template
    prompt_dir = os.path.join(args.input_dir,"prompts")
    # instruction = generated instruction
    instr_dir = os.path.join(args.input_dir,"instructions")

    # sort alphanumerically so that names match in order of reference 
    all_prompt_files_train = sorted(sorted([
                f
                for f in os.listdir(prompt_dir)
                if f.endswith(".pt") and "train" in f
            ], key=str.lower), key=len)
    all_prompt_files_valid = sorted(sorted([
                f
                for f in os.listdir(prompt_dir)
                if f.endswith(".pt") and "valid" in f
            ], key=str.lower), key=len)
    all_prompt_files_test = sorted(sorted([
                f
                for f in os.listdir(prompt_dir)
                if f.endswith(".pt") and "test" in f
            ], key=str.lower), key=len)
    all_prompt_files = [all_prompt_files_test, all_prompt_files_valid, all_prompt_files_test]

    all_instr_files_train = sorted(sorted([
                f
                for f in os.listdir(instr_dir)
                if f.endswith(".pt") and "train" in f
            ], key=str.lower), key=len)
    all_instr_files_valid = sorted(sorted([
                f
                for f in os.listdir(instr_dir)
                if f.endswith(".pt") and "valid" in f
            ], key=str.lower), key=len)
    all_instr_files_test = sorted(sorted([
                f
                for f in os.listdir(instr_dir)
                if f.endswith(".pt") and "test" in f
            ], key=str.lower), key=len)
    all_instr_files = [all_instr_files_test, all_instr_files_valid, all_instr_files_test]

    # iterate over train/val/test splits
    for split_idx, (prompt_split_files, instr_split_files) in enumerate(tqdm(zip(all_prompt_files, all_instr_files))):

        # Create json file to save data for current split
        one_prompt_pt = prompt_split_files[0] # e.g., train.1.pt
        split_json_file_name = one_prompt_pt[:one_prompt_pt.index(".")] + ".json" # e.g., train.1.pt -> train.json
        data_json_path_3 = os.path.join(output_dir_3, split_json_file_name)
        data_json_path_4 = os.path.join(output_dir_4, split_json_file_name)
        data_json_path_5 = os.path.join(output_dir_5, split_json_file_name)


        print("########################")
        print("Working on",split_json_file_name)
        
        # score all the instructons for current split
        for file_idx, (prompt_pt, instr_pt) in enumerate(tqdm(zip(prompt_split_files, instr_split_files))):

            prompt_slice = torch.load(os.path.join(prompt_dir, prompt_pt))
            instr_slice = torch.load(os.path.join(instr_dir, instr_pt)) # technically, same name as prompt_slice but isolate just for formality
            if args.instruction_format=="A_0" or args.instruction_format=="B_0_3":
                doc_dir = os.path.join(args.input_dir,"source_docs")
                source_doc_slice = torch.load(os.path.join(doc_dir, instr_pt))

            for example_idx, (prompt, instr) in enumerate(zip(prompt_slice, instr_slice)):
                
                if args.instruction_format!="D_3" and args.instruction_format!="D_4" and args.instruction_format!="E_6":
                    query = get_query(args.instruction_format, instr.strip())
                else:
                    query, snippets = get_query(args.instruction_format, instr.strip())

                # prepare scoring prompt
                if query != "": # only if the query is NONempty, i.e. if a query was actually generated
                    curation_input = CURATION_PREFIX+f"Instruction:\n{prompt}\n\nResponse:\n{query}"

                    # obtain scoring output from selector model
                    if args.model_name=="chatglm2-6b":
                        response = selector_model.chat(tokenizer, curation_input, history=[])[0]

                    elif args.model_name=="llama2-chat-7b" or args.model_name=="llama2-chat-13b":
                        response = selector_model(llama_ify(curation_input),tokenizer_kwargs=tokenizer_kwargs)[0]['generated_text']#.removeprefix(prompt) NEED THIS? no for now (12/13)

                    # extract score
                    score_matched = REGEX.search(response)
                    score = int(score_matched.group(1)) if score_matched else None

                    # select instruction if score passes threshold
                    if score and score >= 3: 
                        if args.instruction_format=="A_0" or args.instruction_format=="B_0_3":
                            finalized_instr, finalized_answer = finalize_instr(args.instruction_format, prompt, query, source_doc_slice[example_idx]) 
                        elif args.instruction_format=="D_3" or args.instruction_format=="D_4" or args.instruction_format=="E_6":
                            finalized_instr, finalized_answer = finalize_instr(args.instruction_format, prompt, query, snippets)
                        else: 
                            finalized_instr, finalized_answer = finalize_instr(args.instruction_format, prompt, query)
                        
                        # Save data to json for HF dataset creation
                        data = {"instruction": finalized_instr, "answer": finalized_answer, "score": score}
                        with open(data_json_path_3, "a") as json_file:
                            json.dump(data, json_file)
                            json_file.write('\n')

                        # since score is at least 3, check if it's also at least 4
                        if score >= 4:
                            with open(data_json_path_4, "a") as json_file:
                                json.dump(data, json_file)
                                json_file.write('\n')
                            
                            # since score at least 4, check if it's equal to 5
                            if score >= 4.5: # use 4.5 per instruction backtranslation paper spec
                                with open(data_json_path_5, "a") as json_file:
                                    json.dump(data, json_file)
                                    json_file.write('\n')
                
            print("Finished writing selected datapoints part %d to %s"%(file_idx, data_json_path_3))
            
        print("Finished saving selected instructions for %s!"%(split_json_file_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # name of path to appropriate instructions/prompts folders from ./generated_instructions folder
    parser.add_argument("--input_dir", type=str) 
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

    # selector model choices
    parser.add_argument(
        "--model_name",
        choices=[
            "llama2-chat-7b",
            "llama2-chat-13b",
            # "llama2-7b",
            "chatglm2-6b"
        ],
        default="llama2-chat-7b",
        type=str,
    )

    # for now: default to store score 3+, 4+, 5 only
    parser.add_argument("--score_threshold", type=int, default=4.5) 

    args = parser.parse_args()

    if args.instruction_format=="E_2" or args.instruction_format=="E_5":
        process_E2_E5(args)

    else:
        main(args)