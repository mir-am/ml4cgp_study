import pandas as pd
from typing import List
import re
import os
import json
import fnmatch


def get_input_and_mask(src, dst , max_length, tokenizer):
    src_tokens = tokenizer.tokenize(src)
    dst_tokens = tokenizer.tokenize(dst)
    tokens=[tokenizer.cls_token]+src_tokens+[tokenizer.sep_token]+dst_tokens+[tokenizer.sep_token]
    token_length = len(tokens)
    if  token_length > max_length:
        truncation_ratio = max_length/token_length
        src_len = len(src_tokens)
        dst_len = len(dst_tokens)
        if  src_len < dst_len:
            src_tokens = src_tokens[:int(len(src_tokens) * truncation_ratio)]
            dst_tokens = dst_tokens[:max_length - len(src_tokens) - 3]
        else:
            dst_tokens = dst_tokens[:int(len(dst_tokens) * truncation_ratio)]
            src_tokens = src_tokens[:max_length - len(dst_tokens) - 3]
        new_tokens=[tokenizer.cls_token]+src_tokens+[tokenizer.sep_token]+dst_tokens+[tokenizer.sep_token]
        mask = [1 for _ in range(len(new_tokens))]
    else:
        new_tokens = [tokens[i] if i < token_length else tokenizer.pad_token for i in range(max_length)]
        mask = [1 if i < token_length else 0 for i in range(max_length)]

    tokens_ids= tokenizer.convert_tokens_to_ids(new_tokens)
    if len(tokens_ids) > max_length:
        print(len(dst_tokens))
        print(len(src_tokens))
        print(len(tokens_ids))
        import pdb
        pdb.set_trace()
        raise "Truncation errors"
    return tokens_ids, mask

def report_training_samples_types(df: pd.DataFrame):
    print(f"Prune: {df[(df['wiretap'] == 0) & (df['wala-cge-0cfa-noreflect-intf-trans'] == 1)].shape[0]:,}")
    print(f"Add: {df[(df['wiretap'] == 1) & (df['wala-cge-0cfa-noreflect-intf-trans'] == 0)].shape[0]:,}")
    print(f"Agree: {df[(df['wiretap'] == 1) & (df['wala-cge-0cfa-noreflect-intf-trans'] == 1)].shape[0]:,}" )


def normalize_src_code(code: str) -> str:
    """
    normalizes source code into one line
    """
    return " ".join(code.replace("\n", " ").split())

def remove_java_comments(code):
    code = re.sub(re.compile("/\*.*?\*/",re.DOTALL ), "", code) 
    code = re.sub(re.compile("//.*" ), "", code)
    return code

def tokenize_java_identifiers(identifier: str) -> List[str]:
    return [t.lower() for t in re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', identifier)]

def load_method_src(path):
    data = {}
    df = pd.read_csv(path)
    descriptor = df['descriptor']
    code = df['code_wo_cmt']
    for i in range(len(descriptor)):
        if isinstance(code[i], str):
            data[descriptor[i]] = normalize_src_code(code[i])
    return data

def extract_src_file_path(node_uri: str) -> str:
    """
    Extracts the source file path from Wala's CG nodes URI
    """
    return node_uri.split(":")[0].split(".")[0].split("$")[0] + ".java"

def extract_method_name(node_uri: str) -> str:
    return re.search(r".+\.(.+):.+", node_uri).group(1)

def extract_class_name(uri: str) -> str:
    parts = uri.split(':')
    parts = parts[0].split('/') 
    last_part = parts[-1].split('.')
    return last_part[0]

def extract_ns_from_uri(uri: str) -> str:
    parts = uri.split(':')[0]
    parts = parts.split('/')[:-1]
    return "/".join(parts)


def get_java_file_path(class_file_path: str):
    """
    Gets the Java file path from a class file path in JARs
    """
    parts = class_file_path.split("/")
    parts[-1] = parts[-1].replace(".class", "")
    
    if "$" in parts[-1]:
        # Split the last part by "$" and get the first part
        parts[-1] = parts[-1].split("$")[0]
    
    parts[-1] += ".java"
    
    return "/".join(parts)

def find_files(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result
    
def find_model_checkpoint(dir_path, pattern):
    matches = []
    for root, dirs, files in os.walk(dir_path):
        for filename in fnmatch.filter(files, pattern):
            matches.append(os.path.join(root, filename))

    if not matches:
        return None
    
    def extract_step(file_path):
        match = re.search('step=(\d+).ckpt', file_path)
        return int(match.group(1)) if match else -1

    return max(matches, key=extract_step)

def read_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
