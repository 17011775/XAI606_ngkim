import os
import subprocess

def contains_null_byte(filepath):
    #print(f"Checking for null byte in path: {filepath}")
    for i, char in enumerate(filepath):
        if char == '\x00':
            #print(f"Null byte found at position {i}")
            return True
    #print("No null byte found in the filepath.")
    return False

def clean_path(filepath):
    #print(f"Cleaning path: {filepath}")
    cleaned_filepath = filepath.replace('\x00', '')
    #print(f"Cleaned path: {cleaned_filepath}")
    return cleaned_filepath

def link_file(from_file, to_file):
    # 경로 출력 (디버깅)
    #print(f"from_file: {from_file}") # /workspace/ng/data/raw/esd/0012/Neutral/0012_000132.wav
    #print(f"to_file: {to_file}") # /workspace/ng/data/processed/esd/wav_processed/0012_000132.wav 
    
    # null 바이트 검사
    if contains_null_byte(from_file):
        from_file = clean_path(from_file)
        #raise ValueError("From File path contains null byte") # 여기서 에러 
    if contains_null_byte(to_file):
        to_file = clean_path(to_file)
        #raise ValueError("To File path contains null byte") # 여기도 에러 
  
    subprocess.check_call(
        f'ln -s "`realpath --relative-to="{os.path.dirname(to_file)}" "{from_file}"`" "{to_file}"', shell=True)


def move_file(from_file, to_file):
    subprocess.check_call(f'mv "{from_file}" "{to_file}"', shell=True)


def copy_file(from_file, to_file):
    subprocess.check_call(f'cp -r "{from_file}" "{to_file}"', shell=True)


def remove_file(*fns):
    for f in fns:
        subprocess.check_call(f'rm -rf "{f}"', shell=True)