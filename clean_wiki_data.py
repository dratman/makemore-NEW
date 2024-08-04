import json
import re
import os

def clean_wiki_text(text):
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

    # Remove wikitables
    text = re.sub(r'\{\|[\s\S]*?\|\}', '', text)

    # Remove section headings
    text = re.sub(r'==+.*?==+', '', text)

    # Remove other wiki markup (customize as needed)
    text = re.sub(r'\[\[File:.*?\]\]', '', text)
    text = re.sub(r'\[\[Category:.*?\]\]', '', text)
    text = re.sub(r'\[\[.*?\]\]', '', text)
    text = re.sub(r"''+'", '', text)

    # Remove empty lines and trim whitespace
    text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())

    return text

import json
import os

import json
import os

def process_file(input_file, output_dir):
    output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(input_file))[0] + '.txt')

    with open(input_file, 'r', encoding='utf-8') as f_in:
        try:
            # Try to load the entire file as a single JSON object
            data = json.load(f_in)
            if isinstance(data, list):
                # If it's a list of objects
                json_objects = data
            elif isinstance(data, dict):
                # If it's a single object
                json_objects = [data]
            else:
                print(f"Unexpected JSON structure in file {input_file}")
                return
        except json.JSONDecodeError:
            # If that fails, try reading line by line
            f_in.seek(0)  # Go back to the start of the file
            json_objects = []
            for line in f_in:
                try:
                    json_objects.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue  # Skip invalid lines

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for obj in json_objects:
            if 'text' in obj:
                cleaned_text = clean_wiki_text(obj['text'])
                f_out.write(cleaned_text + '\n\n')  # Add two newlines between entries

def main():
    print("Entering main()")

    input_dir =  '/home/ralph/rdLLMScape/makemore_reborn_without_git_history/wikipedia_text/json_wikipedia'
    output_dir = '/home/ralph/rdLLMScape/makemore_reborn_without_git_history/wikipedia_text/clean_text_wikipedia'

    print ("input_dir = ", input_dir)
    print ("output_dir = ", output_dir)

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            input_file = os.path.join(input_dir, filename)
            process_file(input_file, output_dir)

if __name__ == '__main__':
    main()
