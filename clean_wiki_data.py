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

def process_file(input_file, output_dir):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'text' in data:
        cleaned_text = clean_wiki_text(data['text'])

        output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(input_file))[0] + '.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)

def main():
    print("Entering main()")
    input_dir =  '/home/ralph/rdLLMScape/makemore_reborn_without_git_history/wikipedia_text/jsonA_wikipedia'
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
