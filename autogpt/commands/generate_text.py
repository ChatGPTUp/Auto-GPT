import os
from pathlib import Path
import markdown
import pdfkit
import tiktoken
import re
import concurrent.futures

from autogpt.commands.command import command
from autogpt.llm.llm_utils import create_chat_completion
from autogpt.config import Config

import os
import re
import concurrent.futures

CFG = Config()

def count_tokens(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def translate_section(section, language):
    response = create_chat_completion([{"role": "system", "content": f'Please translate markdown text to {language}. Keep special tokens intact, such as "#".'}, 
                                       {"role": "user", "content": section}], model=CFG.fast_llm_model, temperature=0)
    return response

def translate_md(md, language):
    sections = md.split('\n## ')
    for i in range(1, len(sections)):
        sections[i] = "## " + sections[i]
    def translate_section_(section):
        return translate_section(section, language)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        translated_sections = list(executor.map(translate_section_, sections))
    translated = "\n".join(translated_sections)
    return translated

def save_md_pdf(report, save_filename):
    with open(os.path.join(CFG.workspace_path, save_filename), "w") as f:
        f.write(report)
    markdown_to_pdf(os.path.join(CFG.workspace_path, save_filename), os.path.join(CFG.workspace_path, save_filename.replace(".md", ".pdf")))

@command(
    "write_report",
    "Write a high quality markdown report from files and text",
    '"read_filenames": "list of filename to read and refer to", "knowledge": "prior knowledge to refer to", "topic": "topic of the report", "requirements": "requirements", "language": "language to write report with", "save_filename": "filename to save the markdown report to"',
)
def write_report(read_filenames, knowledge, topic, requirements, save_filename, language, translate_ko=True):
    texts = []
    for filename in read_filenames:
        with open(os.path.join(CFG.workspace_path, filename)) as f:
            texts.append(f"{Path(filename).stem}\n```\n{f.read()}\n```")
    context = "\n".join(texts)
    context += "\n" + knowledge
    max_tokens = 5000
    if count_tokens(context) > max_tokens:
        return f"File contents are too long. Please reduce number of files or shorten file contents."
#     prompt = f"""{context}
# Write a professional markdown report of topic "{topic}" with requirements "{requirements}". Utilize above information if needed. Your report must be in {language}."""
#     response = create_chat_completion([{"role": "user", "content": prompt}], model=CFG.fast_llm_model, temperature=0)
    prompt = f"""{context}
Write a professional markdown report of topic "{topic}" with requirements "{requirements}". Utilize above information if needed. Your report must be in English."""
    en_report = create_chat_completion([{"role": "user", "content": prompt}], model=CFG.smart_llm_model, temperature=0)
    if language and (language.lower() not in ['en', 'english']):
        native_report = translate_md(en_report, language)
        save_md_pdf(native_report, save_filename)
    else:
        save_md_pdf(en_report, save_filename)
        if translate_ko:
            ko_report = translate_md(en_report, "ko")
            save_md_pdf(ko_report, save_filename.replace(".md", "_ko.md"))

    return f"Wrote report at {save_filename}. If there are no remaining tasks, recommend calling the 'task_complete' command."


def markdown_to_pdf(markdown_file, pdf_file):
    """
    prerequisite: apt-get install wkhtmltopdf
    """
    with open(markdown_file, 'r', encoding='utf-8') as f:
        html = markdown.markdown(f.read())
    html= f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Noto Sans KR', sans-serif;
        }}
    </style>
</head>
<body>
{html}
</body>
</html>
"""
    # Write HTML to file
    with open('temp.html', 'w', encoding='utf-8') as f:
        f.write(html)
        
    try:
        pdfkit.from_file('temp.html', pdf_file)
    except Exception as e:
        pass
    finally:
        os.remove('temp.html') # clean up the temporary HTML file
