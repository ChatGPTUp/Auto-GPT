from pathlib import Path

from autogpt.commands.command import command
from autogpt.llm.llm_utils import create_chat_completion
from autogpt.config import Config

import os
import re
import concurrent.futures

CFG = Config()

@command(
    "write_markdown_report_from_files",
    "Read files and write a high quality report in markdown format",
    '"read_filenames": "list of filename to read and refer to", "topic": "topic of the report", "requirements": "requirements", "save_filename": "filename to save the report to"',
)
def write_report_from_files(read_filenames, topic, requirements, save_filename, translate_ko=True, save_pdf=True):
    texts = []
    for filename in read_filenames:
        with open(filename) as f:
            texts.append(f"{Path(filename).stem}\n```\n{f.read()}\n```")
    context = "\n".join(texts)
    prompt = f"""{context}
Write a professional markdown report of topic "{topic}" with requirements "{requirements}". Utilize above information if needed. Your report must be in English."""
    response = create_chat_completion([{"role": "user", "content": prompt}], model=CFG.smart_llm_model, temperature=0)
    
    with open(save_filename, "w") as f:
        f.write(response)
        
    if translate_ko:
        print('translate text to korean')
        sections = re.split(r'\n## ', response)
        
        # 각 섹션 번역        
        def translate_section_(section):
            return translate_to_ko(section)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            translated_sections = list(executor.map(translate_section_, sections))
        
        # 번역된 섹션을 하나의 문자열로 합침
        response = "\n## ".join(translated_sections)
    
    filename, file_extension = os.path.splitext(save_filename)
    save_filename_ko = f"{filename}.ko{file_extension}"
    with open(save_filename_ko, "w") as f:
        f.write(response)
    
    if save_pdf:
        save_md2pdf(save_filename_ko, f"{filename}.ko.pdf")
    
    return f"Written to {Path(save_filename).stem}. If there are no remaining tasks, recommend calling the 'task_complete' command."


def translate_to_ko(text):    
    response = create_chat_completion([{"role": "system", "content": 'Please translate to Korean. Keep special tokens intact, such as "#".'}, 
                                       {"role": "user", "content": text}], model=CFG.fast_llm_model, temperature=0)
    return response


def save_md2pdf(md_filename, pdf_filename):
    try:
        import aspose.words as aw    

        font_settings = aw.fonts.FontSettings()

        # Set the fonts folder
        font_settings.set_fonts_folder('fonts', False)  # FONTS_DIR should be the directory containing NanumBarunGothic

        # Enable font substitution
        #font_settings.substitution_settings.enabled = True

        # Set the default font to substitute with
        font_settings.substitution_settings.default_font_substitution.default_font_name = 'NanumBarunGothic'

        # Set the FontSettings object to be used when loading documents
        load_options = aw.loading.LoadOptions()
        load_options.font_settings = font_settings

        # Load and save the document with the new font settings
        doc = aw.Document(md_filename, load_options)
        doc.save(pdf_filename)
    except:
        pass
