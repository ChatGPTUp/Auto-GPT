from pathlib import Path

from autogpt.commands.command import command
from autogpt.llm.llm_utils import create_chat_completion
from autogpt.config import Config

CFG = Config()

@command(
    "write_markdown_report_from_files",
    "Read files and write a high quality report in markdown format",
    '"read_filenames": "list of filename to read and refer to", "topic": "topic of the report", "requirements": "requirements", "save_filename": "filename to save the report to"',
)
def write_report_from_files(read_filenames, topic, requirements, save_filename):
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
    return f"Written to {Path(save_filename).stem}"