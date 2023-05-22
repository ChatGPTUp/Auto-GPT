import gradio as gr
import utils
from api import AutoAPI, get_openai_api_key
import os, shutil
import json
import uuid

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(FILE_DIR), "auto_gpt_workspace")
OUTPUT_DIR_ORIG = OUTPUT_DIR
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

print(FILE_DIR, OUTPUT_DIR)

CSS = """
#chatbot {font-family: monospace;}
#files .generating {display: none;}
#files .min {min-height: 0px;}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=CSS) as app:
    with gr.Column(visible=True) as setup_pane:
        gr.Markdown(f"""# Auto-GPT WebUI""")
        gr.Markdown(
            "* `OPENAI_API_KEY`를 입력해주세요. 또는 `.env` 파일에 `OPENAI_API_KEY`를 설정하면, 자동으로 불러옵니다."
        )
        with gr.Row():
            open_ai_key = gr.Textbox(
                value=get_openai_api_key(),
                label="OpenAI API Key",
                type="password",
            )
            identifier_setup = gr.Textbox(
                value="default",
                label="Identifier",
                type="text"
            )
        gr.Markdown(
            "* `AI Name`, `AI Role`, `AI Goals`를 채워주세요. 또는 `Examples`에서 선택해주세요. 이후 `Start` 버튼을 누르면 태스크를 수행합니다."
        )
        with gr.Row():
            ai_name = gr.Textbox(
                label="AI Name", 
                placeholder="e.g. Entrepreneur-GPT"
            )
            ai_role = gr.Textbox(
                label="AI Role",
                placeholder="e.g. an AI designed to autonomously develop and run businesses with the sole goal of increasing your net worth.",
            )
        top_5_goals = gr.Dataframe(
            row_count=(5, "fixed"),
            col_count=(1, "fixed"),
            headers=["AI Goals - Enter up to 5"],
            type="array"
        )
        start_btn = gr.Button("Start", variant="primary")
        with gr.Accordion("Open for examples"):
            gr.Examples(
                json.load(open(os.path.join(FILE_DIR, "examples.json"))),
                [ai_name, ai_role, top_5_goals],
            )
    with gr.Column(visible=False) as main_pane:
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(elem_id="chatbot").style(height=750)
            with gr.Column(scale=1):
                identifier_main = gr.Textbox(
                    label="Identifier",
                    type="text"
                )

                # section: feedback from user
                gr.Markdown("* Yes or Custom Response")
                with gr.Row():
                    yes_btn = gr.Button("yes", variant="primary", interactive=False)
                    consecutive_yes = gr.Slider(1, 10, 1, step=1, label="몇번 yes?", interactive=False)
                custom_response = gr.Textbox(
                    label="Custom Response",
                    placeholder="Press 'Enter' to Submit.",
                    interactive=False,
                )

                # section: download files
                gr.Markdown("* Download output files")
                html = gr.HTML(
                    lambda: f"""
                        generated files
                        <pre><code style='overflow-x: auto'>{utils.format_directory(OUTPUT_DIR)}</pre></code>
                """, every=1, elem_id="files", 
                )
                download_btn = gr.Button("Download All Files")

    chat_history = gr.State([[None, None]])
    api = gr.State(None)
    hex = gr.State(None)

    def get_hex(html):
        global HEX
        HEX = uuid.uuid4().hex[-8:]
        global OUTPUT_DIR
        OUTPUT_DIR = os.path.join(OUTPUT_DIR_ORIG, HEX)
        if not os.path.exists(OUTPUT_DIR):
            os.mkdir(OUTPUT_DIR)
        return html, HEX


    def start(open_ai_key, ai_name, ai_role, top_5_goals, hex, identifier):
        auto_api = AutoAPI(open_ai_key, ai_name, ai_role, top_5_goals, identifier)
        return gr.Column.update(visible=False), gr.Column.update(visible=True), auto_api, identifier

    def bot_response(chat, api):
        messages = []
        for message in api.get_chatbot_response():
            message = message.replace("    ", "&nbsp;&nbsp;&nbsp;&nbsp;")
            # print(message, "end?")
            messages.append(f"{message}")
            chat[-1][1] = "\n\n".join(messages) + "<br/><br/>Thinking ... (it takes a few seconds)"
            yield chat
        chat[-1][1] = "\n\n".join(messages)
        yield chat

    def send_message(count, chat, api, message="Y"):
        if message != "Y":
            count = 1
        for i in range(count):
            chat.append([message, None])
            yield chat, count - i
            api.send_message(message)
            for updated_chat in bot_response(chat, api):
                yield updated_chat, count - i

    def activate_inputs():
        return {
            yes_btn: gr.Button.update(interactive=True),
            consecutive_yes: gr.Slider.update(interactive=True),
            custom_response: gr.Textbox.update(interactive=True),
        }

    def deactivate_inputs():
        return {
            yes_btn: gr.Button.update(interactive=False),
            consecutive_yes: gr.Slider.update(interactive=False),
            custom_response: gr.Textbox.update(interactive=False),
        }

    def refresh(download):
        return download

    # start_btn.click(
    #     start,
    #     [open_ai_key, ai_name, ai_role, top_5_goals, download],
    #     [setup_pane, main_pane, api, download],
    # ).then(refresh, [download], [download]).then(bot_response, [chat_history, api], chatbot).then(
    #     activate_inputs, None, [yes_btn, consecutive_yes, custom_response]
    # )
    start_btn.click(
        get_hex,
        [html], [html, hex],
    ).then(
        start, [open_ai_key, ai_name, ai_role, top_5_goals, hex, identifier_setup], [setup_pane, main_pane, api, identifier_main],
    ).then(bot_response, [chat_history, api], chatbot).then(
        activate_inputs, None, [yes_btn, consecutive_yes, custom_response]
    )

    yes_btn.click(
        deactivate_inputs, None, [yes_btn, consecutive_yes, custom_response]
    ).then(
        send_message, [consecutive_yes, chat_history, api], [chatbot, consecutive_yes]
    ).then(
        activate_inputs, None, [yes_btn, consecutive_yes, custom_response]
    )
    custom_response.submit(
        deactivate_inputs, None, [yes_btn, consecutive_yes, custom_response]
    ).then(
        send_message,
        [consecutive_yes, chat_history, api, custom_response],
        [chatbot, consecutive_yes],
    ).then(
        activate_inputs, None, [yes_btn, consecutive_yes, custom_response]
    )

    def download_all_files():
        print(OUTPUT_DIR)
        print(HEX)
        shutil.make_archive("outputs", "zip", OUTPUT_DIR)

    download_btn.click(download_all_files).then(None, _js=utils.DOWNLOAD_OUTPUTS_JS)

app.queue(concurrency_count=20).launch(file_directories=[OUTPUT_DIR])
