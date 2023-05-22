import os
import random
import string
import datetime
import time
import re
import json
import subprocess
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from slack_sdk import WebClient
from slack_sdk.signature import SignatureVerifier
from slack_sdk.http_retry.builtin_handlers import RateLimitErrorRetryHandler

load_dotenv('../.env')

app = FastAPI()

rate_limit_handler = RateLimitErrorRetryHandler(max_retry_count=1)
signature_verifier = SignatureVerifier(
    signing_secret=os.environ["SLACK_SIGNING_SECRET"]
)

client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))
client.retry_handlers.append(rate_limit_handler)

def prepare_workspace(user_message):
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    length = 5
    random_str = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))
    workspace_name = date_str + "_" + random_str
    workspace = os.path.join(os.getcwd(), 'auto_gpt_workspace', workspace_name)
    os.makedirs(workspace, exist_ok=True)
    ai_settings = f"""ai_name: AutoAskUp
ai_role: an AI that achieves below GOALS.
ai_goals:
- {user_message}
- Terminate if above goal is achieved.
api_budget: 1"""
    with open(os.path.join(workspace, "ai_settings.yaml"), "w") as f:
        f.write(ai_settings)
    return workspace

def format_stdout(stdout):
    text = stdout.decode()
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    text = ansi_escape.sub('', text)
    text = text.strip()
    index = text.find('THOUGHTS')
    if index != -1:
        return text[index:]
    if "Thinking..." in text:
        return
    prefixes = ('REASONING', 'PLAN', '- ', 'CRITICISM', 'SPEAK', 'NEXT ACTION', 'SYSTEM', '$ SPENT', 'BROWSING')
    if text.startswith(prefixes):
        return text

def process_user_message(user_message):
    # Remove @AutoAskUp from message
    user_message = user_message.replace('<@U058EM1SCEQ>', '').strip()
    # Extract options from message
    options = {
        'debug': False,
        'gpt3_only': True
    }
    if user_message.startswith('?'):
        options['debug'] = True
        user_message = user_message.replace('?', '').strip()
    if user_message.startswith('!'):
        options['gpt3_only'] = False
        user_message = user_message.replace('!', '').strip()
    return user_message, options

def run_autogpt_slack(user_message, options, channel, thread_ts):
    
    # Make workspace folder and write ai_settings.yaml in it
    workspace = prepare_workspace(user_message)
    
    # Run autogpt
    main_dir = os.path.dirname(os.getcwd())
    process = subprocess.Popen(
        ["python", os.path.join(main_dir, 'slack', 'api.py'), os.path.join(main_dir, workspace), str(options['gpt3_only'])],
        cwd=main_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Read stdout and send messages to slack
    started_loop = False
    messages = []
    dollars_spent = 0
    while True:
        stdout = process.stdout.readline()
        if (not stdout) and process.poll() is not None:
            break
        if not stdout:
            continue
        output = format_stdout(stdout)
        if output is None:
            continue
        print(output)
        if output.startswith('$ SPENT'):
            dollars_spent = output.split('$ SPENT:')[1].strip()
        if not options['debug']:
            if output.startswith('SPEAK'):
                output = output[6:].strip()
                client.chat_postMessage(
                    channel=channel,
                    text=output,
                    thread_ts=thread_ts
                )
            elif output.startswith('BROWSING'):
                client.chat_postMessage(
                    channel=channel,
                    text=output,
                    thread_ts=thread_ts
                )
            continue
        if output.startswith('THOUGHTS'):
            started_loop = True
        if not started_loop:
            continue
        messages.append(output)
        if started_loop and output.startswith(('$ SPENT')):
            client.chat_postMessage(
                channel=channel,
                text="\n".join(messages),
                thread_ts=thread_ts
            )
            messages = []
        rc = process.poll()
    if len(messages) > 0:
        # Send remaining messages to slack
        client.chat_postMessage(
            channel=channel,
            text="\n".join(messages),
            thread_ts=thread_ts
        )
        messages = []

    # Print stderr
    for line in process.stderr:
        print(line.decode().strip())

    # Upload files to slack
    for fname in os.listdir(workspace):
        if fname not in ['ai_settings.yaml', 'auto-gpt.json', 'file_logger.txt']:
            file = os.path.join(workspace, fname)
            upload_text_file = client.files_upload(
                channels=channel,
                thread_ts=thread_ts,
                title=fname,
                file=file,
            )

    # Send $ spent message to slack
    client.chat_postMessage(
        channel=channel,
        text=f"Total $ spent: {dollars_spent}",
        thread_ts=thread_ts
    )

@app.post("/")
async def slack_events(request: Request, background_tasks: BackgroundTasks):
    # Get the request body and headers
    body = await request.body()
    headers = request.headers
    print('BODY', body)
    print('HEADER', headers)
    # if body.challenge:
    #     return JSONResponse(content=body.challenge) 

    # Avoid replay attacks
    if abs(time.time() - int(headers.get('X-Slack-Request-Timestamp'))) > 60 * 5:
        raise HTTPException(status_code=401, detail="Invalid timestamp")

    if not signature_verifier.is_valid(
            body=body,
            timestamp=headers.get("X-Slack-Request-Timestamp"),
            signature=headers.get("X-Slack-Signature")):
        raise HTTPException(status_code=401, detail="Invalid signature")

    data = json.loads(body)
    user_message, options = process_user_message(data['event']['text'])
    background_tasks.add_task(run_autogpt_slack, user_message, options, data['event']['channel'], data['event']['ts'])
    start_message = "AutoGPT가 실행됩니다."
    if options['debug']:
        start_message += " (DEBUG MODE)"
    if not options['gpt3_only']:
        start_message += " (GPT-4)"
    client.chat_postMessage(
        channel=data['event']['channel'],
        text=start_message,
        thread_ts=data['event']['ts']
    )
    return JSONResponse(content="Launched AutoGPT.")

@app.get("/")
async def index():
    return 'AutoAskUp'

# nohup uvicorn app:app --host 0.0.0.0 --port 30207 --reload &