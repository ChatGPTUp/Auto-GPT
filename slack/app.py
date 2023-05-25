import os
import random
import string
import datetime
import time
import re
import json
import subprocess
from collections import defaultdict
import signal

import openai
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

thread_ts2pids = defaultdict(list)

def user_message2ai_settings(user_message, api_budget=1, infer=False):
    if infer:
        prompt = f"""
An AI bot will handle given request. Provide name, role and goals for the AI assistant in JSON format with following keys:
- "ai_name": AI names
- "ai_role": AI role, starting with 'an AI that'
- "ai_goals": List of 1~4 necessary sequential goals for the AI.
Simplify ai_goals as much as possible.

Request:
```
Write dummy text to 'dummy_text.txt' file
```
Response:
{{
    "ai_name": "DummyTextBot",
    "ai_role": "an AI that writes dummy text to 'dummy_text.txt' file.",
    "ai_goals": [
        "Generate dummy text.",
        "Write dummy text to 'dummy_text.txt' file."
    ]
}}
Request:
```
Decide whether to buy or sell Tesla stock, and write a report on the decision in markdown format in Korean.
```
Response:
{{
    "ai_name": "TeslaStockBot",
    "ai_role": "an AI that decides whether to buy or sell Tesla stock, and writes a report on the decision in markdown format in Korean.",
    "ai_goals": [
        "Research financial data of Tesla stock.",
        "Research recent news about Tesla.",
        "Based on research, make a decision whether to buy or sell Tesla stock.",
        "Write a report on the decision in markdown format in Korean."
    ]
}}
Request:
```
{user_message}
```
Response:
"""
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0,
            api_key=os.getenv('OPENAI_API_KEY')
        )
        data = json.loads(response.choices[0]['message']['content'])
        ai_goals_str = '\n'.join(['- ' + goal for goal in  data['ai_goals']])
        ai_settings = f"""
ai_name: {data['ai_name']}
ai_role: {data['ai_role']}
ai_goals:
{ai_goals_str}
- Terminate if above goals are achieved.
api_budget: {api_budget}
"""
    else:
        user_messages = user_message.split('\n')
        goals = [f"- {user_message}" for user_message in user_messages if user_message != ""]
        goals = '\n'.join(goals)
        ai_settings = f"""ai_name: AutoAskUp
ai_role: an AI that achieves below GOALS.
ai_goals:
{goals}
- Terminate if above goal is achieved.
api_budget: {api_budget}"""
    return ai_settings

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
        'gpt3_only': True,
        'api_budget': 1,
    }
    if user_message.startswith('?'):
        options['debug'] = True
        user_message = user_message.replace('?', '').strip()
    if user_message.startswith('!'):
        options['gpt3_only'] = False
        user_message = user_message.replace('!', '').strip()

    match = re.search(r'\$(\d+(\.\d+)?)\s*$', user_message)
    if match:
        options['api_budget'] = min(5, float(match.group(1)))
        user_message = user_message[:user_message.rfind(match.group(0))].strip()

    return user_message, options

def run_autogpt_slack(user_message, options, channel, thread_ts):
    
    # Make workspace folder and write ai_settings.yaml in it
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    length = 5
    random_str = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))
    workspace_name = date_str + "_" + random_str
    workspace = os.path.join(os.getcwd(), 'auto_gpt_workspace', workspace_name)
    os.makedirs(workspace, exist_ok=True)
    ai_settings = user_message2ai_settings(user_message, options['api_budget'])
    with open(os.path.join(workspace, "ai_settings.yaml"), "w") as f:
        f.write(ai_settings)

    # Run autogpt
    main_dir = os.path.dirname(os.getcwd())
    process = subprocess.Popen(
        ["python", os.path.join(main_dir, 'slack', 'api.py'), os.path.join(main_dir, workspace), str(options['gpt3_only'])],
        cwd=main_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    ai_settings_message = f"AutoGPT launched with settings:\n{ai_settings.replace('api_budget: ', 'api_budget: $')}"
    print(ai_settings_message)
    client.chat_postMessage(
        channel=channel,
        text=ai_settings_message,
        thread_ts=thread_ts
    )
    
    # add to pid
    thread_ts2pids[thread_ts].append(process.pid)
    print('thread_ts2pids', thread_ts2pids)

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
        text=f"Total Spent: ${round(float(dollars_spent), 2)}",
        thread_ts=thread_ts
    )

    # Delete from pid
    if process.pid in thread_ts2pids[thread_ts]:
        thread_ts2pids[thread_ts].remove(process.pid)
        if len(thread_ts2pids[thread_ts]) == 0:
            del thread_ts2pids[thread_ts]


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
    event = data['event']  
    thread_ts = event['thread_ts'] if 'thread_ts' in event else event['ts']
    
    if user_message.lower() == 'stop':
        # If stop command, kill process
        if thread_ts not in thread_ts2pids:
            client.chat_postMessage(
                channel=event['channel'],
                text="AutoGPT is not launched yet.",
                thread_ts=thread_ts
            )
            return JSONResponse(content="Main process is not running yet.")
        for pid in thread_ts2pids[thread_ts]:
            os.kill(pid, signal.SIGTERM)
        del thread_ts2pids[thread_ts]
        print('thread_ts2pids', thread_ts2pids)
        client.chat_postMessage(
            channel=event['channel'],
            text="AutoGPT is stopped.",
            thread_ts=thread_ts
        )
        return JSONResponse(content="AutoGPT is stopped.")
    
    background_tasks.add_task(run_autogpt_slack, user_message, options, event['channel'], thread_ts)
    start_message = "Preparing to launch AutoGPT..."
    if options['debug']:
        start_message += " (in DEBUG MODE)"
    if not options['gpt3_only']:
        start_message += " (with GPT4)"
    client.chat_postMessage(
        channel=event['channel'],
        text=start_message,
        thread_ts=thread_ts
    )
    return JSONResponse(content="Launched AutoGPT.")

@app.get("/")
async def index():
    return 'AutoAskUp'

# nohup uvicorn app:app --host 0.0.0.0 --port 30207 --reload &