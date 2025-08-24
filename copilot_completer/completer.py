from dataclasses import dataclass

import json
import time
from collections.abc import Callable
from contextlib import suppress
from typing import  Any, ParamSpec

import aiohttp
from prompt_toolkit.buffer import Buffer
import requests
from IPython.core.getipython import get_ipython

from .settings import settings
@dataclass
class CompletionContext:
    full_text: str
    cursor_position: int
    cursor_line: int
    token: str
    limit: int

@dataclass
class SimpleCompletion:
    text: str
    type: str

@dataclass
class SimpleMatcherResult:
    completions: list[SimpleCompletion]


async def fetch_copilot_suggestion(buffer:Buffer) -> str | None:
    if not (ip := get_ipython()):
        return None
    completer = ip.Completer

    # use past history as context for prompt
    history_text = buffer.history.get_strings()[-40:]
    text='\n\n'.join(history_text)[-2048:]
    
    # Get the last cell output from IPython
    last_output = ""
    try:
        # Access the Out dictionary which contains output history
        out_dict = ip.user_ns.get('Out', {})
        if out_dict:
            # Get the latest output (highest numbered key)
            latest_key = max(out_dict.keys()) if out_dict.keys() else None
            if latest_key is not None:
                output_value = out_dict[latest_key]
                last_output = f'\n\n#------ last-output ------\nOut[{latest_key}]: {repr(output_value)}'
    except Exception:
        # If there's any error accessing output, continue without it
        pass
    
    full_text='#--------history------\n' +text + last_output + '\n\n #------ current-line-----\n'+buffer.text

    context = CompletionContext(
        full_text=full_text,
        cursor_position=0,
        cursor_line=0,
        token=completer.splitter.split_line(text, 0),
        limit=1,
    )

    suggestion = await copilot_completer(context)

    if completions := suggestion.completions:
        return completions[0].text
    else:
        return None


async def copilot_completer(
    # completer: IPCompleter,
    context: CompletionContext,
) -> SimpleMatcherResult:
    """
    Use GitHub Copilot to complete code the current line
    This uses the current session as the context for the completion
    but ignores lines that start with % or ! as these are not valid Python
    """

    # Check if any provider is available
    line = context.full_text
    # is_comment = line.startswith("#")
    # if is_comment:
    #     line += "\n"

    # Create the prompt for Copilot
    prompt = f"""{line}"""

    # Get the suggestion from the configured provider
    # If the current line starts with # then we allow the suggestion to be a comment
    if settings.provider == "codestral" and settings.codestral_api_key:
        code = await fetch_codestral_suggestion(prompt, suffix="")
    elif settings.provider == "github" and settings.token:
        code = await fetch_suggestion(prompt, stops=["\n\n"])
    else:
        # Fallback: try codestral first, then github
        if settings.codestral_api_key:
            code = await fetch_codestral_suggestion(prompt, suffix="")
        elif settings.token:
            code = await fetch_suggestion(prompt, stops=["\n\n"])
        else:
            code = ""

    # If the line is a comment then we need to add a newline as the suggestion
    # appears after the comment on a new line
    # text = f"{context.token}\n" if is_comment else context.token
    text = context.token

    # Return the suggestion
    provider_name = settings.provider if settings.provider in ["codestral", "github"] else "copilot"
    return SimpleMatcherResult(
        completions=[SimpleCompletion(text=text + code, type=provider_name)]
    )


async def fetch_suggestion(prompt: str, stops: list[str], suffix: str = "") -> str:
    """
    Get a suggestion from GitHub Copilot asynchronously using aiohttp.
    """

    def get_temperature(line_count: int) -> float:
        line_count = max(1, line_count - 2)
        if line_count <= 1:
            return 0
        elif line_count <= 10:
            return 0.2
        elif line_count < 20:
            return 0.4
        else:
            return 0.8

    # TODO: Calculate next indent

    payload = json.dumps(
        {
            "prompt": prompt,
            "suffix": suffix,
            "max_tokens": 200,
            "temperature": get_temperature(prompt.count("\n")),
            "top_p": 1,
            "n": 1,
            # "logprobs": 0,
            "stop": stops,
            "stream": True,  # The API must be called with stream=True
            # "feature_flags": ["trim_to_block"],
            "extra": {
                "language": "python",
                "next_indent": 0,
                "trim_by_indentation": True,
            },
        },
    )
    headers = {
        "OpenAI-Intent": "copilot-ghost",
        "OpenAI-Organization": "github-copilot",
        "Authorization": f"Bearer {get_copilot_token()}",
        "Content-Type": "application/json",
    }

    lines: list[str] = []

    async with aiohttp.ClientSession() as session:  # noqa: SIM117
        async with session.post(
            "https://copilot-proxy.githubusercontent.com/v1/engines/copilot-codex/completions",
            data=payload,
            headers=headers,
        ) as resp:
            async for line in resp.content:
                if line:
                    line = line.decode("utf-8")[6:]  # Remove the data: prefix
                    with suppress(json.JSONDecodeError):
                        lines.append(str(json.loads(line)["choices"][0]["text"]))

    return "".join(lines)


async def fetch_codestral_suggestion(prompt: str, suffix: str = "") -> str:
    """
    Get a suggestion from Codestral API asynchronously using aiohttp.
    """
    # print(prompt, suffix)
    
    payload = json.dumps({
        "model": "codestral-latest",
        "prompt": prompt,
        "suffix": suffix,
        "stop": ["\n\n"],
        "max_tokens": 100,
        "temperature": 0
    })
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {settings.codestral_api_key}",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://codestral.mistral.ai/v1/fim/completions",
            data=payload,
            headers=headers,
        ) as resp:
            if resp.status == 200:
                result = await resp.json()
                if result.get("choices") and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
    
    return ""


P = ParamSpec("P")
R = tuple[str, float]


def memoize_with_expiry(func: Callable[P, R]) -> Callable[..., str]:
    """
    Function to memoize the return value of a function call
    The function func must return a tuple of value and expiry time
    The expiry time is used to determine when to expire the cache
    """
    cache: dict[Any, R] = {}

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> str:
        key = (args, tuple(kwargs.items()))
        if key in cache:
            value, expiry = cache[key]
            if expiry > time.time():
                return value
        value, expiry = func(*args, **kwargs)
        cache[key] = (value, expiry)
        return value

    return wrapper


@memoize_with_expiry
def get_copilot_token() -> R:
    """
    Get the Copilot token from the GitHub API
    """
    response = requests.get(
        "https://api.github.com/copilot_internal/v2/token",
        headers={
            "content-type": "application/json",
            "accept": "application/json",
            "Authorization": f"token {settings.token}",
        },
    )

    response.raise_for_status()
    result = response.json()
    return result["token"], result["expires_at"]
