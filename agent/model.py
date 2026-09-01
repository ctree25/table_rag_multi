# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
from typing import Optional

import tiktoken
from google.oauth2 import service_account
from google.auth.transport.requests import Request as GoogleAuthRequest
from openai import OpenAI
from transformers import AutoTokenizer
from tenacity import retry, stop_after_attempt, wait_random_exponential


class Model:
    def __init__(self, model_name):
        self.model_name = model_name
        self.provider = self.get_provider(model_name)
        self.context_limit = self.get_context_limit(model_name)
        self.last_usage = {
            'input_token_count': 0,
            'output_token_count': 0,
            'total_token_count': 0,
        }

        # OpenAI / Azure OpenAI
        if self.provider == 'openai':
            azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
            azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')

            if azure_endpoint:
                if not azure_api_key:
                    raise ValueError(
                        'AZURE_OPENAI_API_KEY is not set.'
                    )

                self.client = OpenAI(
                    api_key=azure_api_key,
                    base_url=azure_endpoint,
                )
            else:
                self.client = OpenAI()

            try:
                self.tokenizer = tiktoken.encoding_for_model(model_name)
            except KeyError:
                self.tokenizer = tiktoken.get_encoding('o200k_base')

        # Gemini on Vertex AI via OpenAI-compatible Chat Completions.
        elif self.provider == 'google':
            try:
                from agent.gemini_key import (
                    GCP_PROJECT_ID,
                    GCP_LOCATION,
                    GEMINI_MODEL_NAME,
                    GCP_SERVICE_ACCOUNT_INFO,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "Gemini selected, but agent/gemini_key.py "
                    "is missing. Copy gemini_key.py.example to "
                    "agent/gemini_key.py and fill in the service "
                    "account values."
                ) from exc

            self.gcp_project_id = GCP_PROJECT_ID
            self.gcp_location = GCP_LOCATION
            self.gemini_default_model = GEMINI_MODEL_NAME

            scopes = [
                'https://www.googleapis.com/auth/cloud-platform'
            ]
            self._gcp_credentials = (
                service_account.Credentials.from_service_account_info(
                    GCP_SERVICE_ACCOUNT_INFO,
                    scopes=scopes,
                )
            )

            # Used only for fast local context-length estimates.
            # Actual API token usage is read from response.usage.
            self.tokenizer = tiktoken.get_encoding('o200k_base')
            self.client = None

        # vLLM models
        elif self.provider == 'vllm':
            self.client = OpenAI(
                base_url=os.getenv(
                    'VLLM_BASE_URL',
                    'http://localhost:8000/v1',
                ),
                api_key='token-abc123',
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_provider(self, model_name):
        name = str(model_name).lower()

        if 'gemini' in name:
            return 'google'
        elif 'gpt' in name:
            return 'openai'
        else:
            return 'vllm'

    def get_context_limit(self, model_name):
        if model_name.startswith('gpt-5'):
            # 可用環境變數覆寫；目前先保守使用 128K。
            return int(
                os.getenv('AZURE_OPENAI_CONTEXT_LIMIT', '128000')
            )

        elif model_name in [
            'gpt-4-0125-preview',
            'gpt-4-turbo-2024-04-09',
            'gpt-4o-mini-2024-07-18',
            'gpt-4.1',
            'gpt-4.1-mini',
            'gpt-4.1-nano',
        ]:
            return 128000

        elif model_name == 'gpt-3.5-turbo-0125':
            return 16385

        elif 'gemini' in str(model_name).lower():
            # Gemini 3 Flash Preview supports a 1,048,576-token context
            # window. Keep this overridable for future Gemini models.
            return int(
                os.getenv('GEMINI_CONTEXT_LIMIT', '1048576')
            )
        elif 'qwen3.5' in str(model_name).lower():
            return int(
                os.getenv('VLLM_CONTEXT_LIMIT', '262144')
            )
        elif 'Mistral-Nemo' in model_name:
            return 128000

        else:
            raise ValueError(
                f'Unsupported model: {model_name}'
            )

    def query(self, prompt, **kwargs):
        # Never carry usage from a previous API call.
        self.last_usage = {
            'input_token_count': 0,
            'output_token_count': 0,
            'total_token_count': 0,
        }

        if not prompt:
            return 'Contents must not be empty.'
        if self.provider == 'openai':
            return self.query_openai(prompt, **kwargs)
        elif self.provider == "google":
            return self.query_gemini(prompt, **kwargs)
        elif self.provider == "vllm":
            return self.query_openai(prompt, **kwargs)
        else:
            raise ValueError(f'Unsupported provider: {self.provider}')

    def _normalize_gemini_model_name(self):
        name = str(self.model_name).strip()

        if not name:
            name = self.gemini_default_model

        if name.startswith('google/'):
            return name

        return f'google/{name}'

    def _refresh_gcp_access_token(self):
        # Refresh a little before expiry so long-running jobs do not fail
        # midway through an experiment.
        now = time.time()
        expiry = getattr(self._gcp_credentials, 'expiry', None)

        should_refresh = (
            self._gcp_credentials.token is None
            or expiry is None
            or now >= (expiry.timestamp() - 60)
        )

        if should_refresh:
            self._gcp_credentials.refresh(GoogleAuthRequest())

        return self._gcp_credentials.token

    def _get_gemini_openai_client(self):
        access_token = self._refresh_gcp_access_token()

        location = self.gcp_location

        if location == 'global':
            host = 'https://aiplatform.googleapis.com'
        else:
            host = (
                f'https://{location}-aiplatform.googleapis.com'
            )

        base_url = (
            f'{host}/v1/projects/'
            f'{self.gcp_project_id}/locations/{location}/'
            'endpoints/openapi'
        )

        # Create a fresh lightweight client so the current OAuth token is
        # always used. The underlying Google credential object is reused.
        return OpenAI(
            base_url=base_url,
            api_key=access_token,
        )

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(10),
        reraise=True,
    )
    def query_gemini_with_retry(self, messages, **kwargs):
        request_kwargs = dict(kwargs)

        # TableRAG already performs explicit external ReAct reasoning.
        # Gemini 3.7 Flash supports low / medium / high; use low by
        # default so the experiment uses the model's lowest supported
        # reasoning level unless GEMINI_REASONING_EFFORT overrides it.
        reasoning_effort = request_kwargs.pop(
            'reasoning_effort',
            None,
        )

        if reasoning_effort is None:
            reasoning_effort = os.getenv(
                'GEMINI_REASONING_EFFORT',
                'low',
            )

        if reasoning_effort:
            request_kwargs['reasoning_effort'] = reasoning_effort

        model_name_lower = str(self.model_name).lower()

        if 'gemini-3.7-flash' in model_name_lower:
            # Gemini 3.7 Flash migration requirement:
            # do not send deprecated sampling parameters.
            request_kwargs.pop('temperature', None)
            request_kwargs.pop('top_p', None)
            request_kwargs.pop('top_k', None)

            # TableRAG uses text-based external ReAct. It does not use
            # Gemini/OpenAI API function calling.

        # Keep only non-empty stop sequences.
        stop = request_kwargs.get('stop')
        if isinstance(stop, list):
            stop = [x for x in stop if x]
            if stop:
                request_kwargs['stop'] = stop
            else:
                request_kwargs.pop('stop', None)

        client = self._get_gemini_openai_client()

        response = client.chat.completions.create(
            model=self._normalize_gemini_model_name(),
            messages=messages,
            **request_kwargs,
        )

        # Validate the response INSIDE the retried function.
        # Gemini may occasionally return a completion with no message or
        # no text content. Raising here makes Tenacity retry instead of
        # letting a downstream AttributeError kill the worker.
        choices = getattr(response, 'choices', None)

        if not choices:
            raise RuntimeError(
                'Gemini returned no choices.'
            )

        choice = choices[0]
        message = getattr(choice, 'message', None)
        content = (
            getattr(message, 'content', None)
            if message is not None
            else None
        )

        if not content:
            finish_reason = getattr(
                choice,
                'finish_reason',
                None,
            )

            raise RuntimeError(
                'Gemini returned empty message/content; '
                f'finish_reason={finish_reason}'
            )

        return response

    def query_gemini(
        self,
        prompt,
        system=None,
        rate_limit_per_minute=None,
        **kwargs,
    ):
        # Gemini 3.7 is strongly tool-oriented and can misinterpret the
        # original TableRAG ReAct text protocol as native function calling.
        # TableRAG does NOT use API tools here: Action is plain text that the
        # local solver parses and executes itself.
        if (
            system is None
            and 'gemini-3.7-flash' in str(self.model_name).lower()
        ):
            system = (
                "Use plain-text response mode only. "
                "Do not emit native tool calls or function calls. "
                "The terms Thought:, Action:, Observation:, and "
                "python_repl_ast in the user prompt are plain-text protocol "
                "markers, not API tools. "
                "When an Action is required, write `Action:` followed by one "
                "single-line Python command as ordinary text. "
                "The external TableRAG program will execute that text itself "
                "and will append the Observation. "
                "Never attempt to invoke python_repl_ast as a native function."
            )

        if system is None:
            messages = [
                {
                    'role': 'user',
                    'content': prompt,
                }
            ]
        else:
            messages = [
                {
                    'role': 'system',
                    'content': system,
                },
                {
                    'role': 'user',
                    'content': prompt,
                },
            ]

        response = self.query_gemini_with_retry(
            messages,
            **kwargs,
        )

        usage = getattr(response, 'usage', None)

        input_token_count = int(
            getattr(usage, 'prompt_tokens', 0) or 0
        )
        output_token_count = int(
            getattr(usage, 'completion_tokens', 0) or 0
        )
        total_token_count = int(
            getattr(
                usage,
                'total_tokens',
                input_token_count + output_token_count,
            )
            or (input_token_count + output_token_count)
        )

        self.last_usage = {
            'input_token_count': input_token_count,
            'output_token_count': output_token_count,
            'total_token_count': total_token_count,
        }

        finish_reason = response.choices[0].finish_reason

        if finish_reason == "length":
            print(
                f"[WARNING] Generation hit max_tokens "
                f"finish_reason=length, "
                f"max_tokens={kwargs.get('max_tokens')}"
            )

        response_text = (
            response.choices[0].message.content or ""
        )
        if rate_limit_per_minute:
            time.sleep(60 / rate_limit_per_minute)

        # Already validated in query_gemini_with_retry().
        response_text = response.choices[0].message.content

        return response_text

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(10),
    )
    def query_openai_with_retry(self, messages, **kwargs):
        request_kwargs = dict(kwargs)

        if self.model_name.startswith('gpt-5'):
            # GPT-5 Chat Completions 使用 max_completion_tokens。
            max_tokens = request_kwargs.pop('max_tokens', None)

            if max_tokens is not None:
                request_kwargs['max_completion_tokens'] = max_tokens

            # Azure GPT-5 reasoning models 不接受這些 sampling 參數。
            request_kwargs.pop('temperature', None)
            request_kwargs.pop('top_p', None)

            # GPT-5 Azure deployments do not accept API-side stop.
            # query_openai() will emulate stop locally after generation.
            request_kwargs.pop('stop', None)

            # TableRAG 已經使用外部 ReAct loop，因此關閉模型內部
            # reasoning，降低 token 消耗並讓行為接近 GPT-4.1。
            request_kwargs.setdefault('reasoning_effort', 'none')

        return self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **request_kwargs,
        )

    def query_openai(
            self,
            prompt,
            system=None,
            rate_limit_per_minute=None,
            **kwargs,
    ):
        # Set default system message
        if system is None:
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        else:
            messages = [
                {
                    "role": "system",
                    "content": system,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ]

        response = self.query_openai_with_retry(
            messages,
            **kwargs,
        )

        # Record the actual usage returned by Azure/OpenAI.
        usage = getattr(response, "usage", None)

        input_token_count = int(
            getattr(usage, "prompt_tokens", 0) or 0
        )
        output_token_count = int(
            getattr(usage, "completion_tokens", 0) or 0
        )
        total_token_count = int(
            getattr(
                usage,
                "total_tokens",
                input_token_count + output_token_count,
            )
            or (input_token_count + output_token_count)
        )

        self.last_usage = {
            "input_token_count": input_token_count,
            "output_token_count": output_token_count,
            "total_token_count": total_token_count,
        }

        # Sleep to avoid rate limit if rate limit is set.
        if rate_limit_per_minute:
            time.sleep(60 / rate_limit_per_minute)

        response_text = (
            response.choices[0].message.content or ""
        )

        # GPT-5 Azure deployments do not accept API-side stop.
        # Preserve TableRAG behavior by applying the stop locally.
        if self.model_name.startswith("gpt-5"):
            stop_sequences = kwargs.get("stop")

            if isinstance(stop_sequences, str):
                stop_sequences = [stop_sequences]

            if stop_sequences:
                stop_positions = [
                    response_text.find(sequence)
                    for sequence in stop_sequences
                    if (
                        sequence
                        and response_text.find(sequence) >= 0
                    )
                ]

                if stop_positions:
                    response_text = response_text[
                        :min(stop_positions)
                    ]

        return response_text

    def get_token_count(self, prompt):
        if not prompt:
            return 0

        if self.provider in {'openai', 'google', 'vllm'}:
            # For Gemini this is a local approximation used only for
            # context-length checks. Real token usage comes from the API.
            return len(self.tokenizer.encode(prompt))

        raise ValueError(
            f'Unsupported provider: {self.provider}'
        )


if __name__ == '__main__':
    def test_model(model_name, prompt):
        print(f'Testing model: {model_name}')
        model = Model(model_name)
        print(f'Prompt: {prompt}')
        response = model.query(prompt)
        print(f'Response: {response}')
        num_tokens = model.get_token_count(prompt)
        print(f'Number of tokens: {num_tokens}')
    prompt = 'Hello, how are you?'
    for model in ['gpt-4o-mini-2024-07-18', 'gemini-3-flash-preview']:
    # for model in ['mistralai/Mistral-Nemo-Instruct-2407']:
        test_model(model, prompt)
