

from __future__ import annotations

import http.client
import json
import time
from typing import Any
import traceback
from ...base import LLM


class HttpsApi(LLM):
    def __init__(self, host, key, model, timeout=60, **kwargs):
        """Https API
        Args:
            host   : host name. please note that the host name does not include 'https://'
            key    : API key.
            model  : LLM model name.
            timeout: API timeout.
        """
        super().__init__(**kwargs)
        self._host = host
        self._key = key
        self._model = model
        self._timeout = timeout
        self._kwargs = kwargs
        self._cumulative_error = 0

    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
        """
        如果直接给了massage,则直接给大模型
        如果没有给massage，则自己构建massage：
            1. 如果没有images，则直接用prompt作为text构建content，然后加入[{'role': 'user', 'content': content}]
            2. 如果有images, 则按照先prompt作为text，然后后面接上image_url的方式构建content，然后加入[{'role': 'user', 'content': content}]
        """
        image64s = kwargs.get('image64s', None)  # 从 kwargs 获取 extra_list 参数
        messages_input = kwargs.get('massage', None)

        if messages_input is not None:
            if isinstance(messages_input, dict):
                messages = [messages_input]  # 单消息包装为列表
            else:
                messages = messages_input
        else:
            content = []
            content.append({
                    "type": "text",
                    "text": prompt.strip()
                })

            if image64s is not None:
                for image in image64s:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image}",
                        }
                    })

            messages = [{
                'role': 'user',
                'content': content
            }]

        while True:
            try:
                conn = http.client.HTTPSConnection(self._host, timeout=self._timeout)
                payload = json.dumps({
                    'max_tokens': self._kwargs.get('max_tokens', 8192),
                    'top_p': self._kwargs.get('top_p', None),
                    'temperature': self._kwargs.get('temperature', 1.0),
                    'model': self._model,
                    'messages': messages
                })
                headers = {
                    'Authorization': f'Bearer {self._key}',
                    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                    'Content-Type': 'application/json'
                }
                # 端点路由
                if 'aliyuncs' in self._host:
                    path = '/compatible-mode/v1/chat/completions'
                elif 'bltcy' in self._host:
                    path = '/v1/chat/completions'
                else:
                    raise ValueError(f"Unsupported API endpoint: {self._host}")

                conn.request('POST', path, payload, headers)
                res = conn.getresponse()
                data = res.read().decode('utf-8')
                data = json.loads(data)
                # print(data)
                response = data['choices'][0]['message']['content']
                if self.debug_mode:
                    self._cumulative_error = 0
                return response
            except Exception as e:
                self._cumulative_error += 1
                if self.debug_mode:
                    if self._cumulative_error == 10:
                        raise RuntimeError(f'{self.__class__.__name__} error: {traceback.format_exc()}.'
                                           f'You may check your API host and API key.')
                else:
                    print(f'{self.__class__.__name__} error: {traceback.format_exc()}.'
                          f'You may check your API host and API key.')
                    time.sleep(2)
                continue
