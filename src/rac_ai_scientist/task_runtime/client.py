from __future__ import annotations

import json
from urllib.request import Request, urlopen


class RuntimeClient:
    def __init__(self, url: str, token: str):
        self.url, self.token = url.rstrip("/"), token

    def request(self, operation: str, payload: dict, timeout: float = 30):
        request = Request(self.url + "/" + operation, data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.token})
        with urlopen(request, timeout=timeout) as response:
            value = json.load(response)
        if value.get("error"):
            raise RuntimeError(value["error"])
        return value

    def info(self):
        return self.request("info", {})

    def execute(self, command: str, *, cwd: str | None = None, timeout: float = 600):
        return self.request("execute", {"command": command, "cwd": cwd, "timeout": timeout}, timeout + 30)

    def read_file(self, path: str):
        return self.request("read", {"path": path})["content"]

    def write_file(self, path: str, content: str):
        return self.request("write", {"path": path, "content": content})
