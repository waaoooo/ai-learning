# openapi_gen.py
import json

class OpenAPIGenerator:
    def __init__(self, title: str, version: str, base_url: str):
        self.spec = {
            "openapi": "3.0.0",
            "info": {"title": title, "version": version},
            "servers": [{"url": base_url}],
            "paths": {},
            "components": {"schemas": {}},
        }

    def add_schema(self, name: str, schema: dict):
        self.spec["components"]["schemas"][name] = schema

    def add_endpoint(self, path: str, method: str, summary: str,
                     request_schema: str = None, response_schema: str = None,
                     tags: list = None):
        if path not in self.spec["paths"]:
            self.spec["paths"][path] = {}
        ep = {"summary": summary, "tags": tags or ["default"],
              "responses": {"200": {"description": "成功"}}}
        if request_schema:
            ep["requestBody"] = {"required": True, "content": {
                "application/json": {"schema": {"$ref": f"#/components/schemas/{request_schema}"}}}}
        if response_schema:
            ep["responses"]["200"]["content"] = {
                "application/json": {"schema": {"$ref": f"#/components/schemas/{response_schema}"}}}
        self.spec["paths"][path][method.lower()] = ep

    def export(self) -> str:
        return json.dumps(self.spec, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    api = OpenAPIGenerator("Agent API", "1.0.0", "https://api.example.com/v1")

    api.add_schema("ChatRequest", {
        "type": "object",
        "properties": {
            "message": {"type": "string"},
            "session_id": {"type": "string"},
        },
        "required": ["message"],
    })
    api.add_schema("ChatResponse", {
        "type": "object",
        "properties": {
            "reply": {"type": "string"},
            "tokens_used": {"type": "integer"},
        },
    })

    api.add_endpoint("/chat", "POST", "发送消息", "ChatRequest", "ChatResponse", ["chat"])
    api.add_endpoint("/health", "GET", "健康检查", tags=["system"])
    api.add_endpoint("/sessions", "GET", "列出会话", tags=["sessions"])

    spec = api.spec
    print(f"OpenAPI 版本: {spec['openapi']}")
    print(f"端点数: {sum(len(v) for v in spec['paths'].values())}")
    print(f"Schema 数: {len(spec['components']['schemas'])}")
    print("\n端点列表:")
    for path, methods in spec["paths"].items():
        for method in methods:
            print(f"  {method.upper():6s} {path}")
