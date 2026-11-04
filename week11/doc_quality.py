# doc_quality.py

class DocQualityChecker:
    CHECKS = {
        "has_title": lambda doc: "# " in doc,
        "has_description": lambda doc: len(doc) > 100,
        "has_examples": lambda doc: "```" in doc,
        "has_error_codes": lambda doc: "错误" in doc or "error" in doc.lower(),
        "has_version": lambda doc: "版本" in doc or "version" in doc.lower(),
        "no_todo": lambda doc: "TODO" not in doc and "FIXME" not in doc,
    }

    def check(self, doc: str, name: str = "doc") -> dict:
        results = {}
        for check_name, fn in self.CHECKS.items():
            try:
                results[check_name] = fn(doc)
            except Exception:
                results[check_name] = False

        passed = sum(1 for v in results.values() if v)
        total = len(results)
        print(f"\n文档: {name}")
        print(f"质量得分: {passed}/{total} ({passed/total*100:.0f}%)")
        for k, v in results.items():
            print(f"  {'✅' if v else '❌'} {k}")
        return {"passed": passed, "total": total, "score": passed / total}

if __name__ == "__main__":
    checker = DocQualityChecker()

    good_doc = "# API 文档 v1.0\n本文档描述 Agent API 的使用方法，包括所有端点、请求参数和响应格式。\n```python\nresponse = client.chat('你好')\n```\n错误码: 400 参数错误, 401 未授权, 500 内部错误\n版本: 1.0.0\n"
    checker.check(good_doc, "好文档")

    bad_doc = "TODO: 待补充"
    checker.check(bad_doc, "差文档")

