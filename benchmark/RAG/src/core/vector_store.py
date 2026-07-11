import os
import json
import time
import urllib.request
import urllib.error
import urllib.parse
from typing import Dict, List, Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from adapters.base import StandardDoc, StandardSample
import tiktoken
import openviking as ov


class VikingStoreWrapper:
    def __init__(self, store_path: str):
        self.store_path = store_path
        if not os.path.exists(store_path):
            os.makedirs(store_path)
        
        self.client = ov.SyncOpenViking(path=store_path)
        
        try:
            self.enc = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            print(f"[Warning] tiktoken init failed: {e}")
            self.enc = None

    def count_tokens(self, text: str) -> int:
        if not text or not self.enc:
            return 0
        return len(self.enc.encode(str(text)))

    def build_context_result(
        self,
        context_blocks: List[str],
        uri_prefix: str = "manual_context",
        base_result: Optional[Dict] = None,
    ) -> Dict:
        blocks = [str(block).strip() for block in (context_blocks or []) if str(block).strip()]
        recall_texts = {
            f"{uri_prefix}_{idx}": block
            for idx, block in enumerate(blocks)
        }
        result = dict(base_result or {})
        result["recall_texts"] = recall_texts
        result["context_blocks"] = blocks
        result["retrieved_uris"] = list(recall_texts.keys())
        result["relations_uris"] = []
        result["relations_added_uris"] = []
        result["relation_source_uris"] = {}
        result["relation_group_keys"] = {}
        result["relations_found"] = 0
        result["relations_added"] = 0
        result.setdefault("retrieval_tokens", 0)
        return result

    def ingest(self, samples: List[StandardDoc], max_workers=10, monitor=None, ingest_mode="per_file") -> dict:
        start_time = time.time()
        total_input_tokens = 0
        total_output_tokens = 0
        total_embedding_tokens = 0
        
        if not samples:
            return {
                "time": time.time() - start_time,
                "input_tokens": 0,
                "output_tokens": 0
            }
        
        if ingest_mode == "directory":
            doc_paths = [os.path.abspath(s.doc_path) for s in samples]
            common_ancestor = None
            if doc_paths:
                try:
                    common_ancestor = os.path.commonpath(doc_paths)
                except ValueError:
                    common_ancestor = None
            
            if common_ancestor:
                result = self.client.add_resource(common_ancestor, wait=True, telemetry=True)
                telemetry = result.get("telemetry", {})
                summary = telemetry.get("summary", {})
                tokens = summary.get("tokens", {})
                llm_tokens = tokens.get("llm", {})
                embedding_tokens = tokens.get("embedding", {})
                total_input_tokens = llm_tokens.get("input", 0)
                total_output_tokens = llm_tokens.get("output", 0)
                total_embedding_tokens = embedding_tokens.get("total", 0)
            else:
                for sample in samples:
                    result = self.client.add_resource(sample.doc_path, wait=True, telemetry=True)
                    telemetry = result.get("telemetry", {})
                    summary = telemetry.get("summary", {})
                    tokens = summary.get("tokens", {})
                    llm_tokens = tokens.get("llm", {})
                    embedding_tokens = tokens.get("embedding", {})
                    total_input_tokens += llm_tokens.get("input", 0)
                    total_output_tokens += llm_tokens.get("output", 0)
                    total_embedding_tokens += embedding_tokens.get("total", 0)
        else:
            for sample in samples:
                result = self.client.add_resource(sample.doc_path, wait=True, telemetry=True)
                telemetry = result.get("telemetry", {})
                summary = telemetry.get("summary", {})
                tokens = summary.get("tokens", {})
                llm_tokens = tokens.get("llm", {})
                embedding_tokens = tokens.get("embedding", {})
                total_input_tokens += llm_tokens.get("input", 0)
                total_output_tokens += llm_tokens.get("output", 0)
                total_embedding_tokens += embedding_tokens.get("total", 0)

        return {
            "time": time.time() - start_time,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "embedding_tokens": total_embedding_tokens
        }

    def retrieve(self, query: str, topk: int, target_uri: str = "viking://resources") -> Dict:
        """Retrieve relevant documents: search and read content.

        Returns:
            Dict with keys:
              - recall_texts: {uri: full_content}
              - context_blocks: [truncated_content, ...]
              - retrieved_uris: [uri, ...]
              - retrieval_tokens: int
        """
        # Request 3x documents to ensure enough L2 results after filtering L0/L1
        search_res = self.client.find(query=query, limit=topk * 3, target_uri=target_uri, telemetry=True)

        retrieval_tokens = 0
        if hasattr(search_res, 'telemetry') and search_res.telemetry:
            retrieval_tokens = search_res.telemetry.get('summary', {}).get('tokens', {}).get('embedding', {}).get('total', 0)

        resources = getattr(search_res, 'resources', []) or []

        # Filter out L0 (abstract) and L1 (overview) documents
        resources = [
            r for r in resources
            if not r.uri.endswith(".abstract.md")
            and not r.uri.endswith(".overview.md")
        ]

        # Sort by score (descending) and keep only top `topk` results
        resources = sorted(resources, key=lambda r: getattr(r, 'score', 0), reverse=True)[:topk]

        recall_texts = {}
        context_blocks = []
        retrieved_uris = []

        for r in resources:
            uri = r.uri
            content = self.read_resource(uri)
            retrieved_uris.append(uri)
            recall_texts[uri] = content
            context_blocks.append(content)

        return {
            "recall_texts": recall_texts,
            "context_blocks": context_blocks,
            "retrieved_uris": retrieved_uris,
            "retrieval_tokens": retrieval_tokens,
        }

    def read_resource(self, uri: str) -> str:
        """Read resource content"""
        return str(self.client.read(uri))

    def close(self):
        self.client.close()

    def clear(self):
        """Clear the store"""
        self.client.rm("viking://resources", recursive=True)


class VikingStoreHTTPWrapper:
    """HTTP-based vector store wrapper that connects to an existing OV server.

    Used in fallback modes where the OV server is already running (started by
    vikingbot_runner) to avoid DataDirectoryLocked conflicts.
    """

    def __init__(self, server_url: str, api_key: str = ""):
        self.server_url = server_url.rstrip("/")
        self.api_key = api_key
        try:
            self.enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.enc = None

    def count_tokens(self, text: str) -> int:
        if not text or not self.enc:
            return 0
        return len(self.enc.encode(str(text)))

    def build_context_result(
        self,
        context_blocks: List[str],
        uri_prefix: str = "manual_context",
        base_result: Optional[Dict] = None,
    ) -> Dict:
        blocks = [str(block).strip() for block in (context_blocks or []) if str(block).strip()]
        recall_texts = {
            f"{uri_prefix}_{idx}": block
            for idx, block in enumerate(blocks)
        }
        result = dict(base_result or {})
        result["recall_texts"] = recall_texts
        result["context_blocks"] = blocks
        result["retrieved_uris"] = list(recall_texts.keys())
        result["relations_uris"] = []
        result["relations_added_uris"] = []
        result["relation_source_uris"] = {}
        result["relation_group_keys"] = {}
        result["relations_found"] = 0
        result["relations_added"] = 0
        result.setdefault("retrieval_tokens", 0)
        return result

    def _request(self, method: str, path: str, data: dict = None) -> dict:
        url = f"{self.server_url}{path}"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        if data is not None:
            body = json.dumps(data).encode("utf-8")
            req = urllib.request.Request(url, data=body, headers=headers, method=method)
        else:
            req = urllib.request.Request(url, headers=headers, method=method)

        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def retrieve(self, query: str, topk: int, target_uri: str = "viking://resources") -> Dict:
        # Request 3x documents to ensure enough L2 results after filtering L0/L1
        resp = self._request("POST", "/api/v1/search/find", {
            "query": query,
            "limit": topk * 3,
            "target_uri": target_uri,
            "telemetry": True,
        })

        retrieval_tokens = 0
        telemetry = resp.get("telemetry")
        if telemetry:
            retrieval_tokens = (
                telemetry.get("summary", {})
                .get("tokens", {})
                .get("embedding", {})
                .get("total", 0)
            )

        result = resp.get("result", {})
        resources = result.get("resources", []) or []

        # Filter out L0 (abstract) and L1 (overview) documents
        resources = [
            r for r in resources
            if not r.get("uri", "").endswith(".abstract.md")
            and not r.get("uri", "").endswith(".overview.md")
        ]

        # Sort by score (descending) and keep only top `topk` results
        resources = sorted(resources, key=lambda r: r.get("score", 0), reverse=True)[:topk]

        recall_texts = {}
        context_blocks = []
        retrieved_uris = []

        for r in resources:
            uri = r.get("uri", "")
            content = self.read_resource(uri)
            retrieved_uris.append(uri)
            recall_texts[uri] = content
            context_blocks.append(content)

        return {
            "recall_texts": recall_texts,
            "context_blocks": context_blocks,
            "retrieved_uris": retrieved_uris,
            "retrieval_tokens": retrieval_tokens,
        }

    def read_resource(self, uri: str) -> str:
        encoded_uri = urllib.parse.quote(uri, safe="")
        resp = self._request("GET", f"/api/v1/content/read?uri={encoded_uri}")
        return str(resp.get("result", ""))

    def close(self):
        pass
