import json
import os
import re
from typing import Dict, List

from core.vector_store import VikingStoreWrapper, VikingStoreHTTPWrapper

# --- Relation matching utilities ---

_ENGLISH_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "as", "be", "was", "were",
    "been", "are", "am", "do", "did", "does", "has", "had", "have", "will",
    "would", "could", "should", "may", "might", "shall", "can", "not", "no",
    "nor", "so", "if", "then", "than", "that", "this", "these", "those",
    "what", "which", "who", "whom", "how", "when", "where", "why",
    "all", "each", "every", "both", "few", "more", "most", "other", "some",
    "such", "only", "own", "same", "too", "very", "just", "about", "above",
    "after", "again", "also", "any", "because", "before", "below", "between",
    "during", "into", "its", "out", "over", "through", "under", "until",
    "up", "down", "here", "there", "once", "further", "her", "his", "she",
    "he", "him", "his", "her", "hers", "its", "they", "them", "their",
    "theirs", "our", "ours", "your", "yours", "we", "you", "me", "my",
    "myself", "yourself", "himself", "herself", "itself", "themselves",
    "ourselves", "yourselves", "being", "having", "doing",
})


def _extract_keywords(text: str) -> set:
    if not text:
        return set()
    tokens = text.lower().split()
    result = set()
    for t in tokens:
        t = t.strip(".,;:!?\"'()[]{}—–-")
        if len(t) <= 2 or t in _ENGLISH_STOPWORDS:
            continue
        result.add(t)
    return result


def _cosine_similarity(a: list, b: list) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class VikingStoreWithRelations(VikingStoreWrapper):
    """Vector store enhanced with .relations_{strategy}.jsonl edges.

    Overrides retrieve():
    1. Vector search (via parent class)
    2. For each result, query .relations_{strategy}.jsonl for related URIs
    3. Read and append related documents to context
    """

    def __init__(self, store_path: str, relations_topk: int = 0,
                 use_query_expansion: bool = False, llm=None, embedder=None,
                 strategy: str = "llm_review",
                 relation_keyword_threshold: float = 0.7,
                 relation_vector_threshold: float = 0.7):
        super().__init__(store_path)
        self.relations_topk = relations_topk
        self._vikingfs_path = os.path.join(store_path, "viking")
        self.use_query_expansion = use_query_expansion
        self._llm = llm
        self._embedder = embedder
        self._strategy = strategy
        self._relation_keyword_threshold = float(relation_keyword_threshold)
        self._relation_vector_threshold = float(relation_vector_threshold)
        self._relations_filename = ".relations.jsonl" if strategy == "blind" else f".relations_{strategy}.jsonl"
        self._embed_cache: dict[str, list] = {}
        self._ref_caches: dict[str, dict[str, dict]] = {}

    def _uri_to_parent_path(self, uri: str) -> str:
        if uri.startswith("viking://"):
            rel = uri[len("viking://"):]
        else:
            rel = uri
        local_path = os.path.join(self._vikingfs_path, rel)
        if os.path.isdir(local_path):
            return local_path
        return os.path.dirname(local_path)

    def _embed_text(self, text: str):
        if not self._embedder or not text:
            return None
        if text in self._embed_cache:
            return self._embed_cache[text]
        try:
            result = self._embedder.embed(text)
            self._embed_cache[text] = result
            return result
        except Exception as e:
            print(f"[Warning] Embedding failed: {e}")
            return None

    def _load_ref_cache(self, parent_dir: str) -> dict[str, dict]:
        if parent_dir in self._ref_caches:
            return self._ref_caches[parent_dir]
        cache: dict[str, dict] = {}
        ref_path = os.path.join(parent_dir, ".reference_questions.jsonl")
        if os.path.exists(ref_path):
            try:
                with open(ref_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            qid = rec.get("id", "")
                            if qid:
                                cache[qid] = {"question": rec.get("question", ""), "embedding": rec.get("embedding")}
                        except json.JSONDecodeError:
                            continue
            except Exception:
                pass
        self._ref_caches[parent_dir] = cache
        return cache

    def _resolve_ref_question(self, parent_dir: str, question_id: str) -> dict | None:
        cache = self._load_ref_cache(parent_dir)
        return cache.get(question_id)

    def _query_relations(self, uri: str, query: str) -> List[str]:
        parent_dir = self._uri_to_parent_path(uri)
        jsonl_path = os.path.join(parent_dir, self._relations_filename)
        if not os.path.exists(jsonl_path):
            return []

        query_embedding = None
        query_keywords = set()
        if query:
            query_keywords = _extract_keywords(query)
            query_embedding = self._embed_text(query)

        ref_cache: dict[str, dict | None] = {}
        results = []
        seen = set()
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                uri1 = rec.get("uri1", "")
                uri2 = rec.get("uri2", "")
                if uri1 == uri2 or (uri1 != uri and uri2 != uri):
                    continue
                target = uri2 if uri1 == uri else uri1
                if target in seen:
                    continue

                rec_weight = rec.get("weight", 1.0)
                rec_query = ""
                rec_embedding = None
                if "question_id" in rec:
                    qid = rec["question_id"]
                    if qid not in ref_cache:
                        ref_cache[qid] = self._resolve_ref_question(parent_dir, qid)
                    ref = ref_cache[qid]
                    if ref:
                        rec_query = ref.get("question", "")
                        rec_embedding = ref.get("embedding")
                else:
                    rec_query = rec.get("query_question", "")
                    rec_embedding = rec.get("query_embedding")

                qid = rec.get("question_id", "")
                if qid == "" and "question_id" in rec:
                    seen.add(target)
                    results.append((target, rec_weight))
                    continue

                if not query:
                    seen.add(target)
                    results.append((target, rec_weight))
                    continue

                kw_matched = False
                if query_keywords and rec_query:
                    rec_keywords = _extract_keywords(rec_query)
                    if rec_keywords:
                        overlap = len(query_keywords & rec_keywords)
                        if overlap / len(query_keywords) > self._relation_keyword_threshold:
                            kw_matched = True

                vec_matched = False
                if query_embedding and rec_embedding:
                    sim = _cosine_similarity(query_embedding, rec_embedding)
                    if sim > self._relation_vector_threshold:
                        vec_matched = True

                if kw_matched or vec_matched:
                    seen.add(target)
                    results.append((target, rec_weight))

        results.sort(key=lambda x: x[1], reverse=True)
        return [uri for uri, _ in results]

    def _generate_search_queries(self, query: str) -> List[str]:
        if not self._llm:
            return [query]
        prompt = (
            "Given the following question, generate 3 diverse search queries that would help find relevant information. "
            "Return ONLY the queries, one per line, no numbering or bullets.\n\n"
            f"Question: {query}\n\nSearch queries:"
        )
        try:
            response = self._llm.invoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)
            queries = [q.strip() for q in text.strip().split("\n") if q.strip()]
            queries = [re.sub(r"^\d+[\.\)]\s*", "", q) for q in queries]
            queries = [q for q in queries if len(q) > 5][:5]
            if query not in queries:
                queries.insert(0, query)
            return queries if queries else [query]
        except Exception:
            return [query]

    def retrieve(self, query: str, topk: int, target_uri: str = "viking://resources"):
        expanded_queries = None
        if self.use_query_expansion and self._llm:
            expanded_queries = self._generate_search_queries(query)
            vector_query = " ".join(expanded_queries)
        else:
            vector_query = query

        ret = super().retrieve(vector_query, topk, target_uri)

        vector_uris = list(ret["retrieved_uris"])
        vector_uris_set = set(vector_uris)

        related_uris = []
        related_uris_set = set()
        frontier = list(vector_uris)
        seen_uris = set(vector_uris)

        while frontier:
            next_frontier = []
            for uri in frontier:
                try:
                    rels = self._query_relations(uri, query)
                    for r in rels:
                        if r not in seen_uris and r not in related_uris_set:
                            related_uris.append((r, uri))
                            related_uris_set.add(r)
                            next_frontier.append(r)
                except Exception:
                    continue
            seen_uris.update(next_frontier)
            frontier = next_frontier

        relations_uris = []
        relations_blocks = []
        for rel_uri, source_uri in related_uris:
            try:
                content = self.read_resource(rel_uri)
                if not content:
                    continue
                ret["recall_texts"][rel_uri] = content
                relations_blocks.append(content[:8000])
                relations_uris.append(rel_uri)
            except Exception:
                continue
        ret["context_blocks"] = relations_blocks + ret["context_blocks"]

        ret["relations_uris"] = relations_uris
        ret["relations_found"] = len(related_uris)
        ret["relations_added"] = len(relations_uris)

        if expanded_queries is not None:
            ret["expanded_queries"] = expanded_queries

        return ret


class VikingStoreHTTPWithRelations(VikingStoreHTTPWrapper):
    """HTTP-based vector store with local relations expansion.

    Uses VikingStoreHTTPWrapper for vector search + read, then applies
    the same relations logic as VikingStoreWithRelations by reading
    .relations_{strategy}.jsonl from the local store_path.
    """

    def __init__(self, server_url: str, api_key: str = "", store_path: str = "",
                 embedder=None, strategy: str = "llm_review",
                 relation_keyword_threshold: float = 0.7,
                 relation_vector_threshold: float = 0.7):
        super().__init__(server_url, api_key)
        self._store_path = store_path
        self._vikingfs_path = os.path.join(store_path, "viking") if store_path else ""
        self._embedder = embedder
        self._strategy = strategy
        self._relation_keyword_threshold = float(relation_keyword_threshold)
        self._relation_vector_threshold = float(relation_vector_threshold)
        self._relations_filename = ".relations.jsonl" if strategy == "blind" else f".relations_{strategy}.jsonl"
        self._embed_cache: dict[str, list] = {}
        self._ref_caches: dict[str, dict[str, dict]] = {}

    def _uri_to_parent_path(self, uri: str) -> str:
        if uri.startswith("viking://"):
            rel = uri[len("viking://"):]
        else:
            rel = uri
        local_path = os.path.join(self._vikingfs_path, rel)
        if os.path.isdir(local_path):
            return local_path
        return os.path.dirname(local_path)

    def _embed_text(self, text: str):
        if not self._embedder or not text:
            return None
        if text in self._embed_cache:
            return self._embed_cache[text]
        try:
            result = self._embedder.embed(text)
            self._embed_cache[text] = result
            return result
        except Exception:
            return None

    def _load_ref_cache(self, parent_dir: str) -> dict[str, dict]:
        if parent_dir in self._ref_caches:
            return self._ref_caches[parent_dir]
        cache: dict[str, dict] = {}
        ref_path = os.path.join(parent_dir, ".reference_questions.jsonl")
        if os.path.exists(ref_path):
            try:
                with open(ref_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            qid = rec.get("id", "")
                            if qid:
                                cache[qid] = {"question": rec.get("question", ""), "embedding": rec.get("embedding")}
                        except json.JSONDecodeError:
                            continue
            except Exception:
                pass
        self._ref_caches[parent_dir] = cache
        return cache

    def _resolve_ref_question(self, parent_dir: str, question_id: str) -> dict | None:
        cache = self._load_ref_cache(parent_dir)
        return cache.get(question_id)

    def _query_relations(self, uri: str, query: str) -> List[str]:
        parent_dir = self._uri_to_parent_path(uri)
        jsonl_path = os.path.join(parent_dir, self._relations_filename)
        if not os.path.exists(jsonl_path):
            return []

        query_embedding = None
        query_keywords = set()
        if query:
            query_keywords = _extract_keywords(query)
            query_embedding = self._embed_text(query)

        ref_cache: dict[str, dict | None] = {}
        results = []
        seen = set()
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                uri1 = rec.get("uri1", "")
                uri2 = rec.get("uri2", "")
                if uri1 == uri2 or (uri1 != uri and uri2 != uri):
                    continue
                target = uri2 if uri1 == uri else uri1
                if target in seen:
                    continue

                rec_weight = rec.get("weight", 1.0)
                rec_query = ""
                rec_embedding = None
                if "question_id" in rec:
                    qid = rec["question_id"]
                    if qid not in ref_cache:
                        ref_cache[qid] = self._resolve_ref_question(parent_dir, qid)
                    ref = ref_cache[qid]
                    if ref:
                        rec_query = ref.get("question", "")
                        rec_embedding = ref.get("embedding")
                else:
                    rec_query = rec.get("query_question", "")
                    rec_embedding = rec.get("query_embedding")

                qid = rec.get("question_id", "")
                if qid == "" and "question_id" in rec:
                    seen.add(target)
                    results.append((target, rec_weight))
                    continue

                if not query:
                    seen.add(target)
                    results.append((target, rec_weight))
                    continue

                kw_matched = False
                if query_keywords and rec_query:
                    rec_keywords = _extract_keywords(rec_query)
                    if rec_keywords:
                        overlap = len(query_keywords & rec_keywords)
                        if overlap / len(query_keywords) > self._relation_keyword_threshold:
                            kw_matched = True

                vec_matched = False
                if query_embedding and rec_embedding:
                    sim = _cosine_similarity(query_embedding, rec_embedding)
                    if sim > self._relation_vector_threshold:
                        vec_matched = True

                if kw_matched or vec_matched:
                    seen.add(target)
                    results.append((target, rec_weight))

        results.sort(key=lambda x: x[1], reverse=True)
        return [uri for uri, _ in results]

    def retrieve(self, query: str, topk: int, target_uri: str = "viking://resources"):
        ret = super().retrieve(query, topk, target_uri)

        if not self._vikingfs_path:
            return ret

        vector_uris = list(ret["retrieved_uris"])
        related_uris = []
        related_uris_set = set()
        frontier = list(vector_uris)
        seen_uris = set(vector_uris)

        while frontier:
            next_frontier = []
            for uri in frontier:
                try:
                    rels = self._query_relations(uri, query)
                    for r in rels:
                        if r not in seen_uris and r not in related_uris_set:
                            related_uris.append((r, uri))
                            related_uris_set.add(r)
                            next_frontier.append(r)
                except Exception:
                    continue
            seen_uris.update(next_frontier)
            frontier = next_frontier

        relations_uris = []
        relations_blocks = []
        for rel_uri, source_uri in related_uris:
            try:
                content = self.read_resource(rel_uri)
                if not content:
                    continue
                ret["recall_texts"][rel_uri] = content
                relations_blocks.append(content[:8000])
                relations_uris.append(rel_uri)
            except Exception:
                continue
        ret["context_blocks"] = relations_blocks + ret["context_blocks"]

        ret["relations_uris"] = relations_uris
        ret["relations_found"] = len(related_uris)
        ret["relations_added"] = len(relations_uris)

        return ret
