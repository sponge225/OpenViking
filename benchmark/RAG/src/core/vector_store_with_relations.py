import json
import os
import re
from typing import Dict, List

from core.vector_store import VikingStoreWrapper, VikingStoreHTTPWrapper

# --- Relation matching utilities ---

_DEFAULT_RELATION_SIMILARITY_THRESHOLD = 0.6


def _cosine_similarity(a: list, b: list) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _select_top_relation_groups(group_scores: dict[str, float], topk: int) -> dict[str, float]:
    if topk <= 0:
        return dict(group_scores)
    ranked_groups = sorted(group_scores.items(), key=lambda x: (-x[1], x[0]))
    return dict(ranked_groups[:topk])


def _update_active_relation_groups(
    active_group_scores: dict[str, float],
    candidates: list[dict],
    topk: int,
) -> tuple[set[str] | None, set[str]]:
    if topk <= 0:
        return None, set()

    group_scores = dict(active_group_scores)
    for item in candidates:
        group_key = item["group_key"]
        similarity = item["similarity"]
        if similarity > group_scores.get(group_key, -1.0):
            group_scores[group_key] = similarity

    selected_scores = _select_top_relation_groups(group_scores, topk)
    active_group_scores.clear()
    active_group_scores.update(selected_scores)
    selected_groups = set(selected_scores)
    return selected_groups, set(group_scores) - selected_groups


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
                 similarity_threshold: float = _DEFAULT_RELATION_SIMILARITY_THRESHOLD):
        super().__init__(store_path)
        self.relations_topk = int(relations_topk or 0)
        self.relations_similarity_threshold = float(
            similarity_threshold if similarity_threshold is not None
            else _DEFAULT_RELATION_SIMILARITY_THRESHOLD
        )
        self._vikingfs_path = os.path.join(store_path, "viking")
        self.use_query_expansion = use_query_expansion
        self._llm = llm
        self._embedder = embedder
        self._strategy = strategy
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

    def _collect_relation_candidates(
        self,
        uri: str,
        query: str,
        query_embedding,
    ) -> list[dict]:
        parent_dir = self._uri_to_parent_path(uri)
        jsonl_path = os.path.join(parent_dir, self._relations_filename)
        if not os.path.exists(jsonl_path):
            return []

        ref_cache: dict[str, dict | None] = {}
        candidates: list[dict] = []
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
                # 单向边：只有当 uri == uri1 时，uri2 才是有用的目标
                if uri1 != uri or uri1 == uri2:
                    continue
                target = uri2
                if target.endswith(".abstract.md") or target.endswith(".overview.md"):
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

                if not query:
                    group_key = rec.get("question_id", "") or rec_query or target
                    candidates.append({
                        "target": target,
                        "source_uri": uri,
                        "weight": rec_weight,
                        "similarity": 0.0,
                        "group_key": group_key,
                    })
                    continue

                if query_embedding and rec_embedding:
                    sim = _cosine_similarity(query_embedding, rec_embedding)
                    if sim > self.relations_similarity_threshold:
                        group_key = rec.get("question_id", "") or rec_query or target
                        candidates.append({
                            "target": target,
                            "source_uri": uri,
                            "weight": rec_weight,
                            "similarity": sim,
                            "group_key": group_key,
                        })

        return candidates

    def _expand_relation_beam(self, seed_uris: list[str], query: str) -> list[tuple[str, str]]:
        if self.relations_topk <= 0:
            return []
        query_embedding = self._embed_text(query) if query else None
        if query and not query_embedding:
            return []

        active_group_scores: dict[str, float] = {}
        pruned_groups: set[str] = set()
        seed_uri_set = set(seed_uris)
        seen_uris = set(seed_uris)
        relation_records: dict[str, dict] = {}
        frontier = list(seed_uris)

        while frontier:
            candidates: list[dict] = []
            for uri in frontier:
                try:
                    candidates.extend(self._collect_relation_candidates(uri, query, query_embedding))
                except Exception:
                    continue

            if self.relations_topk > 0:
                candidates = [
                    item for item in candidates
                    if item["group_key"] not in pruned_groups
                ]
                selected_groups, newly_pruned = _update_active_relation_groups(
                    active_group_scores,
                    candidates,
                    self.relations_topk,
                )
                pruned_groups.update(newly_pruned)
            else:
                selected_groups = None

            next_frontier = []
            for item in sorted(candidates, key=lambda x: (-x["similarity"], x["target"])):
                if selected_groups is not None and item["group_key"] not in selected_groups:
                    continue

                target = item["target"]
                if target in seed_uri_set:
                    continue

                existing = relation_records.get(target)
                if existing:
                    existing_is_active = (
                        selected_groups is None
                        or existing["group_key"] in selected_groups
                    )
                    if existing_is_active and item["similarity"] <= existing["similarity"]:
                        continue
                relation_records[target] = item

                if target not in seen_uris:
                    seen_uris.add(target)
                    next_frontier.append(target)

            frontier = next_frontier

        if self.relations_topk > 0:
            active_groups = set(active_group_scores)
            final_records = [
                item for item in relation_records.values()
                if item["group_key"] in active_groups
            ]
        else:
            final_records = list(relation_records.values())

        final_records.sort(key=lambda x: (-x["similarity"], x["target"]))
        return [(item["target"], item["source_uri"]) for item in final_records]

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
        related_uris = self._expand_relation_beam(vector_uris, query)

        relations_uris = []
        relations_blocks = []
        for rel_uri, source_uri in related_uris:
            try:
                content = self.read_resource(rel_uri)
                if not content:
                    continue
                ret["recall_texts"][rel_uri] = content
                relations_blocks.append(content)
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
                 embedder=None, strategy: str = "llm_review", relations_topk: int = 0,
                 similarity_threshold: float = _DEFAULT_RELATION_SIMILARITY_THRESHOLD):
        super().__init__(server_url, api_key)
        self._store_path = store_path
        self._vikingfs_path = os.path.join(store_path, "viking") if store_path else ""
        self._embedder = embedder
        self._strategy = strategy
        self.relations_topk = int(relations_topk or 0)
        self.relations_similarity_threshold = float(
            similarity_threshold if similarity_threshold is not None
            else _DEFAULT_RELATION_SIMILARITY_THRESHOLD
        )
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

    def _collect_relation_candidates(
        self,
        uri: str,
        query: str,
        query_embedding,
    ) -> list[dict]:
        parent_dir = self._uri_to_parent_path(uri)
        jsonl_path = os.path.join(parent_dir, self._relations_filename)
        if not os.path.exists(jsonl_path):
            return []

        ref_cache: dict[str, dict | None] = {}
        candidates: list[dict] = []
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
                # 单向边：只有当 uri == uri1 时，uri2 才是有用的目标
                if uri1 != uri or uri1 == uri2:
                    continue
                target = uri2
                if target.endswith(".abstract.md") or target.endswith(".overview.md"):
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

                if not query:
                    group_key = rec.get("question_id", "") or rec_query or target
                    candidates.append({
                        "target": target,
                        "source_uri": uri,
                        "weight": rec_weight,
                        "similarity": 0.0,
                        "group_key": group_key,
                    })
                    continue

                if query_embedding and rec_embedding:
                    sim = _cosine_similarity(query_embedding, rec_embedding)
                    if sim > self.relations_similarity_threshold:
                        group_key = rec.get("question_id", "") or rec_query or target
                        candidates.append({
                            "target": target,
                            "source_uri": uri,
                            "weight": rec_weight,
                            "similarity": sim,
                            "group_key": group_key,
                        })

        return candidates

    def _expand_relation_beam(self, seed_uris: list[str], query: str) -> list[tuple[str, str]]:
        if self.relations_topk <= 0:
            return []
        query_embedding = self._embed_text(query) if query else None
        if query and not query_embedding:
            return []

        active_group_scores: dict[str, float] = {}
        pruned_groups: set[str] = set()
        seed_uri_set = set(seed_uris)
        seen_uris = set(seed_uris)
        relation_records: dict[str, dict] = {}
        frontier = list(seed_uris)

        while frontier:
            candidates: list[dict] = []
            for uri in frontier:
                try:
                    candidates.extend(self._collect_relation_candidates(uri, query, query_embedding))
                except Exception:
                    continue

            if self.relations_topk > 0:
                candidates = [
                    item for item in candidates
                    if item["group_key"] not in pruned_groups
                ]
                selected_groups, newly_pruned = _update_active_relation_groups(
                    active_group_scores,
                    candidates,
                    self.relations_topk,
                )
                pruned_groups.update(newly_pruned)
            else:
                selected_groups = None

            next_frontier = []
            for item in sorted(candidates, key=lambda x: (-x["similarity"], x["target"])):
                if selected_groups is not None and item["group_key"] not in selected_groups:
                    continue

                target = item["target"]
                if target in seed_uri_set:
                    continue

                existing = relation_records.get(target)
                if existing:
                    existing_is_active = (
                        selected_groups is None
                        or existing["group_key"] in selected_groups
                    )
                    if existing_is_active and item["similarity"] <= existing["similarity"]:
                        continue
                relation_records[target] = item

                if target not in seen_uris:
                    seen_uris.add(target)
                    next_frontier.append(target)

            frontier = next_frontier

        if self.relations_topk > 0:
            active_groups = set(active_group_scores)
            final_records = [
                item for item in relation_records.values()
                if item["group_key"] in active_groups
            ]
        else:
            final_records = list(relation_records.values())

        final_records.sort(key=lambda x: (-x["similarity"], x["target"]))
        return [(item["target"], item["source_uri"]) for item in final_records]

    def retrieve(self, query: str, topk: int, target_uri: str = "viking://resources"):
        ret = super().retrieve(query, topk, target_uri)

        if not self._vikingfs_path:
            return ret

        vector_uris = list(ret["retrieved_uris"])
        related_uris = self._expand_relation_beam(vector_uris, query)

        relations_uris = []
        relations_blocks = []
        for rel_uri, source_uri in related_uris:
            try:
                content = self.read_resource(rel_uri)
                if not content:
                    continue
                ret["recall_texts"][rel_uri] = content
                relations_blocks.append(content)
                relations_uris.append(rel_uri)
            except Exception:
                continue
        ret["context_blocks"] = relations_blocks + ret["context_blocks"]

        ret["relations_uris"] = relations_uris
        ret["relations_found"] = len(related_uris)
        ret["relations_added"] = len(relations_uris)

        return ret
