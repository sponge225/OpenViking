# Copyright (C) 2025-2026 Shu Wang
# SPDX-License-Identifier: Apache-2.0 OR AGPL-3.0-only

import yaml
from bookrag_core.configs.mineru_config import MinerU
from bookrag_core.configs.llm_config import LLMConfig
from bookrag_core.configs.tree_config import TreeConfig
from bookrag_core.configs.graph_config import GraphConfig
from bookrag_core.configs.vlm_config import VLMConfig
from bookrag_core.configs.rag_config import RAGConfig
from bookrag_core.configs.vdb_config import VDBConfig
from pydantic import BaseModel, Field
from typing import Optional


class SystemConfig(BaseModel):
    """
    Top-level application configuration model.
    Pydantic will automatically handle nested validation and instantiation.
    """

    # LLM Configurations
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    mineru: MinerU = Field(default_factory=MinerU)

    # Index Configurations
    tree: TreeConfig = Field(default_factory=TreeConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    vdb: VDBConfig = Field(default_factory=VDBConfig)

    # Other Index selection
    index_type: Optional[str] = "gbc"  # Options: "gbc", "tree", "vanilla", "bm25", "raptor", "pdf_vanilla"

    rag_force_reprocess: Optional[bool] = False

    # Import concurrency is deliberately separate from query/evaluation
    # concurrency. ``mineru_workers`` controls independent PDF parsing
    # processes. ``doc_workers`` controls independent post-MinerU document
    # pipelines. ``ingest_workers`` controls aggregate-tree summary and KG
    # stages.
    mineru_workers: int = Field(default=1, ge=1)
    doc_workers: int = Field(default=1, ge=1)
    ingest_workers: int = Field(default=1, ge=1)

    # RAG Configurations
    rag: RAGConfig = Field(default_factory=RAGConfig)

    # Paths
    source_format: Optional[str] = "pdf"
    pdf_path: Optional[str] = "/home/wangshu/multimodal/GBC-RAG/test/double_paper.pdf"
    save_path: Optional[str] = "/home/wangshu/multimodal/GBC-RAG/test/tree_index"

    # # 新增: 专门用于存放评估结果的根目录
    # evaluation_output_path: Optional[str] = Field(
    #     default="/home/wangshu/multimodal/GBC-RAG/test/tree_index/evaluation_results",
    #     description="Root directory to save evaluation results."
    # )


def load_system_config(path: str = "../configs/default.yaml") -> SystemConfig:
    with open(path, "r") as f:
        raw_config = yaml.safe_load(f)

    if "rag" in raw_config:
        rag_data = raw_config["rag"]
        raw_config["rag"] = {"strategy_config": rag_data}

    cfg = SystemConfig(**raw_config)
    return cfg
