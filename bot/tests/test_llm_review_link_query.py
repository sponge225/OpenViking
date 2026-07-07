# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Regression tests for LLM-review relation link query extraction."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vikingbot.agent.loop import _extract_link_query


class LinkQueryExtractionTest(unittest.TestCase):
    def test_keeps_plain_user_question(self):
        query = "What version introduced ERR_FS_CP_DIR_TO_NON_DIR?"

        self.assertEqual(_extract_link_query(query), query)

    def test_strips_question_prefix(self):
        self.assertEqual(
            _extract_link_query("Question: What is the country of origin for Buck-Tick?"),
            "What is the country of origin for Buck-Tick?",
        )
        self.assertEqual(
            _extract_link_query("Here's the question:\nWhat does assert.rejects() do?"),
            "What does assert.rejects() do?",
        )

    def test_extracts_question_from_benchmark_prompt_wrapper(self):
        wrapped = (
            "Answer this question as briefly as possible. Use only the information available "
            "in the database. Do not use any external source. Always use OpenViking tools first. "
            "Search first, then read the results to answer. Always search in "
            "viking://resources/ path.\n\n\n"
            "Question: Do Michigan's airports include both Elko Regional Airport and "
            "Gerald R. Ford International Airport?"
        )

        self.assertEqual(
            _extract_link_query(wrapped),
            "Do Michigan's airports include both Elko Regional Airport and "
            "Gerald R. Ford International Airport?",
        )

    def test_drops_historical_answer_contamination(self):
        contaminated = (
            "In Node.js, what is the behavior of assert.rejects()?\n"
            "Historical answer: Perfect! I found the exact answer in the documentation.\n"
            "Documents selected as useful by that session:\n"
            "  PRIORITY-1. [viking://resources/VersionRAG_processed_docs/...]\n"
            "\n"
            "[YOUR STRATEGY]\n"
            "=== SEARCH RESULTS ===\n"
            "<!-- relations_found:1 searched:11 -->"
        )

        self.assertEqual(
            _extract_link_query(contaminated),
            "In Node.js, what is the behavior of assert.rejects()?",
        )

    def test_rejects_relation_search_result_as_query(self):
        self.assertEqual(_extract_link_query("Question: === SEARCH RESULTS ===\n1. bad"), "")
        self.assertEqual(_extract_link_query("Question: relation_reason bad"), "")

    def test_rejects_unreasonably_long_query(self):
        self.assertEqual(_extract_link_query("x" * 2001), "")


if __name__ == "__main__":
    unittest.main()
