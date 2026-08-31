import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from app.core.config import settings
from app.services.chunking import chunk_documents
from app.services.vectorstore import replace_source_chunks
from app.utils.guardrails import has_sufficient_context, keep_relevant_results
from app.utils.guardrails import NO_CONTEXT_RESPONSE


class AccuracyHelperTests(unittest.TestCase):
    def test_source_replacement_deletes_matching_ids(self):
        class FakeCollection:
            def __init__(self):
                self.deleted_ids = None

            def get(self, where):
                self.where = where
                return {"ids": ["old-1", "old-2"]}

            def delete(self, ids):
                self.deleted_ids = ids

        class FakeStore:
            def __init__(self):
                self._collection = FakeCollection()

        fake_store = FakeStore()
        with patch(
            "app.services.vectorstore.get_vectorstore", return_value=fake_store
        ), patch("app.services.vectorstore.add_chunks", return_value=1) as add:
            result = replace_source_chunks("report.pdf", [Document(page_content="new")])

        self.assertEqual(result, 1)
        self.assertEqual(fake_store._collection.where, {"source": "report.pdf"})
        self.assertEqual(fake_store._collection.deleted_ids, ["old-1", "old-2"])
        add.assert_called_once()

    def test_distance_gate_rejects_irrelevant_matches(self):
        close = Document(page_content="relevant")
        far = Document(page_content="irrelevant")
        results = [(close, settings.MAX_DISTANCE), (far, settings.MAX_DISTANCE + 0.01)]

        self.assertTrue(has_sufficient_context(results))
        self.assertEqual(keep_relevant_results(results), [(close, settings.MAX_DISTANCE)])

    def test_distance_gate_rejects_empty_or_all_far_matches(self):
        far = Document(page_content="irrelevant")
        results = [(far, settings.MAX_DISTANCE + 0.01)]

        self.assertFalse(has_sufficient_context([]))
        self.assertFalse(has_sufficient_context(results))
        self.assertEqual(keep_relevant_results(results), [])

    def test_structured_documents_are_not_split(self):
        content = "field: value\n" * 200
        document = Document(
            page_content=content,
            metadata={"source": "records.yaml", "structured": True},
        )

        chunks = chunk_documents([document])

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].page_content, content)
        self.assertIn("chunk_id", chunks[0].metadata)

    def test_no_context_response_is_a_safe_abstention(self):
        # The validator must not reject a correct refusal just because the
        # document cannot explicitly prove that a fact is absent.
        from app.agents.graph import validator_node

        state = {
            "question": "What is the deadline?",
            "answer": NO_CONTEXT_RESPONSE,
            "context_sufficient": True,
            "retrieved_chunks": [
                {
                    "text": "The project covers document question answering.",
                    "chunk_id": "chunk-1",
                    "metadata": {"source": "project.pdf", "page": 1},
                }
            ],
            "retries": 0,
            "trace": [],
        }

        result = validator_node(state)

        self.assertTrue(result["validation_passed"])
        self.assertIn("safe abstention", result["trace"][-1])

    def test_exact_question_terms_rescue_chunk_above_distance_gate(self):
        from app.agents.graph import retriever_node

        title = Document(
            page_content="Generative AI and ML Capstone Project",
            metadata={"source": "project.pdf", "page": 1},
        )
        answer_chunk = Document(
            page_content=(
                "The goal of this capstone project is to develop a Generative "
                "AI-powered application."
            ),
            metadata={"source": "project.pdf", "page": 2},
        )

        with patch(
            "app.agents.graph.similarity_search",
            return_value=[
                (title, settings.MAX_DISTANCE + 0.1),
                (answer_chunk, settings.MAX_DISTANCE + 0.1),
            ],
        ):
            result = retriever_node(
                {
                    "question": "What is the goal of this capstone project?",
                    "search_query": "capstone project goal objective",
                    "trace": [],
                }
            )

        self.assertTrue(result["context_sufficient"])
        self.assertEqual(result["retrieved_chunks"][0]["metadata"]["page"], 2)


if __name__ == "__main__":
    unittest.main()
