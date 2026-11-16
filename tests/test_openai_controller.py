"""Offline code checks for src/agent.py; the fake client does not measure model quality."""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from agent import OpenAIController, ask  # noqa: E402
from retrievers import BM25Retriever  # noqa: E402

CHUNKS = [{'id': 'c1', 'text': 'Wundt founded a laboratory in Leipzig in 1879.', 'source_id': 'd', 'source': 'd', 'page': 1, 'metadata': {}},
          {'id': 'c2', 'text': 'Skinner studied reinforcement.', 'source_id': 'd', 'source': 'd', 'page': 2, 'metadata': {}}]


class FakeClient:
    def __init__(self, calls):
        self.calls, self.seen = iter(calls), []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kw):
        self.seen.append(kw)
        name, args = next(self.calls)
        msg = NS(tool_calls=[NS(function=NS(name=name, arguments=json.dumps(args)))])
        return NS(choices=[NS(message=msg)], usage=NS(prompt_tokens=100, completion_tokens=10))


def run(calls, question='Where did Wundt found a laboratory?'):
    client = FakeClient(calls)
    c = OpenAIController('gpt-4o-mini', client)
    return ask({'id': 'q', 'question': question}, c, BM25Retriever(CHUNKS), use_filter=False, top_k=2, budget=800, max_searches=2), client


class ControllerTests(unittest.TestCase):
    def test_search_then_answer_with_citation(self):
        r, client = run([('search', {'query': 'Wundt laboratory'}), ('answer', {'answer': 'Leipzig [c1]', 'citations': ['c1']})])
        self.assertEqual((r['status'], r['citations'], r['n_searches']), ('answered', ['c1'], 1))
        self.assertEqual(r['prompt_tokens'], 200)
        self.assertEqual([t['function']['name'] for t in client.seen[0]['tools']], ['search'])  # first step: search only
        self.assertIn('<retrieved_context>', client.seen[1]['messages'][1]['content'])

    def test_abstain_tool(self):
        r, _ = run([('search', {'query': 'zimbardo'}), ('abstain', {'reason': 'none'})])
        self.assertEqual((r['status'], r['citations']), ('abstain', []))

    def test_provider_error_is_recorded_not_mocked(self):
        r, _ = run([('bogus', {})])
        self.assertEqual(r['status'], 'error'); self.assertIn('Tool không hợp lệ', r['error'])


if __name__ == '__main__':
    unittest.main()
