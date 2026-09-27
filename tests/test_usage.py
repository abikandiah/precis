import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from precis import usage
from precis.llm import complete


def _response(prompt=100, completion=20, cached=0, cost: object = 0.01):
    return SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
            model_extra={"cost": cost} if cost is not None else {},
        )
    )


def test_records_tokens_and_reported_cost():
    with usage.track() as tracked:
        usage.record_llm_response(_response(cached=40))
        usage.record_llm_response(_response(cost=0.02))
    assert tracked.llm_calls == 2
    assert tracked.prompt_tokens == 200
    assert tracked.completion_tokens == 40
    assert tracked.cached_tokens == 40
    assert tracked.llm_cost_usd == 0.03
    assert tracked.cost_missing == 0


def test_a_response_without_a_cost_is_counted_as_missing_not_free():
    with usage.track() as tracked:
        usage.record_llm_response(_response(cost=None))
        usage.record_llm_response(SimpleNamespace())
    assert tracked.llm_calls == 2
    assert tracked.cost_missing == 2
    assert "2 call(s) reported no cost" in tracked.summary()


def test_mocked_usage_attributes_are_ignored():
    with usage.track() as tracked:
        usage.record_llm_response(MagicMock())
    assert tracked.prompt_tokens == 0
    assert tracked.cost_missing == 1


def test_searches_count_credits_by_depth():
    with usage.track() as tracked:
        usage.record_search(deep=False)
        usage.record_search(deep=True)
    assert tracked.searches == 2
    assert tracked.search_credits == 3
    assert tracked.total_cost_usd == 3 * usage.SEARCH_USD_PER_CREDIT


def test_nothing_is_recorded_outside_a_scope():
    usage.record_llm_response(_response())
    usage.record_search(deep=True)
    with usage.track() as tracked:
        pass
    assert tracked.llm_calls == 0


def test_scopes_nest_without_leaking():
    with usage.track() as outer:
        usage.record_search(deep=False)
        with usage.track() as inner:
            usage.record_search(deep=False)
        usage.record_search(deep=False)
    assert (outer.searches, inner.searches) == (2, 1)


def test_parallel_tasks_record_onto_the_scope_they_were_started_in():
    async def call():
        usage.record_llm_response(_response())

    async def run():
        await asyncio.gather(*(call() for _ in range(3)))

    with usage.track() as tracked:
        asyncio.run(run())
    assert tracked.llm_calls == 3


async def test_llm_calls_are_recorded_and_ask_the_gateway_for_cost():
    client = MagicMock()
    response = _response(cost=0.005)
    response.choices = [SimpleNamespace(message=SimpleNamespace(content="hi"), finish_reason="stop")]
    client.chat.completions.create = AsyncMock(return_value=response)
    with usage.track() as tracked:
        assert await complete(client, messages=[{"role": "user", "content": "x"}], model="m") == "hi"
    assert tracked.llm_calls == 1
    assert tracked.llm_cost_usd == 0.005
    assert client.chat.completions.create.call_args.kwargs["extra_body"] == {"usage": {"include": True}}
