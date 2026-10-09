"""The page shows the same token total as the terminal: everything the model processed, cached or not."""

from tests.agent.test_caching import cached_script


def test_the_web_page_gets_the_total_including_cached_tokens(make_web):
    w = make_web(cached_script())
    w.chat("Show me the plan.")
    usage = [m["usage"] for m in w.state()["transcript"] if m.get("usage")][-1]
    assert usage["tokens"] == 8210 and usage["steps"] == 3
