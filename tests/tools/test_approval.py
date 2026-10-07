import pytest

from jobshop.tools.approval import ApprovalAuthority, ApprovalError, schedule_digest
from tests.helpers import FakeClock, assign, schedule

BIND = dict(draft_id="D1", base_version=3, schedule_digest="abc")


@pytest.fixture
def auth():
    clock = FakeClock()
    return ApprovalAuthority(secret=b"k", ttl_s=300, clock=clock), clock


def test_a_fresh_token_for_the_exact_commit_is_accepted(auth):
    authority, _ = auth
    authority.consume(authority.issue(**BIND), **BIND)  # no exception


def test_a_token_works_only_once(auth):
    authority, _ = auth
    token = authority.issue(**BIND)
    authority.consume(token, **BIND)
    with pytest.raises(ApprovalError, match="already been used"):
        authority.consume(token, **BIND)


@pytest.mark.parametrize("override", [
    {"draft_id": "D2"},
    {"base_version": 4},
    {"schedule_digest": "other"},
])
def test_a_token_is_bound_to_draft_version_and_schedule(auth, override):
    authority, _ = auth
    token = authority.issue(**BIND)
    with pytest.raises(ApprovalError, match="different draft, version or schedule"):
        authority.consume(token, **{**BIND, **override})


def test_a_failed_check_does_not_burn_the_token(auth):
    authority, _ = auth
    token = authority.issue(**BIND)
    with pytest.raises(ApprovalError):
        authority.consume(token, **{**BIND, "draft_id": "D2"})
    authority.consume(token, **BIND)  # still good for the right commit


def test_a_token_expires(auth):
    authority, clock = auth
    token = authority.issue(**BIND)
    clock.advance(301)
    with pytest.raises(ApprovalError, match="expired"):
        authority.consume(token, **BIND)


def test_a_token_is_valid_right_up_to_its_expiry(auth):
    authority, clock = auth
    token = authority.issue(**BIND)
    clock.advance(299)
    authority.consume(token, **BIND)


def test_a_token_from_another_authority_is_rejected():
    other = ApprovalAuthority(secret=b"different")
    mine = ApprovalAuthority(secret=b"k")
    with pytest.raises(ApprovalError, match="signature"):
        mine.consume(other.issue(**BIND), **BIND)


def test_authorities_without_a_given_secret_get_random_ones():
    a, b = ApprovalAuthority(), ApprovalAuthority()
    with pytest.raises(ApprovalError):
        b.consume(a.issue(**BIND), **BIND)


def test_tampering_with_the_claim_or_signature_is_rejected(auth):
    authority, _ = auth
    body, signature = authority.issue(**BIND).split(".")
    forged_body = authority.issue(**{**BIND, "draft_id": "D9"}).split(".")[0]
    for bad in (f"{forged_body}.{signature}", f"{body}.{signature[:-2]}AA", f"{body}.AAAA"):
        with pytest.raises(ApprovalError):
            authority.consume(bad, **BIND)


@pytest.mark.parametrize("junk", ["", "garbage", "a.b.c", ".", "...", "😀", "x" * 500])
def test_malformed_tokens_raise_approval_error_not_other_exceptions(auth, junk):
    authority, _ = auth
    with pytest.raises(ApprovalError):
        authority.consume(junk, **BIND)


def test_non_string_tokens_are_rejected(auth):
    authority, _ = auth
    with pytest.raises(ApprovalError):
        authority.consume(None, **BIND)  # type: ignore[arg-type]


def test_schedule_digest_ignores_order_and_changes_with_content():
    a = assign("o1", "A", "M1", 0, 10)
    b = assign("o2", "A", "M1", 10, 20)
    assert schedule_digest(schedule([a, b])) == schedule_digest(schedule([b, a]))
    moved = assign("o2", "A", "M1", 11, 21)
    assert schedule_digest(schedule([a, b])) != schedule_digest(schedule([a, moved]))
