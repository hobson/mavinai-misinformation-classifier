import pytest
from random_guesser import random_guesser_v1
from random_guesser import random_guesser_v2


def test_random_guesser_v1():
    labels = random_guesser_v1("Lorem ipsum dolor sit amet, consectetur adipiscing elit.")
    assert isinstance(labels, dict)
    assert len(labels) == 14
    assert "score_accountability" in labels
    with pytest.raises(KeyError):
        labels['intentionally_testing_undefined_key']

    # will fail because of spelling/grammar errors in the label names:
    assert "is_biased" in labels


def test_random_guesser_v2():
    labels = random_guesser_v2("Lorem ipsum dolor sit amet, consectetur adipiscing elit.")
    assert isinstance(labels, dict)
    assert len(labels) == 14
    assert "score_accountability" in labels
    with pytest.raises(TypeError):
        labels['score_accountability'] + 'cannot add float to str'

    # will fail because of spelling/grammar errors in the label names:
    assert "is_biased" in labels


if __name__ == '__main__':
    test_random_guesser_v1()
    test_random_guesser_v2()
