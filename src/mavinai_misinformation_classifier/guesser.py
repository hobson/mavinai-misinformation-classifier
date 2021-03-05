import string
import random


def random_guesser_v1(passage):
    """Takes in a string and returns a dictionary with
    6 keys for the binary values representing
    features of the string, and another six keys
    representing the scores of those features.

    Note: for the scores, 0 represents no score,
    while -1 represents '?'

    Arguments
    ---------
    passage: string to be converted to dictionary with feature scores.
    """

    features_dict = {}

    features = ["is_accountability", "is_unobjectivity",
                "is_inaccuracy", "is_fact-basedness",
                "is_influential", "is_opinionated"]

    features_score = ["score_accountability", "score_unobjectivity",
                      "score_inaccuracy", "score_fact-basedness",
                      "score_influential", "score_opinionated"]

    for i in range(len(features)):
        features_dict[features[i]] = random.randrange(2)
        features_dict[features_score[i]] = random.randrange(-1, 4)

    return features_dict


def random_guesser_v2(passage):
    """Takes in a string and returns a dictionary with
    6 features as keys and tuple values representing the
    binary value and score, respectively.

    Note: for the scores, 0 represents no score, while -1 represents '?'

    Arguments
    ---------
    passage: string to be converted to dictionary with feature scores.
    """

    features_dict = {}

    features = ["accountability", "unobjectivity",
                "inaccuracy", "fact-basedness",
                "influential", "opinionated"]

    for feature in features:
        features_dict[feature] = (random.randrange(2), random.randrange(-1, 4))

    return features_dict
