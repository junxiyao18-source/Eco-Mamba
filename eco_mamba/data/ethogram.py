"""Animal Kingdom ethogram hierarchy used by Eco-Mamba.

The group order and memberships follow Sections 3.1/4.5.5 and Table 2 of the
paper. Table 2 contains a few shifted numeric IDs, so memberships are resolved
against the canonical Animal Kingdom action names and stored here as IDs.
"""

AK_GROUP_NAMES = [
    "Affection", "Aggressive", "Communication", "Death", "Defensive",
    "Feeding", "General behavior", "Life cycle", "Maintenance", "Movement",
    "Predation", "Resting", "Sexual behavior", "Shelter-related",
    "Social interactions", "Transport",
]

NUM_GROUPS = len(AK_GROUP_NAMES)

_ACTIONS_BY_GROUP = {
    0: [63, 65, 106],
    1: [1, 14, 17, 18, 27, 47, 62, 89, 91, 94, 112, 137, 138],
    2: [3, 10, 15, 56, 136],
    3: [21, 39],
    4: [11, 23, 26, 29, 42, 51, 96, 97, 117, 120],
    5: [8, 38, 40, 80, 105],
    6: [50, 53, 68, 75, 79, 111, 118, 119, 126],
    7: [43, 55, 60, 71, 77, 129],
    8: [22, 31, 32, 58, 73, 82, 83, 90, 99, 103, 104, 132, 135],
    9: [
        0, 16, 24, 25, 28, 30, 33, 34, 36, 37, 41, 44, 46, 48, 49, 52,
        57, 59, 64, 66, 67, 69, 70, 72, 74, 76, 78, 93, 98, 100, 101,
        107, 108, 113, 114, 115, 116, 121, 122, 123, 125, 128, 130, 131,
        133, 134,
    ],
    10: [7, 54, 88, 127],
    11: [95, 109, 110, 139],
    12: [19, 20, 35, 81, 84, 85, 86, 92],
    13: [9],
    14: [2, 45, 61, 87, 102, 124],
    15: [4, 5, 6, 12, 13],
}

AK_ACTION_TO_GROUP_ID = {
    action_id: group_id
    for group_id, action_ids in _ACTIONS_BY_GROUP.items()
    for action_id in action_ids
}

assert len(AK_GROUP_NAMES) == 16
assert set(AK_ACTION_TO_GROUP_ID) == set(range(140))
assert set(AK_ACTION_TO_GROUP_ID.values()) == set(range(NUM_GROUPS))
