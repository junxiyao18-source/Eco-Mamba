"""
Ecological Behavior Group Mapping for Animal Kingdom Dataset
Hierarchical organization of 140 action classes into 16 major groups
"""

# ============================================================================
# 16 Major Behavior Groups (Ecological Hierarchy)
# ============================================================================
AK_GROUP_NAMES = [
    "Affection",              # 0  - Displaying affection and bonding
    "Aggressive",             # 1  - Aggressive and combative behaviors
    "Aquatic",                # 2  - Swimming and water-related behaviors
    "Capture",                # 3  - Predation and prey capture
    "Climb",                  # 4  - Climbing and scaling
    "Courtship",              # 5  - Mating and courtship displays
    "Escape",                 # 6  - Fleeing and escape behaviors
    "Feed",                   # 7  - Feeding and foraging
    "Groom",                  # 8  - Grooming and hygiene
    "Jump",                   # 9  - Jumping and leaping
    "Manipulate",             # 10 - Object manipulation and tooling
    "Movement",               # 11 - Locomotion and basic movement
    "Rest",                   # 12 - Resting and inactivity
    "Reproduction",           # 13 - Reproduction and birth-related
    "Shelter",                # 14 - Nest and shelter construction
    "Social",                 # 15 - Social and group behaviors
]

assert len(AK_GROUP_NAMES) == 16, "Must have exactly 16 groups"
NUM_GROUPS = 16

# ============================================================================
# Animal Kingdom Action (0-139) to Group ID (0-15) Mapping
# ============================================================================
AK_ACTION_TO_GROUP_ID = {
    # Affection (0)
    4: 0,      # Being carried
    5: 0,      # Being carried in mouth
    12: 0,     # Carrying
    13: 0,     # Carrying in mouth
    63: 0,     # Holding hands
    65: 0,     # Hugging
    73: 0,     # Licking
    75: 0,     # Lying on top
    106: 0,    # Showing affection
    
    # Aggressive (1)
    1: 1,      # Attacking
    8: 1,      # Biting
    14: 1,     # Chasing
    18: 1,     # Competing for dominance
    26: 1,     # Displaying defensive pose
    47: 1,     # Fighting
    54: 1,     # Getting bullied
    96: 1,     # Retaliating
    
    # Aquatic (2)
    28: 2,     # Diving
    37: 2,     # Drifting
    52: 2,     # Flying (can include gliding over water)
    57: 2,     # Gliding
    101: 2,    # Running on water
    121: 2,    # Surfacing
    123: 2,    # Swimming
    124: 2,    # Swimming in circles
    
    # Capture (3)
    7: 3,      # Being eaten
    91: 3,     # Preying
    112: 3,    # Spitting venom
    119: 3,    # Stinging
    137: 3,    # Wrapping itself around prey
    138: 3,    # Wrapping prey
    
    # Climb (4)
    16: 4,     # Climbing
    59: 4,     # Hanging
    
    # Courtship (5)
    81: 5,     # Performing sexual display
    84: 5,     # Performing copulatory mounting
    85: 5,     # Performing sexual exploration
    86: 5,     # Performing sexual pursuit
    130: 5,    # Unmounting
    
    # Escape (6)
    42: 6,     # Escaping
    46: 6,     # Falling
    51: 6,     # Fleeing
    97: 6,     # Retreating
    
    # Feed (7)
    38: 7,     # Drinking
    40: 7,     # Eating
    45: 7,     # Exploring (often food-related)
    79: 7,     # Panting (after feeding/chasing)
    
    # Groom (8)
    58: 8,     # Grooming
    82: 8,     # Performing allo-grooming
    90: 8,     # Preening
    92: 8,     # Puffing its throat
    99: 8,     # Rubbing its head
    103: 8,    # Shaking
    104: 8,    # Shaking head
    135: 8,    # Washing
    
    # Jump (9)
    64: 9,     # Hopping
    67: 9,     # Jumping
    
    # Manipulate (10)
    76: 10,    # Manipulating object
    93: 10,    # Pulling
    
    # Movement (11)
    0: 11,     # Abseiling
    17: 11,    # Coiling
    20: 11,    # Dancing on water
    29: 11,    # Doing a back kick
    30: 11,    # Doing a backward tilt
    31: 11,    # Doing a chin dip
    32: 11,    # Doing a face dip
    33: 11,    # Doing a neck raise
    34: 11,    # Doing a side tilt
    35: 11,    # Doing push up
    36: 11,    # Doing somersault
    48: 11,    # Flapping
    49: 11,    # Flapping tail
    50: 11,    # Flapping its ears
    69: 11,    # Landing
    72: 11,    # Leaning
    78: 11,    # Moving
    98: 11,    # Rolling
    100: 11,   # Running
    115: 11,   # Squatting
    116: 11,   # Standing
    125: 11,   # Swinging
    126: 11,   # Tail swishing
    128: 11,   # Turning around
    133: 11,   # Walking
    134: 11,   # Walking on water
    
    # Rest (12)
    21: 12,    # Dead
    39: 12,    # Dying
    68: 12,    # Keeping still
    70: 12,    # Lying down
    74: 12,    # Lying on its side
    95: 12,    # Resting
    108: 12,   # Sitting
    109: 12,   # Sleeping
    110: 12,   # Sleeping in its nest
    
    # Reproduction (13)
    55: 13,    # Giving birth
    71: 13,    # Laying eggs
    
    # Shelter (14)
    9: 14,     # Building nest
    41: 14,    # Entering its nest
    44: 14,    # Exiting nest
    
    # Social (15) - remaining actions
    2: 15,     # Attending
    3: 15,     # Barking
    6: 15,     # Being dragged
    10: 15,    # Calling
    11: 15,    # Camouflaging
    15: 15,    # Chirping
    19: 15,    # Dancing
    22: 15,    # Defecating
    23: 15,    # Defensive rearing
    24: 15,    # Detaching as a parasite
    25: 15,    # Digging
    27: 15,    # Disturbing another animal
    43: 15,    # Exiting cocoon
    53: 15,    # Gasping for air
    56: 15,    # Giving off light
    60: 15,    # Hatching
    61: 15,    # Having a flehmen response
    62: 15,    # Hissing
    66: 15,    # Immobilized
    77: 15,    # Molting
    80: 15,    # Pecking
    83: 15,    # Performing allo-preening
    87: 15,    # Playing
    88: 15,    # Playing dead
    89: 15,    # Pounding
    94: 15,    # Rattling
    102: 15,   # Sensing
    105: 15,   # Sharing food
    107: 15,   # Sinking
    111: 15,   # Spitting
    113: 15,   # Spreading
    114: 15,   # Spreading wings
    117: 15,   # Standing in alert
    118: 15,   # Startled
    120: 15,   # Struggling
    122: 15,   # Swaying
    127: 15,   # Trapped
    129: 15,   # Undergoing chrysalis
    131: 15,   # Unrolling
    132: 15,   # Urinating
    136: 15,   # Waving
    139: 15,   # Yawning
}

# Validate the static Animal Kingdom ethogram at import time.
assert len(AK_ACTION_TO_GROUP_ID) == 140, f"Expected 140 actions, got {len(AK_ACTION_TO_GROUP_ID)}"
assert min(AK_ACTION_TO_GROUP_ID.keys()) == 0, "Min action ID should be 0"
assert max(AK_ACTION_TO_GROUP_ID.keys()) == 139, "Max action ID should be 139"
assert min(AK_ACTION_TO_GROUP_ID.values()) == 0, "Min group ID should be 0"
assert max(AK_ACTION_TO_GROUP_ID.values()) == 15, "Max group ID should be 15"
