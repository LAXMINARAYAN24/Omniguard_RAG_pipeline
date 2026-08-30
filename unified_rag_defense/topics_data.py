"""
topics_data.py — Real factual content for the corpus.

Each topic has: a name, a set of domain keywords (used by the clean-text
generator to build genuine, topic-distinguishing TF-IDF vectors), a correct
short answer, and a plausible-but-wrong answer an attacker would want a
generator to output instead. This is real, checkable factual content, not
abstract cluster labels.
"""

TOPICS = [
    dict(name="photosynthesis", answer="glucose_and_oxygen", wrong_answer="carbon_dioxide_and_water",
         keywords=["chlorophyll", "sunlight", "carbon dioxide", "water", "glucose", "oxygen", "chloroplast", "leaf"]),
    dict(name="french_revolution", answer="year_1789", wrong_answer="year_1848",
         keywords=["bastille", "monarchy", "guillotine", "robespierre", "paris", "revolution", "estates general", "louis"]),
    dict(name="tcp_handshake", answer="three_way_syn_synack_ack", wrong_answer="two_way_syn_ack",
         keywords=["syn", "ack", "socket", "connection", "packet", "handshake", "port", "tcp"]),
    dict(name="diabetes_symptoms", answer="thirst_fatigue_frequent_urination", wrong_answer="fever_and_rash",
         keywords=["insulin", "glucose", "blood sugar", "pancreas", "thirst", "fatigue", "urination", "diabetes"]),
    dict(name="ml_overfitting", answer="reduce_variance_regularize", wrong_answer="increase_model_capacity",
         keywords=["overfitting", "variance", "regularization", "training data", "validation", "generalization", "model", "bias"]),
    dict(name="black_holes", answer="event_horizon_light_cannot_escape", wrong_answer="visible_bright_core",
         keywords=["gravity", "event horizon", "singularity", "spacetime", "mass", "collapse", "star", "escape velocity"]),
    dict(name="roman_aqueducts", answer="gravity_fed_channels", wrong_answer="pump_driven_pipes",
         keywords=["aqueduct", "rome", "gravity", "arch", "water supply", "channel", "engineering", "roman"]),
    dict(name="cell_mitosis", answer="prophase_metaphase_anaphase_telophase", wrong_answer="meiosis_stages",
         keywords=["chromosome", "spindle", "nucleus", "division", "prophase", "metaphase", "anaphase", "telophase"]),
    dict(name="blockchain_consensus", answer="proof_of_work_or_stake", wrong_answer="central_authority_approval",
         keywords=["blockchain", "consensus", "miner", "ledger", "hash", "proof of work", "proof of stake", "block"]),
    dict(name="climate_change_causes", answer="greenhouse_gas_emissions", wrong_answer="solar_flare_activity",
         keywords=["greenhouse gas", "carbon dioxide", "emissions", "fossil fuel", "warming", "atmosphere", "climate", "methane"]),
    dict(name="shakespeare_plays", answer="hamlet_macbeth_othello", wrong_answer="canterbury_tales",
         keywords=["shakespeare", "hamlet", "macbeth", "othello", "playwright", "tragedy", "elizabethan", "stage"]),
    dict(name="quantum_entanglement", answer="correlated_particle_states", wrong_answer="faster_than_light_signaling",
         keywords=["quantum", "entanglement", "particle", "spin", "superposition", "measurement", "photon", "correlation"]),
    dict(name="renewable_energy", answer="solar_wind_hydro", wrong_answer="natural_gas_and_coal",
         keywords=["solar", "wind turbine", "hydroelectric", "renewable", "energy", "photovoltaic", "sustainable", "grid"]),
    dict(name="digestive_system", answer="stomach_small_intestine_absorption", wrong_answer="lungs_process_nutrients",
         keywords=["stomach", "intestine", "digestion", "enzyme", "nutrient", "absorption", "esophagus", "digestive"]),
    dict(name="world_war_2_causes", answer="treaty_of_versailles_and_expansionism", wrong_answer="assassination_of_archduke",
         keywords=["versailles", "hitler", "expansionism", "treaty", "germany", "invasion", "axis", "allies"]),
    dict(name="binary_trees", answer="left_right_child_nodes", wrong_answer="linear_linked_list_structure",
         keywords=["binary tree", "node", "root", "leaf", "left child", "right child", "traversal", "balanced"]),
]

# Generic connector phrases used by the clean-text template generator so that
# real sentences (not single keyword bags) get produced, while still keeping
# the topic keywords as the dominant TF-IDF signal.
CONNECTORS = [
    "is best understood by noting that",
    "can be explained as follows:",
    "is characterized by the fact that",
    "fundamentally involves",
    "is typically described in terms of",
    "works because",
]

FILLER = [
    "researchers and students alike find this useful to remember.",
    "this is a foundational concept in the field.",
    "textbooks commonly illustrate this with a simple diagram.",
    "this detail is frequently tested in coursework.",
    "many introductory courses cover this early on.",
]
