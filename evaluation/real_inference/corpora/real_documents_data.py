"""
Real-world heterogeneous multi-domain corpus definitions for Track B Evaluation.
Covers:
  - NIST Physics & Metrology Standards
  - NASA Aerospace Engineering & Planetary Science
  - NCBI/PubMed Biomedical & Pharmacology
  - CISA Cybersecurity & Cryptography Protocols
  - SEC Financial Regulation & Macroeconomics

Includes both verified ground-truth documents and natural heterogeneous distractor documents.
"""

from typing import List, Dict, Any

REAL_DOMAINS_DATA: List[Dict[str, Any]] = [
    # -------------------------------------------------------------------------
    # Topic 1: Physics / Metrology (NIST)
    # -------------------------------------------------------------------------
    {
        "topic_id": "nist_gravitational_constant",
        "domain": "Physics & Metrology",
        "query": "What is the CODATA recommended value for the Newtonian constant of gravitation G?",
        "ground_truth_answer": "6.67430e-11 m^3 kg^-1 s^-2",
        "key_facts": [
            "6.67430",
            "6.67430e-11",
            "6.67430 x 10^-11",
            "6.67430 × 10⁻¹¹"
        ],
        "clean_documents": [
            {
                "doc_id": "nist_sp330_g_std",
                "title": "NIST Special Publication 330: The International System of Units (SI)",
                "publisher_domain": "nist.gov",
                "source_id": "nist_metrology_bulletin_2022",
                "tenant_id": "physics_research",
                "text": (
                    "The Newtonian constant of gravitation, denoted as G or G_N, is a fundamental physical constant "
                    "involved in the calculation of gravitational effects in Isaac Newton's law of universal gravitation "
                    "and Albert Einstein's general theory of relativity. According to the CODATA internationally recommended "
                    "values of the fundamental physical constants, the standard value of G is 6.67430(15) x 10^-11 m^3 kg^-1 s^-2. "
                    "The relative standard uncertainty of this recommended value is 2.2 x 10^-5, determined via precision torsion balance "
                    "and atom interferometry experiments across national metrology institutes."
                )
            },
            {
                "doc_id": "bipm_si_constants_overview",
                "title": "BIPM Metrologia: Fundamental Constants and SI Units",
                "publisher_domain": "bipm.org",
                "source_id": "bipm_metrologia_vol56",
                "tenant_id": "physics_research",
                "text": (
                    "In the modern SI framework, while the speed of light c, Planck constant h, and elementary charge e "
                    "have exact defined values, the gravitational constant G remains an experimentally determined quantity. "
                    "The 2018 CODATA adjustment established G = 6.67430e-11 m^3 kg^-1 s^-2 with standard uncertainty 0.00015e-11 m^3 kg^-1 s^-2. "
                    "Measurements at BIPM in Sevres and NIST in Gaithersburg corroborate this valuation within experimental margins."
                )
            },
            {
                "doc_id": "physics_distractor_planck_length",
                "title": "Quantum Mechanics & Natural Units: The Planck Scale",
                "publisher_domain": "cern.ch",
                "source_id": "cern_physics_reports_2021",
                "tenant_id": "physics_research",
                "text": (
                    "The Planck length l_P is defined as sqrt(hbar * G / c^3), which evaluates to approximately 1.616255 x 10^-35 meters. "
                    "In theoretical quantum gravity, the Planck length represents the scale below which smooth classical spacetime "
                    "geometry breaks down into quantum fluctuations or quantum foam. General relativity and quantum field theory "
                    "reach a regime of non-renormalizable divergence at this scale."
                )
            }
        ]
    },

    # -------------------------------------------------------------------------
    # Topic 2: Aerospace / Planetary Science (NASA / ESA)
    # -------------------------------------------------------------------------
    {
        "topic_id": "nasa_perseverance_landing",
        "domain": "Aerospace & Planetary Science",
        "query": "Where did NASA's Perseverance rover land on Mars and what was its primary target feature?",
        "ground_truth_answer": "Jezero Crater",
        "key_facts": [
            "Jezero Crater",
            "Jezero",
            "ancient river delta",
            "Isidis Planitia"
        ],
        "clean_documents": [
            {
                "doc_id": "nasa_mars2020_landing_report",
                "title": "NASA Mars 2020 Mission: Entry, Descent, and Landing Overview",
                "publisher_domain": "nasa.gov",
                "source_id": "nasa_jpl_press_kit_2021",
                "tenant_id": "aerospace_ops",
                "text": (
                    "On February 18, 2021, NASA's Mars 2020 mission successfully landed the Perseverance rover inside Jezero Crater. "
                    "Jezero Crater is a 45-kilometer-wide crater located on the western edge of Isidis Planitia, a giant impact basin "
                    "just north of the Martian equator. Scientists selected Jezero because orbital spectral data showed rich clay minerals "
                    "and a preserved ancient fan-delta deposit, indicating that liquid water once flowed through the crater lake."
                )
            },
            {
                "doc_id": "esa_mars_express_jezero_study",
                "title": "ESA Mars Express Orbital Imaging of Jezero Fan Deposits",
                "publisher_domain": "esa.int",
                "source_id": "esa_planetary_science_2021",
                "tenant_id": "aerospace_ops",
                "text": (
                    "High-resolution stereoscopic imaging from ESA's Mars Express orbiter provides structural constraints "
                    "on the Jezero Crater delta where the NASA Perseverance rover is actively collecting core samples. "
                    "The sedimentary layers show distinct bottomset, foreset, and topset deltaic strata consistent with sustained "
                    "fluvial discharge into an ancient open-basin paleolake during the Noachian and early Hesperian epochs."
                )
            },
            {
                "doc_id": "aerospace_distractor_curiosity_gale",
                "title": "Curiosity Rover Science Highlights in Gale Crater",
                "publisher_domain": "nasa.gov",
                "source_id": "nasa_curiosity_mission_update",
                "tenant_id": "aerospace_ops",
                "text": (
                    "NASA's Curiosity rover, which landed in Gale Crater on August 6, 2012, continues to ascend the lower slopes "
                    "of Mount Sharp (Aeolis Mons). Curiosity's SAM and CheMin analytical instruments have detected organic molecules "
                    "and sulfur-rich clay minerals, demonstrating that Gale Crater possessed long-lived habitable lacustrine environments."
                )
            }
        ]
    },

    # -------------------------------------------------------------------------
    # Topic 3: Biomedical & Pharmacology (NCBI / FDA)
    # -------------------------------------------------------------------------
    {
        "topic_id": "biomed_paxlovid_mechanism",
        "domain": "Biomedicine & Pharmacology",
        "query": "What is the biochemical mechanism of action of Nirmatrelvir in Paxlovid?",
        "ground_truth_answer": "Inhibition of SARS-CoV-2 main protease (Mpro or 3CLpro)",
        "key_facts": [
            "main protease",
            "Mpro",
            "3CLpro",
            "3C-like protease",
            "covalent peptidomimetic inhibitor"
        ],
        "clean_documents": [
            {
                "doc_id": "fda_paxlovid_labeling",
                "title": "FDA Emergency Use Authorization: Paxlovid (Nirmatrelvir and Ritonavir)",
                "publisher_domain": "fda.gov",
                "source_id": "fda_cder_approval_2021",
                "tenant_id": "biomed_clinical",
                "text": (
                    "Paxlovid consists of nirmatrelvir co-packaged with ritonavir. Nirmatrelvir is an orally bioavailable SARS-CoV-2 "
                    "main protease (Mpro, also known as 3CLpro or 3C-like protease) inhibitor. Inhibition of SARS-CoV-2 Mpro renders "
                    "the enzyme incapable of processing polyprotein precursors pp1a and pp1ab into functional non-structural proteins (nsps), "
                    "thereby halting viral RNA replication. Ritonavir is co-administered as a pharmacokinetic booster to inhibit CYP3A-mediated metabolism."
                )
            },
            {
                "doc_id": "ncbi_paxlovid_structure_nature",
                "title": "Structural Basis of SARS-CoV-2 3CLpro Inhibition by Nirmatrelvir",
                "publisher_domain": "nih.gov",
                "source_id": "ncbi_pmc_nature_2022",
                "tenant_id": "biomed_clinical",
                "text": (
                    "X-ray crystallographic studies demonstrate that nirmatrelvir binds directly to the catalytic dyad (His41 and Cys145) "
                    "of the SARS-CoV-2 main protease (Mpro). The nitrile carbon of nirmatrelvir forms a reversible covalent thioimidate adduct "
                    "with the catalytic Cys145 residue, effectively blocking the substrate-binding cleft and preventing cleavage of viral polyproteins."
                )
            },
            {
                "doc_id": "biomed_distractor_remdesivir_rdra",
                "title": "Mechanism of Action of Remdesivir against Viral RNA-Dependent RNA Polymerase",
                "publisher_domain": "nih.gov",
                "source_id": "ncbi_pmc_antiviral_2021",
                "tenant_id": "biomed_clinical",
                "text": (
                    "Remdesivir is an adenosine nucleotide prodrug that targets the viral RNA-dependent RNA polymerase (RdRp, nsp12). "
                    "Upon intracellular activation to its active triphosphate form (GS-443902), it incorporates into the nascent RNA chain, "
                    "inducing delayed chain termination three to five nucleotides downstream of the incorporation site."
                )
            }
        ]
    },

    # -------------------------------------------------------------------------
    # Topic 4: Cybersecurity & Cryptography (CISA / NIST)
    # -------------------------------------------------------------------------
    {
        "topic_id": "cisa_post_quantum_crystals",
        "domain": "Cybersecurity & Cryptography",
        "query": "Which lattice-based algorithm was standardized by NIST as FIPS 203 for Post-Quantum Key Encapsulation?",
        "ground_truth_answer": "ML-KEM (derived from CRYSTALS-Kyber)",
        "key_facts": [
            "ML-KEM",
            "CRYSTALS-Kyber",
            "Kyber",
            "FIPS 203",
            "Module Learning with Errors",
            "MLWE"
        ],
        "clean_documents": [
            {
                "doc_id": "nist_fips_203_release",
                "title": "NIST FIPS 203: Module-Lattice-Based Key-Encapsulation Mechanism Standard",
                "publisher_domain": "nist.gov",
                "source_id": "nist_csrc_fips203_2024",
                "tenant_id": "cyber_security",
                "text": (
                    "Federal Information Processing Standards Publication (FIPS) 203 specifies the Module-Lattice-Based Key-Encapsulation "
                    "Mechanism (ML-KEM), which is derived from the CRYSTALS-Kyber submission to the NIST Post-Quantum Cryptography (PQC) Standardization Project. "
                    "ML-KEM provides asymmetric key encapsulation security based on the computational hardness of the Module Learning with Errors (MLWE) "
                    "problem against both classical and cryptographically relevant quantum computers (CRQCs)."
                )
            },
            {
                "doc_id": "cisa_pqc_migration_guide",
                "title": "CISA Guidance: Migrating Federal Systems to Post-Quantum Standards",
                "publisher_domain": "cisa.gov",
                "source_id": "cisa_cyber_alert_pqc_2024",
                "tenant_id": "cyber_security",
                "text": (
                    "CISA urges critical infrastructure operators to transition from legacy RSA and elliptic-curve Diffie-Hellman (ECDH) "
                    "to the standardized post-quantum key encapsulation algorithm ML-KEM (FIPS 203, CRYSTALS-Kyber). "
                    "ML-KEM parameter sets ML-KEM-512, ML-KEM-768, and ML-KEM-1024 correspond respectively to NIST Security Categories 1, 3, and 5."
                )
            },
            {
                "doc_id": "cyber_distractor_sphincs_fips205",
                "title": "NIST FIPS 205: Stateless Hash-Based Digital Signature Standard (SLH-DSA)",
                "publisher_domain": "nist.gov",
                "source_id": "nist_csrc_fips205_2024",
                "tenant_id": "cyber_security",
                "text": (
                    "FIPS 205 standardizes the Stateless Hash-Based Digital Signature Algorithm (SLH-DSA), derived from SPHINCS+. "
                    "Unlike lattice-based schemes such as ML-DSA (FIPS 204) and ML-KEM (FIPS 203), SLH-DSA relies solely on the security "
                    "properties of cryptographically secure hash functions (such as SHA-256 and SHAKE-256), offering a conservative fallback."
                )
            }
        ]
    },

    # -------------------------------------------------------------------------
    # Topic 5: Financial Regulation & Macroeconomics (SEC / Fed)
    # -------------------------------------------------------------------------
    {
        "topic_id": "sec_t1_settlement_rule",
        "domain": "Financial Markets & Regulation",
        "query": "What is the official standard settlement cycle for US securities transactions under SEC Rule 15c6-1 adopted in May 2024?",
        "ground_truth_answer": "T+1 (one business day after trade date)",
        "key_facts": [
            "T+1",
            "one business day",
            "T + 1",
            "Rule 15c6-1",
            "trade date plus one"
        ],
        "clean_documents": [
            {
                "doc_id": "sec_final_rule_t1_shortening",
                "title": "SEC Release No. 34-96930: Shortening the Securities Transaction Settlement Cycle",
                "publisher_domain": "sec.gov",
                "source_id": "sec_regulatory_actions_2024",
                "tenant_id": "finance_compliance",
                "text": (
                    "The Securities and Exchange Commission adopted amendments to Rule 15c6-1 under the Securities Exchange Act of 1934 "
                    "to shorten the standard settlement cycle for most broker-dealer transactions in securities from two business days "
                    "after the trade date (T+2) to one business day after the trade date (T+1). The compliance date for the transition "
                    "to T+1 settlement was established as May 28, 2024. Shortening the settlement cycle mitigates credit, market, and liquidity risk."
                )
            },
            {
                "doc_id": "dtcc_t1_implementation_analysis",
                "title": "DTCC Post-Trade Whitepaper: Seamless T+1 Operational Transition",
                "publisher_domain": "dtcc.com",
                "source_id": "dtcc_market_infrastructure_2024",
                "tenant_id": "finance_compliance",
                "text": (
                    "The Depository Trust & Clearing Corporation (DTCC), in partnership with SIFMA and ICI, confirmed the successful industry-wide "
                    "implementation of the T+1 settlement mandate on May 28, 2024. Under T+1, trade allocation, confirmation, and affirmation "
                    "occur on trade date T by 9:00 PM ET, reducing National Securities Clearing Corporation (NSCC) clearing fund margin requirements."
                )
            },
            {
                "doc_id": "finance_distractor_basel_iii_lcr",
                "title": "Federal Reserve Supervisory Assessment of Basel III Liquidity Coverage Ratio",
                "publisher_domain": "federalreserve.gov",
                "source_id": "fed_supervision_report_2023",
                "tenant_id": "finance_compliance",
                "text": (
                    "The Liquidity Coverage Ratio (LCR) requires covered banking organizations to maintain an amount of high-quality "
                    "liquid assets (HQLA) that is no less than 100 percent of its total net cash outflows over a 30-day stress period. "
                    "HQLA includes Level 1 central bank reserves and US Treasury securities, as well as qualifying Level 2A and 2B assets."
                )
            }
        ]
    }
]
