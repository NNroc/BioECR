tag_default = {"O": 0, "B": 1, "I": 2}
tag_docred = {"O": 0, "B-PER": 1, "I-PER": 2, "B-TIME": 3, "I-TIME": 4, "B-LOC": 5, "I-LOC": 6,
              "B-MISC": 7, "I-MISC": 8, "B-NUM": 9, "I-NUM": 10, "B-ORG": 11, "I-ORG": 12}
tag_BioRED = {"O": 0, "B-ChemicalEntity": 1, "I-ChemicalEntity": 2,
              "B-DiseaseOrPhenotypicFeature": 3, "I-DiseaseOrPhenotypicFeature": 4,
              "B-GeneOrGeneProduct": 5, "I-GeneOrGeneProduct": 6, "B-OrganismTaxon": 7, "I-OrganismTaxon": 8,
              "B-SequenceVariant": 9, "I-SequenceVariant": 10, "B-CellLine": 11, "I-CellLine": 12}
# DiseaseOrPhenotypicFeature, GeneOrGeneProduct, ChemicalEntity, OrganismTaxon, SequenceVariant, CellLine
tag_CDR = {"O": 0, "B-Chemical": 1, "I-Chemical": 2, "B-Disease": 3, "I-Disease": 4}
tag_GDA = {"O": 0, "B-Gene": 1, "I-Gene": 2, "B-Disease": 3, "I-Disease": 4}

e_types_description_name2id_cdr = {"Chemical": 0, "Disease": 1}
id2e_types_description_name_cdr = {0: "Chemical", 1: "Disease"}
e_types_description_name2id_gda = {"Gene": 0, "Disease": 1}
id2e_types_description_name_gda = {0: "Gene", 1: "Disease"}
e_types_description_name2id_biored = {"ChemicalEntity": 0, "DiseaseOrPhenotypicFeature": 1, "GeneOrGeneProduct": 2,
                                      "SequenceVariant": 3, "OrganismTaxon": 4, "CellLine": 5}
id2e_types_description_name_biored = {0: "ChemicalEntity", 1: "DiseaseOrPhenotypicFeature", 2: "GeneOrGeneProduct",
                                      3: "SequenceVariant", 4: "OrganismTaxon", 5: "CellLine"}

rel2id_biored = {'NR': 0, 'Association': 1, 'Positive_Correlation': 2, 'Bind': 3, 'Negative_Correlation': 4,
                 'Comparison': 5, 'Conversion': 6, 'Cotreatment': 7, 'Drug_Interaction': 8}
id2rel_biored = {0: 'NR', 1: 'Association', 2: 'Positive_Correlation', 3: 'Bind', 4: 'Negative_Correlation',
                 5: 'Comparison', 6: 'Conversion', 7: 'Cotreatment', 8: 'Drug_Interaction'}
rel2id_cdr = {'NR': 0, 'CID': 1}
id2rel_cdr = {0: 'NR', 1: 'CID'}
rel2id_gda = {'NR': 0, 'GDA': 1}
id2rel_gda = {0: 'NR', 1: 'GDA'}

pairs2rel_cdr = [(0, 1, 1)]
pairs2rel_gda = [(0, 1, 1)]
pairs2rel_biored = [(0, 0, 1), (0, 0, 2), (0, 0, 4), (0, 0, 5), (0, 0, 6), (0, 0, 7), (0, 0, 8),
                    (0, 1, 1), (0, 1, 2), (0, 1, 4),
                    (0, 2, 1), (0, 2, 2), (0, 2, 4), (0, 2, 3), (0, 2, 7), (0, 2, 8),
                    (0, 3, 1), (0, 3, 2), (0, 3, 4),
                    (1, 2, 1), (1, 2, 2), (1, 2, 4),
                    (1, 3, 1), (1, 3, 2), (1, 3, 4),
                    (2, 2, 1), (2, 2, 2), (2, 2, 4), (2, 2, 3)]

cdr_pairs = [("Chemical", "Disease")]
gda_pairs = [("Gene", "Disease")]
biored_pairs = [("DiseaseOrPhenotypicFeature", "GeneOrGeneProduct"),
                ("DiseaseOrPhenotypicFeature", "SequenceVariant"),
                ("ChemicalEntity", "ChemicalEntity"),
                ("ChemicalEntity", "GeneOrGeneProduct"),
                ("ChemicalEntity", "DiseaseOrPhenotypicFeature"),
                ("GeneOrGeneProduct", "GeneOrGeneProduct"),
                ("ChemicalEntity", "SequenceVariant")]

rel_type2rel_type_description = {"PER": "person", "TIME": "time", "LOC": "location", "MISC": "miscellaneous",
                                 "NUM": "number", "ORG": "organization",
                                 "ChemicalEntity": "chemical", "DiseaseOrPhenotypicFeature": "disease",
                                 "GeneOrGeneProduct": "gene", "OrganismTaxon": "organism", "SequenceVariant": "variant",
                                 "CellLine": "cell", "Chemical": "chemical", "Disease": "disease", "Gene": "gene",
                                 "CHEMICAL": "chemical", "GENE": "gene", }
