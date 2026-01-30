import os
from collections import defaultdict
import json
import re
from gatenlp import Document

DATA_DIR = "./data_philips"
ENTITIES_DIR = "./entities"
OUTPUT_DIR = "./gatenlp_output"

# Configuration for clustering
ENABLE_CLUSTERING = True  # Set to True to enable entity clustering


def load_texts(data_dir: str) -> dict:
    text_doc_dict = defaultdict(str)
    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        if os.path.isfile(file_path):
            with open(file_path, "r") as fileReader:
                content = fileReader.read()
                text_doc_dict[file] = content
    return text_doc_dict


def build_document(doc_name: str, doc_text: str, entities_dir: str, enable_clustering: bool) -> Document:
    base_doc = Document(doc_text)
    base_doc.name = doc_name.replace(".txt", "")
    ent_set = base_doc.annset("entities_")
    ent_json_path = os.path.join(entities_dir, doc_name.replace(".txt", ".json"))

    # Dictionary to store clusters by entity type, then by normalized text
    clusters_by_type = {}
    cluster_id = 1  # Progressive cluster ID counter

    with open(ent_json_path, "r") as json_file:
        ent_obj = json.load(json_file)
        for ent_type in ent_obj.keys():
            mapped_ent_type = ent_type.upper()

            if enable_clustering:
                clusters_by_type[mapped_ent_type] = {}

            for entity_mention in ent_obj[ent_type]:
                # Escape special regex characters in the entity mention
                escaped_entity = re.escape(entity_mention)

                # Create a more flexible pattern for entities with special characters
                # Use word boundary only at the beginning if it starts with a word character
                # Use word boundary only at the end if it ends with a word character
                start_boundary = r"\b" if entity_mention[0].isalnum() else r"(?<!\w)"
                end_boundary = r"\b" if entity_mention[-1].isalnum() else r"(?!\w)"
                pattern = start_boundary + escaped_entity + end_boundary

                # Find all occurrences using regex
                counter = 0
                for match in re.finditer(pattern, doc_text, re.IGNORECASE):
                    counter += 1
                    start = match.start()
                    end = match.end()
                    actual_text = doc_text[start:end]  # Get the actual matched text (preserves case)

                    # Add annotation with mapped entity type and get the actual annotation object with its ID
                    annotation = ent_set.add(start, end, mapped_ent_type, features={"text": actual_text})
                    actual_annotation_id = annotation.id  # Get the real GateNLP annotation ID

                    if enable_clustering:
                        # Use the actual matched text for clustering (normalized), only within same type
                        normalized_key = actual_text.lower().strip()

                        if normalized_key not in clusters_by_type[mapped_ent_type]:
                            clusters_by_type[mapped_ent_type][normalized_key] = {
                                "id": cluster_id,  # Unique cluster ID
                                "title": actual_text,  # Use first occurrence as cluster title
                                "type": mapped_ent_type,
                                "nelements": 0,
                                "mentions": []
                            }
                            cluster_id += 1  # Increment for next cluster

                        clusters_by_type[mapped_ent_type][normalized_key]["mentions"].append({
                            "id": actual_annotation_id,  # Use the real GateNLP annotation ID
                            "mention": actual_text
                        })
                        clusters_by_type[mapped_ent_type][normalized_key]["nelements"] = len(
                            clusters_by_type[mapped_ent_type][normalized_key]["mentions"]
                        )

                print(f"Entity '{entity_mention}' in {doc_name}: found {counter} matches")

    # Add clusters to document features in the correct format (only if clustering is enabled)
    if enable_clustering and clusters_by_type:
        all_clusters = []
        for ent_type, type_clusters in clusters_by_type.items():
            all_clusters.extend(list(type_clusters.values()))

        if all_clusters:
            base_doc.features["clusters"] = {
                "entities_": all_clusters
            }

    return base_doc


def save_documents(docs: list, output_dir: str):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for doc in docs:
        file_path = os.path.join(output_dir, doc.name + ".json")
        with open(file_path, "w") as fileWriter:
            json.dump(doc.to_dict(), fileWriter, indent=4)


def main():
    print(f"CLUSTERING_ENABLED={ENABLE_CLUSTERING}")

    text_doc_dict = load_texts(DATA_DIR)
    gdocs = []
    for key, value in text_doc_dict.items():
        gdocs.append(build_document(key, value, ENTITIES_DIR, ENABLE_CLUSTERING))

    save_documents(gdocs, OUTPUT_DIR)


if __name__ == "__main__":
    main()
