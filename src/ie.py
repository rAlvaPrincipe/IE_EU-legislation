from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import re 
import json
from pathlib import Path

openai_api_key=""
model  = "gpt-4o-mini"
prompt = "ner_extractor_v1.2.txt"   # "extractor" for a list of entitites, "annotator" for text with tags


with open("prompts/" + prompt, "r", encoding="utf-8") as f:
    prompt_template = f.read()
llm = ChatOpenAI(model_name=model, temperature=0, openai_api_key=openai_api_key)


def extract_entities(doc):
    prompt_filled = prompt_template.replace("<INPUT_TEXT>\n[put you text here]", f"<INPUT_TEXT>\n{doc}")
    response = llm([HumanMessage(content=prompt_filled)])
    raw_output = response.content

    try:
        entities = json.loads(raw_output)
    except json.JSONDecodeError:
        entities = raw_output

    return entities



for doc_file in Path("./data").glob("*.txt"):
    with open(doc_file, "r", encoding="utf-8") as f:
        doc = f.read()
        print(doc_file)

        entities = extract_entities(doc)
        print(entities)

        base_name = re.search(r"data/(.*)\.txt", str(doc_file)).group(1)
        destination_file = Path("./entities/" + base_name)
        destination_file.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(entities, dict):
            with open(destination_file.with_suffix(".json"), "w", encoding="utf-8") as out:
                json.dump(entities, out, indent=2, ensure_ascii=False)
        else:
            with open(destination_file.with_suffix(".txt"), "w", encoding="utf-8") as out:
                out.write(entities)