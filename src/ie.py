from anyio import Path
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pathlib import Path
import json
import os



MODEL = "gpt-4o-mini"

CORPUS_DIR = Path("./data_philips")
PROMPT_FILE = Path("./prompts/ner_extractor_philips_v1.2.txt")
OUTPUT_DIR = Path("./entities")

def load_env(path: Path = Path(".env")):
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value



def extract_entities(doc_text: str, prompt_template: str, llm: ChatOpenAI):
    prompt_filled = prompt_template.replace(
        "<INPUT_TEXT>\n[put you text here]",
        f"<INPUT_TEXT>\n{doc_text}"
    )

    response = llm.invoke([HumanMessage(content=prompt_filled)])
    raw_output = (response.content or "").strip()

    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        return raw_output


def save_output(result, out_base_path: Path):
    out_base_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(result, dict) or isinstance(result, list):
        out_path = out_base_path.with_suffix(".json")
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        out_path = out_base_path.with_suffix(".txt")
        out_path.write_text(str(result), encoding="utf-8")




def main():
    load_env()
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env or your environment.")

    prompt_template = PROMPT_FILE.read_text(encoding="utf-8")
    llm = ChatOpenAI(model=MODEL,temperature=0, openai_api_key=openai_api_key)


    for doc_path in sorted(CORPUS_DIR.glob("*.txt")):
        print(f"Processing: {doc_path.name}")

        doc_text = doc_path.read_text(encoding="utf-8")
        result = extract_entities(doc_text, prompt_template, llm)

        out_base = OUTPUT_DIR / doc_path.stem
        save_output(result, out_base)


if __name__ == "__main__":
    main()
