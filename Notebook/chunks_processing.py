import json

def extract_page_md_map(json_data):
    page_md_map = {}
    for page_data in json_data.get("pages", []):
        page_number = str(page_data.get("page"))
        md_content = page_data.get("md", "")
        page_md_map[page_number] = md_content
    return page_md_map

def chunk_markdown_by_topic(page_md_dict):
    chunks = []
    chunk_counter = 1  # Start chunk_order from 1

    # Persistent across pages
    current_topic = None
    current_subtopic = None

    for page_str in sorted(page_md_dict.keys(), key=lambda x: int(x)):
        page = int(page_str)
        md = page_md_dict[page_str]
        buffer = []

        lines = md.splitlines()

        def flush_buffer():
            nonlocal buffer, chunk_counter
            if buffer:
                content = "\n".join(buffer).strip()
                if current_topic or current_subtopic:
                    chunks.append({
                        "chunk_order": chunk_counter,
                        "page": page,
                        "topic": current_topic,
                        "subtopic": current_subtopic,
                        "content": content
                    })
                    chunk_counter += 1
                buffer.clear()

        for line in lines:
            stripped = line.strip()

            if stripped.startswith("# "):  # Level 1 Heading
                flush_buffer()
                current_topic = stripped[2:].strip()
                current_subtopic = None
            elif stripped.startswith("## "):  # Level 2 Heading
                flush_buffer()
                current_subtopic = stripped[3:].strip()
            elif stripped.startswith("### "):  # Level 3 Heading
                flush_buffer()
                current_subtopic = stripped[4:].strip()
            else:
                buffer.append(stripped)

        flush_buffer()

    return chunks

if __name__ == "__main__":
    # Load original JSON
    with open("output_v2.json", "r") as f:
        data = json.load(f)

    # Step 1: Extract page-wise markdown
    page_md_dict = extract_page_md_map(data)

    # Step 2: Chunk by topic/subtopic (retaining across pages)
    topic_chunks = chunk_markdown_by_topic(page_md_dict)

    # Step 3: Save result
    with open("topic_chunks.json", "w") as f:
        json.dump(topic_chunks, f, indent=2)

    print(f"✅ Done! Extracted {len(topic_chunks)} topic chunks with topic retention across pages.")
