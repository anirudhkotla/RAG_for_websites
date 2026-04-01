def structured_chunk(text, chunk_size=500, max_lines_per_chunk=20):
    """
    Splits text into chunks for embeddings.

    Args:
        text (str): Full text to split.
        chunk_size (int): Approximate character length per chunk.
        max_lines_per_chunk (int): Max number of lines per chunk.

    Returns:
        List[str]: List of text chunks.
    """
    chunks = []
    current_chunk = []
    current_length = 0

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    
    for line in lines:
        # If adding this line exceeds limits, save current chunk
        if current_length + len(line) > chunk_size or len(current_chunk) >= max_lines_per_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

        current_chunk.append(line)
        current_length += len(line)

    # Add remaining text
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks