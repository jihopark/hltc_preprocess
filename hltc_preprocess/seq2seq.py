

def add_padding(tokenized_texts, max_len, padding_token="<pad>"):
    for i,t in enumerate(tokenized_texts):
        for _ in range(max_len - len(t)):
            tokenized_texts[i].append(padding_token)
    return tokenized_texts

def add_position_token(tokenized_texts, special_token, position):
    assert position == "front" or "back"
    for i in range(len(tokenized_texts)):
        if position == "front":
            tokenized_texts[i].insert(0, special_token)
        else:
            tokenized_texts[i].append(special_token)
    return tokenized_texts
