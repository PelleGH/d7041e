from transformers import AutoModel, AutoTokenizer


def minilm():
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    return model, tokenizer

def mpnet():
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

    return model, tokenizer

def multilingual():
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    model = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    return model, tokenizer

def get_models():
    return [
        ("all-MiniLM-L6-v2", minilm()),
        ("all-mpnet-base-v2", mpnet()),
        ("paraphrase-multilingual-MiniLM-L12-v2", multilingual()),
    ]