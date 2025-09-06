import torch
from einops import rearrange ## Sử dụng để thay đổi thứ tự trục (axes)
                             ## Gộp nhiều chiều thành 1 chiều
                             ## Tách 1 chiều thành nhiều con
                             ## Transpose tensor
from transformers import AutoTokenizer, AutoModel
# gloal params
MODEL = None
TOKENIZER = None
BERT_MODEL_DIM = 768
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_tokenizer():
    global TOKENIZER
    if TOKENIZER is None:
        TOKENIZER = AutoTokenizer.from_pretrained('bert-base-cased')
    return TOKENIZER

def get_bert():
    global MODEL
    if MODEL is None:
        MODEL = AutoModel.from_pretrained('bert-base-cased')
    MODEL.to(device)
    MODEL.eval()
    return MODEL

# tokenize
def tokenize(texts, add_special_tokens = True):
    if not isinstance(texts, (list, tuple)):
        texts = [texts]
    tokenizer = get_tokenizer()
    encoding = tokenizer(
        texts,
        add_special_tokens = add_special_tokens, # add [CLS], [SEP]
        padding = True,
        return_tensors = 'pt'
    )
    return encoding.input_ids

# embedding
@torch.no_grad()
def bert_embed(token_ids, return_cls_repr = False, eps = 1e-8, pad_id = 0.):
    model = get_bert()
    tokenizer = get_tokenizer()
    pad_id = tokenizer.pad_token_id
    token_ids = token_ids.to(device)
    mask = (token_ids != pad_id).to(device)

    outputs = model(
        input_ids = token_ids,
        attention_mask = mask,
        output_hidden_states = True
    )
    hidden_state = outputs.hidden_states[-1] # [b, n, d]
    if return_cls_repr:
        return hidden_state[:, 0] # [b, d]
    
    if mask is None:
        return hidden_state.mean(dim=1)
    mask = mask[:, 1:]
    mask = rearrange(mask, 'b n -> b n 1')
    numer = (hidden_state[:, 1:] * mask).sum(dim=1)
    denom = mask.sum(dim=1)
    masked_mean = numer / (denom + eps)
    return masked_mean

if __name__=="__main__":
    texts = [
        "Hello world!",
        "Video Diffusion Models"
    ]
    token_ids = tokenize(texts)
    print("[DEBUG TOKENIZER] Token IDs shape: ", token_ids)
    embeddings = bert_embed(token_ids, return_cls_repr=True)
    print("[DEBUG BERT EMBEDDING] Embedding shape: ", embeddings)