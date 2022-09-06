from transformers import T5Tokenizer, RobertaForMaskedLM
import streamlit as st

tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-roberta-base")
tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading

model = RobertaForMaskedLM.from_pretrained("rinna/japanese-roberta-base")

st.title("興味深掘り支援AI")
st.subheader("あなたの興味あることを[MASK]と合わせて入力してください")
message = st.text_input("メッセージ")


if message:
    # original text
    text_orig = message

    # prepend [CLS]
    text = "[CLS]" + text_orig

    # tokenize
    tokens = tokenizer.tokenize(text)
    #print(tokens)

    #print('mask index :' , tokens.index('[MASK]'))
    masked_idx = tokens.index('[MASK]')

    tokens[masked_idx] = tokenizer.mask_token

    # convert to ids
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    # convert to tensor
    import torch
    token_tensor = torch.tensor([token_ids])

    # get the top 10 predictions of the masked token
    model = model.eval()
    with torch.no_grad():
        outputs = model(token_tensor)
        predictions = outputs[0][0, masked_idx].topk(10)

    #print(text_orig)

    for i, index_t in enumerate(predictions.indices):
        index = index_t.item()
        token = tokenizer.convert_ids_to_tokens([index])[0]
        st.write(i, token)