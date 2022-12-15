import streamlit as st

st.title("興味深掘り支援AI")
st.subheader("あなたが興味あることを入力してください")
st.write("広がり依頼は同系列の内容，深まり依頼は関連する内容が提案されます")
#st.subheader("例：私は釣りと[MASK]が好きです")
message = st.text_input("例：釣り")



def recommend_AI(message):
    from transformers import T5Tokenizer, RobertaForMaskedLM
    

    tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-roberta-base")
    tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading

    model = RobertaForMaskedLM.from_pretrained("rinna/japanese-roberta-base")
    st.subheader("AIがあなたにおすすめする内容")
    message = '私は'+ message + 'と[MASK]が好きです'

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
        if token not in  ['セックス']:
            st.write(i+1, token)

if st.button("広がり依頼"):
    recommend_AI(message)




#st.subheader("深まり支援")
#message2 = st.text_input('例：ドライブ')



def recommend_AI2(message2):
    from transformers import T5Tokenizer, RobertaForMaskedLM
    

    tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-roberta-base")
    tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading

    model = RobertaForMaskedLM.from_pretrained("rinna/japanese-roberta-base")
    st.subheader("AIがあなたにおすすめする内容")
    message2 = '私は'+ message2 + 'の[MASK]をよく知っています'

    # original text
    text_orig = message2

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
        if token not in  ['セックス']:
            st.write(i+1, token)

if st.button("深まり依頼"):
    recommend_AI2(message)


keyword1 = st.text_input("キーワード1")
keyword2 = st.text_input("キーワード2")



def recommend_AI3(keyword1,keyword2):
    from transformers import T5Tokenizer, RobertaForMaskedLM
    

    tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-roberta-base")
    tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading

    model = RobertaForMaskedLM.from_pretrained("rinna/japanese-roberta-base")
    st.subheader("AIがあなたにおすすめする内容")
    message12 = '私は'+ keyword1 + 'の' + keyword2 + 'の[MASK]についてよく知っています'

    # original text
    text_orig = message12

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
        if token not in  ['セックス']:
            st.write(i+1, token)

if st.button("さらに深まり依頼"):
    recommend_AI2(keyword1,keyword2)