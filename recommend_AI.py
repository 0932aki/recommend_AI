import streamlit as st
import pandas as pd
from PIL import Image


st.title("好きなこと探しを支援するAI")

image = Image.open('ai.png')

st.image(image)

st.subheader("ステージ1：好きなこと探しのきっかけ作り")
st.write("少しでも好きなことや興味あることを入力してください")

#st.subheader("例：私は釣りと[MASK]が好きです")
message = st.text_input("↓単語で入力してください　例：プログラミング")



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

if st.button("AIに興味を広げるヒントを教えてもらう"):
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

if st.button("AIに興味を深めるヒントを教えてもらう"):
    recommend_AI2(message)

st.subheader("")

st.subheader("ステージ2：好きなことの探究")
st.write("キーワードを増やして好きなことを深めていきましょう")

number = st.slider('キーワード数', 1, 10, 2)
keyword_list = []
for i in range(number):
    a = 'keyword' + str(i+1)
    a = st.text_input("キーワード"+str(i+1))
    keyword_list.append(a)

#keyword1 = st.text_input("キーワード1")
#keyword2 = st.text_input("キーワード2")



def recommend_AI3(keyword_list):
    from transformers import T5Tokenizer, RobertaForMaskedLM
    

    tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-roberta-base")
    tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading

    model = RobertaForMaskedLM.from_pretrained("rinna/japanese-roberta-base")
    

    message12 = '私は'

    for k in keyword_list:
        message12 += k + 'の'

    st.write(pd.DataFrame({'keyword':keyword_list}))    

    st.subheader("AIがあなたにおすすめする内容")

    message12 += '[MASK]についてよく知っています'
    

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
            
    

if st.button("AIにさらに興味を深めるヒントを教えてもらう"):
    recommend_AI3(keyword_list)