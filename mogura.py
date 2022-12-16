import streamlit as st
import pandas as pd
from PIL import Image

image1 = Image.open('logo.jpg')
image2 = Image.open('mogu.jpeg')

col1, col2, col3, col4, col5, col5 = st.columns(6)
with col1:
   st.image(image1,width=100)
with col2:
   st.image(image2,width=100)

st.title("好きなことの探究支援AIアプリ")
st.subheader("ステップ1：好きなことが見つからない人のきっかけ作り")
st.write("少しでも興味あることを入力してください")
message = st.text_input("↓単語で入力してください　例：プログラミング")



@st.cache
def BERT():
    from transformers import T5Tokenizer, RobertaForMaskedLM
    

    tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-roberta-base")
    tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading

    model = RobertaForMaskedLM.from_pretrained("rinna/japanese-roberta-base")

    return model,tokenizer

model,tokenizer = BERT()




def recommend_AI1(message1):

    model,tokenizer = BERT()
    
    st.subheader("AIがあなたにおすすめするキーワード")
    message1 = '私は'+ message1 + 'と[MASK]が大好きです。'

    # original text
    text_orig = message1

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
        predictions = outputs[0][0, masked_idx].topk(15)

    #print(text_orig)

    for i, index_t in enumerate(predictions.indices):
        index = index_t.item()
        token = tokenizer.convert_ids_to_tokens([index])[0]
        if token not in  ['セックス','ポルノ','セクシー']:
            if token == '<unk>':
                token = '-'
                st.write(i+1, token)
            else:
                st.write(i+1, token)

if st.button("AIに好きを広げるヒントを聞く"):
    recommend_AI1(message)





def recommend_AI2(message2):

    model,tokenizer = BERT()

    st.subheader("AIがあなたにおすすめするキーワード")
    message2 = '私は'+ message2 + 'が大好きです。だから、[MASK]について詳しく知っています。'

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
        predictions = outputs[0][0, masked_idx].topk(15)

    #print(text_orig)

    for i, index_t in enumerate(predictions.indices):
        index = index_t.item()
        token = tokenizer.convert_ids_to_tokens([index])[0]
        if token not in  ['セックス','ポルノ','セクシー']:
            if token == '<unk>':
                token = '-'
                st.write(i+1, token)
            else:
                st.write(i+1, token)

if st.button("AIに好きを深めるヒントを聞く"):
    recommend_AI2(message)




st.subheader("")

st.subheader("ステップ2：好きなことの概念化")
st.write("自分の好きなことを抽象的に捉えましょう")

message = st.text_input("↓単語で入力してください")



def recommend_AI3(message3):

    model,tokenizer = BERT()

    st.subheader("AIがあなたにおすすめするキーワード")
    #message2 = '私は'+ message2 + 'が大好きです。それは、[MASK]だからです。'
    message3 = '私は'+ message3 + 'の、[MASK]なところが大好きです。'

    # original text
    text_orig = message3

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
        predictions = outputs[0][0, masked_idx].topk(15)

    #print(text_orig)

    for i, index_t in enumerate(predictions.indices):
        index = index_t.item()
        token = tokenizer.convert_ids_to_tokens([index])[0]
        if token not in  ['セックス','ポルノ','セクシー']:
            if token == '<unk>':
                token = '-'
                st.write(i+1, token)
            else:
                st.write(i+1, token)

if st.button("AIに好きな概念のヒントを聞く"):
    recommend_AI3(message)



def recommend_AI4(message4):

    model,tokenizer = BERT()

    st.subheader("AIがあなたにおすすめするキーワード")
    message4 = '私は'+ message4 + 'が大好きです。それは、[MASK]だからです。'
    #message2 = '私は'+ message2 + 'に対して[MASK]というイメージを持っています。'

    # original text
    text_orig = message4

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
        predictions = outputs[0][0, masked_idx].topk(15)

    #print(text_orig)

    for i, index_t in enumerate(predictions.indices):
        index = index_t.item()
        token = tokenizer.convert_ids_to_tokens([index])[0]
        if token not in  ['セックス','ポルノ','セクシー']:
            if token == '<unk>':
                token = '-'
                st.write(i+1, token)
            else:
                st.write(i+1, token)

if st.button("AIに好きな理由のヒントを聞く"):
    recommend_AI4(message)





st.subheader("")

st.subheader("ステップ3：好きなことの探究")
st.write("キーワードを組み合わせて自分の好きを探究しましょう")

number = st.slider('キーワード数', 1, 10, 2)
keyword_list = []
for i in range(number):
    a = 'keyword' + str(i+1)
    a = st.text_input("キーワード"+str(i+1))
    keyword_list.append(a)




def recommend_AI5(keyword_list):

    model,tokenizer = BERT()
    

    concatenated = '私は'

    for k in keyword_list:
        concatenated += k + '、'

    st.write(pd.DataFrame({'keyword':keyword_list}))    

    st.subheader("AIがあなたにおすすめするキーワード")

    concatenated += 'の[MASK]について調査をしています。'
    

    # original text
    text_orig = concatenated

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
        predictions = outputs[0][0, masked_idx].topk(15)

    #print(text_orig)

    for i, index_t in enumerate(predictions.indices):
        index = index_t.item()
        token = tokenizer.convert_ids_to_tokens([index])[0]
        if token not in  ['セックス','ポルノ','セクシー']:
            if token == '<unk>':
                token = '-'
                st.write(i+1, token)
            else:
                st.write(i+1, token)
            
    

if st.button("AIにさらに好きを深めるヒントを聞く"):
    recommend_AI5(keyword_list)