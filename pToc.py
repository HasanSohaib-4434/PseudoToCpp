import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

with open('tokenizer_pseudo.pkl', 'rb') as f:
    tokenizer_pseudo = pickle.load(f)

with open('tokenizer_cpp.pkl', 'rb') as f:
    tokenizer_cpp = pickle.load(f)

model = load_model('transformer_model.keras')

def generate_code(pseudo_text, max_seq_len=100):
    pseudo_seq = tokenizer_pseudo.texts_to_sequences([pseudo_text])
    pseudo_padded = pad_sequences(pseudo_seq, maxlen=max_seq_len, padding='post')

    generated_tokens = [tokenizer_cpp.word_index['<start>']]

    for _ in range(max_seq_len):
        decoder_input = pad_sequences([generated_tokens], maxlen=max_seq_len, padding='post')
        pred_probs = model.predict([pseudo_padded, decoder_input], verbose=0)
        next_token = np.argmax(pred_probs[0, len(generated_tokens) - 1])

        if next_token == tokenizer_cpp.word_index.get('<end>', 0):
            break

        generated_tokens.append(next_token)

    cpp_code = ' '.join(tokenizer_cpp.index_word.get(token, '') for token in generated_tokens if token > 0)
    return cpp_code

st.title("Pseudo Code to C++ Converter")

pseudo_input = st.text_area("Enter Pseudo Code:")

if st.button("Generate C++ Code"):
    if pseudo_input.strip():
        cpp_output = generate_code(pseudo_input)
        st.text_area("Generated C++ Code:", cpp_output, height=200)
    else:
        st.warning("Please enter some pseudo code.")
