import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

with open('pseudocode_tokenizer.pkl', 'rb') as f:
    pseudocode_tokenizer = pickle.load(f)
with open('cpp_tokenizer.pkl', 'rb') as f:
    cpp_tokenizer = pickle.load(f)

transformer = tf.keras.models.load_model('transformer_model.keras', compile=False)

def generate_cpp_code(pseudocode, max_len=150):
    input_seq = pseudocode_tokenizer.texts_to_sequences(["<sos> " + pseudocode + " <eos>"])
    input_seq = pad_sequences(input_seq, maxlen=max_len, padding='post')
    
    pred_seq = transformer.predict(input_seq)
    pred_indices = np.argmax(pred_seq, axis=-1)[0]
    cpp_tokens = [cpp_tokenizer.index_word.get(idx, '') for idx in pred_indices if idx > 0]
    
    return ' '.join(cpp_tokens).replace('<sos>', '').replace('<eos>', '').strip()

st.title("📝 Pseudocode to C++ Code Converter")
st.write("Enter your pseudocode and get the equivalent C++ code!")

pseudocode_input = st.text_area("Enter Pseudocode:", "")

if st.button("Generate C++ Code"):
    if pseudocode_input.strip():
        cpp_output = generate_cpp_code(pseudocode_input)
        st.subheader("Generated C++ Code:")
        st.code(cpp_output, language="cpp")
    else:
        st.warning("Please enter some pseudocode.")

