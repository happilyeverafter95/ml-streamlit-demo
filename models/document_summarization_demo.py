from typing import List, Dict, Tuple

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import streamlit as st


class QuestionAnswerer:
    def __init__(self) -> None:
        self.generator: tf.python.saved_model.load.Loader
        self.answers: List[str]

    def load_generator(self) -> None:
        self.generator = hub.Module('https://tfhub.dev/google/bertseq2seq/roberta24_bbc/1')

    def get_answer(self, input_documents: List[str]) -> Tuple[str, float]:
        tokenized_question = self.model.signatures['question_encoder'](
                    tf.constant([question]))

        predicted = np.inner(tokenized_question['outputs'], self.tokenized_answers['outputs'])
        return self.answers[np.argmax(predicted)], max(predicted[0])


@st.cache(allow_output_mutation=True)
def instantiate_model():
    qa_bot = QuestionAnswerer()
    qa_bot.load_model()
    qa_bot.load_answers()
    return qa_bot


if __name__ == '__main__':
    st.title('QA Bot')
    question = st.text_area('Try asking our bot a question ...')
    qa_bot = instantiate_model()

    if question:
        answer, confidence = qa_bot.get_answer(question)
        st.markdown(f'Bot Answer: {answer}')
        st.markdown(f'Confidence: {confidence}')