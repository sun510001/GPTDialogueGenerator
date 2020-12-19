# -*- coding:utf-8 -*-

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def chat(tokenizer, model):
    # Let's chat for 5 lines
    # list_ = ["Who touched my food?", "Why would you do that?"]
    dia_log = []
    while True:
        for step in range(5):
        # for i in range(len(list_)):
            user_utterance = input("User: ")
            # user_utterance = list_[i]
            # print(user_utterance)
            dia_log.append(user_utterance)
            # encode the new user input, add the eos_token and return a tensor in Pytorch
            new_user_input_ids = tokenizer.encode(user_utterance + tokenizer.eos_token, return_tensors='pt')
            # step_reset = 0
            # append the new user input tokens to the chat history
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

            # generated a response while limiting the total chat history to 1000 tokens,
            chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

            # pretty print last ouput tokens from bot
            r_utterance = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
            print("DialoGPT: {}".format(r_utterance))
            dia_log.append(r_utterance)

        print("--->", dia_log)
        print("new round ... ")


if __name__ == '__main__':
    # tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
    # model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

    chat(tokenizer, model)