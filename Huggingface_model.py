import json
import requests

API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
headers = {"Authorization": "Bearer api_VwEDgHShdakgiOvZvfgKDIHPTkhgqODdyK"}

def query(payload):
	data = json.dumps(payload)
	response = requests.request("POST", API_URL, headers=headers, data=data)
	return json.loads(response.content.decode("utf-8"))

def getuserchat(user_chat,user_chat_list,hugging_face_model_list):

    past_user_inputs = []
    generated_responses = []
    if len(user_chat_list) > 1:
        for x in user_chat_list:
            past_user_inputs.append(x)
        for y in hugging_face_model_list:
            generated_responses.append(y)
        
        while len(past_user_inputs) > 6:
            past_user_inputs.pop(0)
            generated_responses.pop(0)
            
        data = query(
        {
            "inputs": {
            
                "past_user_inputs": past_user_inputs,
                "generated_responses": generated_responses,
                "text": user_chat,
            },
        }
        )
    else:
      data = query(
         {
             "inputs": {
            
                 "text": user_chat

             },
         }
         )

    return data

{'generated_text': ' Hey, how are you? I just got back from walking my dog. Do you have any pets?', 
'conversation': {'past_user_inputs': ['hey'], 'generated_responses': [' Hey, how are you? I just got back from walking my dog. Do you have any pets?']}}


