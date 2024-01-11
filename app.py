import streamlit as st
from dotenv import load_dotenv
import openai
import os
import base64

from PIL import Image
import io

from openai import AzureOpenAI

load_dotenv()

aoai_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

print(os.getenv("AZURE_OPENAI_API_VERSION"))

system_prompt = {"role":"system",
                 "content":
                 """
                 你是一个负责中小学数学教学的AI老师，你会识别图片中的数学题并以中文给出答案，然后你会对解题过程的思路进行详细分析。
                 如果图片中没有数学题，则直接回答“无法识别到数学题。
                 """
                }

#init chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], bytes):
            # if the message is an image, display it with st.image
            st.image(message["content"])
        else:
            st.markdown(message["content"])

chat_container = st.empty()

# Upload image in side bar
with st.sidebar:
    st.title('AI Math Teacher')
    
    with st.form("Upload and Process", True):
        upload_file = st.file_uploader("Choose an image with math problem...", type=["png", "jpg", "jpeg"])
        submitted = st.form_submit_button("Analyze")

        if submitted and upload_file:
            st.session_state.messages = []

            print(upload_file)
            image = Image.open(upload_file)
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='JPEG')
            img_bytes = img_bytes.getvalue()

            # Convert the image bytes to a base64 string
            img_base64 = base64.b64encode(img_bytes).decode('ascii')

            img_message = {"role": "user", "content": img_bytes, "content_base64": img_base64}
            st.session_state.messages.append(img_message)

            with chat_container.container():
                with st.chat_message("user"):
                    st.image(img_message["content"])

                with st.chat_message("assistant"):
    
                    #call GPT-4V to analyze the image
                    messages=[
                        system_prompt,
                        {
                            "role": "user", 
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{img_base64}",
                                        # "url": "https://p0.ssl.qhmsg.com/t01da1e4ef25adfd30a.jpg" //this is for test only
                                    }
                                }
                            ],
                        },
                    ]

                    extra_payload={
                        "enhancements": {
                            "ocr": {
                                "enabled": True
                            },
                            "grounding": {
                                "enabled": True
                            }
                        },
                        "dataSources": [
                        {
                            "type": "AzureComputerVision",
                            "parameters": {
                                "endpoint": "https://junlin-vision-westus.cognitiveservices.azure.com/",
                                "key": "09b11537f1dd4fd88fb0e6ea2f5d20b6"
                            }
                        }],
                    }

                    full_response = ""
                    message_placeholder = st.empty()
                    print(os.environ.get("AZURE_OPENAI_API_DEPLOYMENT"))
                    response = aoai_client.chat.completions.create(
                        model=os.environ.get("AZURE_OPENAI_API_DEPLOYMENT"),
                        # extra_body={
                        #     "enhancements": {
                        #         "ocr": {
                        #             "enabled": True
                        #         },
                        #         "grounding": {
                        #             "enabled": True
                        #         }
                        #     },
                        #     "dataSources": [
                        #         {
                        #             "type": "AzureComputerVision",
                        #             "parameters": {
                        #                 "endpoint": "https://junlin-vision-westus.cognitiveservices.azure.com/",
                        #                 "key": "09b11537f1dd4fd88fb0e6ea2f5d20b6"
                        #             }
                        #         }],
                        # },
                        messages=messages,
                        temperature=0,
                        max_tokens=2000,
                        top_p=0.95,
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                        stream=True,
                    )
                    
                    for chunk in response:
                        if chunk.choices[0].delta.content is not None:
                            full_response += (chunk.choices[0].delta.content or "")
                            message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)

            # headers = {
            #     "Content-Type": "application/json",
            #     "api-key": os.environ.get("AZURE_OPENAI_API_KEY"),
            # }

            # AZURE_OPENAI_API_ENDPOINT = os.environ.get("AZURE_OPENAI_API_ENDPOINT")
            # AZURE_OPENAI_API_DEPLOYMENT = os.environ.get("AZURE_OPENAI_API_DEPLOYMENT")
            # AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION")

            # gpt4v_endpoint = f"{AZURE_OPENAI_API_ENDPOINT}openai/deployments/{AZURE_OPENAI_API_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
            #Send request
            # try:
            #     response = requests.post(
            #         gpt4v_endpoint,
            #         headers=headers,
            #         json=payload,
            #         stream=True
            #     )
            #     response.raise_for_status()
            # except requests.exceptions.HTTPError as err:
            #     print(f"Failed to make the request: {err}")
            # print(response)


            # Seems GPT-4V API currently doens't support streaming
            
            # for line in mock_result:
            #     if line:
            #         answer_j = json.loads(line)
            #         print(answer_j)
            #         with chat_container.container():
            #             with st.chat_message("assistant"):
            #                 st.markdown(answer_j["choices"][0]["message"]["content"])
                    
            #         st.session_state.messages.append({"role": "assistant", "content": answer_j["choices"][0]["message"]["content"]})



if user_input := st.chat_input("What is up"):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

#     with st.chat_message("assistant"):
#         history_to_keep = 10
#         message_history = [
#             {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
#         ]
#         messages = [system_prompt] + message_history[-history_to_keep:]
#         print(messages)

#         message_placeholder = st.empty()
#         full_response = ""

#         for response in openai.ChatCompletion.create(
#             engine=os.environ.get("AZURE_OPENAI_API_DEPLOYMENT"),
#             messages=messages,
#             temperature=0,
#             max_tokens=2000,
#             top_p=0.95,
#             frequency_penalty=0.0,
#             presence_penalty=0.0,
#             stop=None,
#             stream=True
#         ):
#             if response.choices and 'delta' in response.choices[0] and 'content' in response.choices[0].delta:
#                 full_response += (response.choices[0].delta.content or "")
#                 message_placeholder.markdown(full_response + "▌")


#         message_placeholder.markdown(full_response)

#     st.session_state.messages.append({"role": "assistant", "content": full_response})


