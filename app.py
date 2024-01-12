import streamlit as st
from dotenv import load_dotenv
import os
import base64

from PIL import Image
import io

from openai import AzureOpenAI

load_dotenv()

endpoint = os.getenv("AZURE_OPENAI_API_ENDPOINT")
deployment = os.environ.get("AZURE_OPENAI_API_DEPLOYMENT")
aoai_client = AzureOpenAI(
    base_url=f"{endpoint}/openai/deployments/{deployment}/extensions",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

cv_endpoint = os.getenv("AZURE_CV_ENDPOINT")
cv_key = os.getenv("AZURE_CV_KEY")

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

            # Get thunmbnail image
            max_size= (600, 450)
            image.thumbnail(max_size)
            img_thumbnail_bytes = io.BytesIO()
            image.save(img_thumbnail_bytes, format='JPEG')

            img_message = {"role": "user", "content": img_thumbnail_bytes, "content_base64": img_base64}
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

                    full_response = ""
                    message_placeholder = st.empty()
                    
                    response = aoai_client.chat.completions.create(
                        model=deployment,
                        messages=messages,
                        extra_body={
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
                                        "endpoint": cv_endpoint,
                                        "key": cv_key,
                                    }
                                }],
                        },
                        temperature=0,
                        max_tokens=2000,
                        top_p=0.95,
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                        stream=True,
                    )
                    
                    for chunk in response:
                        
                        if hasattr(chunk.choices[0], 'messages') and len(chunk.choices[0].messages) > 0:
                            message = chunk.choices[0].messages[0]
                            if 'delta' in message and 'content' in message['delta']:
                                content = message['delta']['content']

                                if 'role'  not in message['delta']: 
                                    full_response += content
                                    message_placeholder.markdown(full_response + "▌")
                                
                    message_placeholder.markdown(full_response)
                            
                    