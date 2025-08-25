import os
import base64
from datetime import datetime
from mistralai import Mistral

class Mistralmodel:
    def __init__(self):
        self.api_key = "6TwrzTjQNpn5E2TyCrm4DuaMKOUVDkog"#os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment variables")
        self.client = Mistral(api_key=self.api_key)
        self.text_model = "mistral-large-latest"
        self.vision_model = "pixtral-large-latest" # "pixtral-12b-2409"
    
    def encode_image(self, image_path):
        """Encode image to base64 format"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except FileNotFoundError:
            print(f"❌ Error: File {image_path} not found")
            return None
        except Exception as e:
            print(f"❌ Image encoding error: {e}")
            return None
    
    def get_image_mime_type(self, image_path):
        """Determine MIME type based on file extension"""
        file_ext = os.path.splitext(image_path)[1].lower()
        mime_types = {
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg'
        }
        return mime_types.get(file_ext, 'image/jpeg')
    
    def chat_with_text(self, text, system_prompt, example, assistant_prompt):
        """Text-only conversation"""
        try:
            chat_response = self.client.chat.complete(
                model=self.text_model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    # Example
                    {
                        "role": "user",
                        "content": example
                    },
                    {
                        "role": "assistant",
                        "content": assistant_prompt
                    },
                    {
                        "role": "user",
                        "content": text,
                    },
                ]
            )
            
            response_text = chat_response.choices[0].message.content
            
            return response_text
            
        except Exception as e:
            print(f"❌ Text model call error: {e}")
            return None
    
    def chat_with_vision(self, text, image_path, system_prompt, example, assistant_prompt):
        """Image + text conversation (using vision model)"""
        try:
            print(f"🖼️  Encoding image: {os.path.basename(image_path)}")
            base64_image = self.encode_image(image_path)
            
            mime_type = self.get_image_mime_type(image_path)
            
            # Prepare message content
            content = [
                {
                    "type": "text",
                    "text": text
                },
                {
                    "type": "image_url",
                    "image_url": f"data:{mime_type};base64,{base64_image}"
                }
            ]
            
            print("🤖 Calling vision model...")
            chat_response = self.client.chat.complete(
                model=self.vision_model,
                messages=[
                    
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    # Example
                    {
                        "role": "user",
                        "content": example
                    },
                    {
                        "role": "assistant",
                        "content": assistant_prompt
                    },
                    {
                        "role": "user",
                        "content": text,
                    },
                    {
                        "role": "user",
                        "content": content
                    }
                ]
            )
            
            response_text = chat_response.choices[0].message.content
            
            return response_text
            
        except Exception as e:
            print(f"❌ Vision model call error: {e}")
            return None
    

    


