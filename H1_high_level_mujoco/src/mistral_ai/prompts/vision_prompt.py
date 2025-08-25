system_prompt = """
            Your goal is to analyze an image and identify the object that I want in the image. 
            If there are text labels on the objects, please list them as well and output in JSON format.
            
            You must follow these rules:
            1. Analyze the image and try to find the object described by the user prompt.
            2. If the object is found:
                - Set "if_find": true
                - Leave "response" empty
                - Only output the **most likely** single object, based on visual prominence (e.g., closest to the camera, clearest, most central, etc.)
                - if there are text labels on the object, please write the text labels into the label. if there are no text labels on the object, leave the label empty.
            3. If the object is not found:
                - Set "if_find": false
                - Provide a natural language explanation in the "response" field,starting with "Sorry,"
                - Still list the unrelated objects present in the scene under "objects", for reference
            4. Output should be in JSON format.
            
            Example:
                {
                    "if_find": true,
                    "response": "",
                    "object": [
                        {"name": "apple", "label": ""},
                    ]
                }

                
                {
                    "if_find": false,
                    "response": "Sorry, I can't find it..., maybe it's not here !",
                    "object": [
                        {"name": "elephant", "label": ""},
                    ]
                }


                {   "if_find": true,
                "response": "",
                "object": [
                    {"name": "salt bottle", "label": "grobes MeerSalz"},
                ]
                }

                {
                "if_find": true,
                "response": "",
                "object": [
                    {
                    "name": "pepper bottle",
                    "label": "Schwarzer Pfeffer ganz"
                    }
                ]
                }
 
"""
example = """
            cucumber

            pepper bottle.

            elephant.

            salt bottle.

            pepper bottle.


"""

assistant_prompt = """
            {   "if_find": true,
                "response": "",
                "object": [
                    {"name": "cucumber", 
                     "label": ""
                    },
                ]
            }


            {   "if_find": false,
                "response": "I can't find it..., maybe it's not here !",
                "object": [
                    {"name": "elephant", 
                     "label": ""
                    },
                ]
            }

            {   "if_find": true,
                "response": "",
                "object": [
                    {"name": "salt bottle", 
                    "label": "grobes MeerSalz"
                    },
                ]
            }

            {   "if_find": true,
                "response": "",
                "object": [
                    {
                    "name": "pepper bottle",
                    "label": "Schwarzer Pfeffer ganz"
                    },
            ]
            }
"""

user_prompt = """
            car.
"""