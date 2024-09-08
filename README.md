# Extracting Nutrition from product details

Using Langchain and LLM to extract nutritional info from product names



LLMs can help to answer the contextual infortaion such as ingredients in some item upto great extent. For example consider the following product that we would like to extract the products nutritions.

```python
# !pip install langchain
# !pip install langchain_community
import openai, os, pandas as pd
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import ChatPromptTemplate
import json
import time

import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 1000)


load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

def get_product_template(product, model):
    
    from langchain.chat_models import ChatOpenAI
    
    review_template = """Take a deep breath. For the following product, extract the following information and output as python dictionary:

                        kwd:              List the top ingredients used to make this product.

                        nutritions:       List the top nutritions found in this product.

                        text: {product}

                        """
        
    prompt = ChatPromptTemplate.from_template(template=review_template)
    
    messages = prompt.format_messages(product=product)
    
    chat_model= ChatOpenAI(temperature=0.2, model=model)
    
    response = chat_model(messages)

    return response.content


def get_formatted_response(response):
    return json.loads(response[response.find('{'):response.rfind('}')+1])

question='can you plz find me ingredients for items in triple quotes """poppi ginger lime prebiotic soda""" '

st=time.time()
response=get_product_template(question,model="gpt-4o")
en=time.time()

print(f'time take {en-st}')

```
#### Product: poppi ginger lime prebiotic soda

#### Ingredients: 
```python
print(" ".join(get_formatted_response(response)['kwd']))
##### carbonated water, apple cider vinegar, lemon juice, ginger juice, lime juice, organic cane sugar, natural flavors, stevia

print(f"\nNutrition for this product\n")
##### calories, total fat, sodium, total carbohydrates, sugars, protein
```
