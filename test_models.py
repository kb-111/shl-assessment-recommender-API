import google.generativeai as genai

genai.configure(api_key="AIzaSyDss1Ckwuo7oUehqX3zNvAPjeCXcvHsiHo")

for m in genai.list_models():
    print(m.name)