import google.generativeai as genai

# ✅ Use your valid API key from https://aistudio.google.com/app/apikey
genai.configure(api_key="AIzaSyCstJAYDgQIt3pxQn0_trq9hCTOTO12fGo")

# ✅ Print available models to confirm access
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name)

# ✅ Use the correct model name as listed above (no "models/" prefix)
model = genai.GenerativeModel("gemini-2.5-flash")

prompt = "Hello"
response = model.generate_content(prompt)

print("\nResponse:\n", response.text)
