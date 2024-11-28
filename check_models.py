import openai

# Replace with your actual API key
openai.api_key = "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"  # Use your full API key here




try:
    # Example GPT-3.5-turbo request
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! Can you confirm GPT-3.5-turbo is working?"}
        ]
    )
    print("Response:", response['choices'][0]['message']['content'])
except openai.error.InvalidRequestError as e:
    print("Invalid Request:", str(e))
except openai.error.AuthenticationError as e:
    print("Authentication Error:", str(e))
except openai.error.OpenAIError as e:
    print("General OpenAI Error:", str(e))
except Exception as e:
    print("An unexpected error occurred:", str(e))