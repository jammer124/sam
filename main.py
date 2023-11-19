# main.py
from flask import Flask, request, jsonify
import replicate
import news

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def generate_prediction():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "How are you?")

        # Extract words longer than 4 characters from the prompt
        keywords = [word for word in prompt.split() if len(word) > 4]

        # Fetch news related to the keywords
        news_articles = news.fetch_news(keywords)

        # Create a new prompt by combining the original prompt and the news articles
        # news_prompt = ' '.join([article['content'] for article in news_articles])
        new_prompt = f"{prompt}"  # {news_prompt}"

        model_url = "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"
        input_params = {
            "debug": False,
            "top_k": data.get("top_k", 50),
            "top_p": data.get("top_p", 1),
            "prompt": new_prompt,
            "temperature": data.get("temperature", 0.4),
            "system_prompt": data.get("system_prompt", '''
                You are Sam Altman the CEO of Open AI so talk like him 
                Here are some examples of Sam Altman's way of speaking:
                "I think the most important thing is to be clear and concise. 
                You should be able to explain your ideas in a way that is easy for everyone to understand."
                "I'm not afraid to make bold predictions. I think it's important to think big and set ambitious goals."
                "I'm always open to new ideas. I think it's important to be willing to change your mind if you're presented with new evidence."
                Sam Altman's way of speaking is a reflection of his thinking style. 
                He is a clear thinker who is able to see the big picture. 
                He is also a pragmatist who is not afraid to get his hands dirty. 
                This combination of qualities makes him a successful venture capitalist and a leader in the tech industry.'''
            ),
            "max_new_tokens": data.get("max_new_tokens", 500),
            "min_new_tokens": data.get("min_new_tokens", -1)
        }

        output = replicate.run(model_url, input=input_params)

        # Concatenate the predictions into a single string
        predictions = ' '.join(output)

        return jsonify({"predictions": predictions, "news": news_articles})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
