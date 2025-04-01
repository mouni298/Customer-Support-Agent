# import streamlit as st
# from langchain.schema import Document
# from app import graph  
# import traceback# Import the compiled graph from your existing code

# # Streamlit app setup
# st.title("Agentic App - Question Answering System")
# st.write("Ask a question, and the system will generate an answer based on the available context.")

# # User input
# user_input = st.text_input("Enter your question:", placeholder="Type your question here...")

# # Process user input
# if st.button("Submit"):
#     if user_input.strip():
#         st.write("Processing your question...")
#         try:
#             # Stream the events from the graph
#             for event in graph.stream({"question": user_input}):
#                 for value in event.values():
#                     if "generation" in value:
#                         st.write("### Generated Answer:")
#                         st.write(value["generation"])
#         except Exception as e:
#             st.error(f"An error occurred: {e} {traceback.format_exc()}")
#     else:
#         st.warning("Please enter a valid question.")


from flask import Flask, request, jsonify, render_template
from app import graph  # Import the compiled graph from your existing code
import traceback

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML form for user input

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        # Get the user input from the form
        user_input = request.form.get('question', '').strip()
        if not user_input:
            return jsonify({"error": "Please enter a valid question."}), 400

        # Process the question using the graph
        response = []
        for event in graph.stream({"question": user_input}):
            for value in event.values():
                if "generation" in value:
                    response.append(value["generation"])

        # Return the generated answer
        if response:
            return jsonify({"answer": response})
        else:
            return jsonify({"error": "No answer generated."}), 500
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}", "traceback": traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(debug=True)