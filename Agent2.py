# Import necessary libraries
import os
import openai
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import textwrap
import ollama
# Set up OpenAI API key
openai.api_key = 'sk-7hL_Hsn-nWcxUL3obrthxG6tLbiWcT7QTqtTeBu0NpT3BlbkFJXNsBOD2oqrd0IBya3Wsk-h0B0Im02FGWfCWuj6jwgA'

# Function to generate an educational plan
def generate_educational_plan2(subject):
    prompt = f"Create a comprehensive learning plan for {subject}. Include key topics, subtopics, and recommended learning objectives. Format : 1 - Topic 1 /// 2 - Topic 2 /// 3 - Topic 3 ... etc."
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    plan = response.choices[0].message.content
    return plan

def generate_educational_plan(subject , model_name = "llama3.1"):
    prompt = f"Create a comprehensive learning plan for {subject}. Include key topics, subtopics, and recommended learning objectives. Format : 1 - Topic 1 /// 2 - Topic 2 /// 3 - Topic 3 ... etc."
    response = ollama.chat(model=model_name, messages=[
        {
            'role': 'user',
            'content': prompt,
        },
    ])
    plan = response['message']['content']
    return plan


# Function to extract text from PDF files
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

# Function to read text files
def read_text_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Function to load and process documents from a directory
def load_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        elif filename.endswith('.txt'):
            text = read_text_file(file_path)
        else:
            continue
        documents.append({'text': text, 'metadata': {'source': filename}})
    return documents

# **New Function**: Split documents into chunks
def split_into_chunks(text, chunk_size=500):
    sentences = text.split('. ')
    chunks = []
    current_chunk = ''
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + '. '
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + '. '
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Function to create embeddings and index documents
def create_embeddings_and_index(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return model, index, embeddings

# Function to extract topics from the plan
def extract_topics_from_plan(plan):
    topics = []
    for line in plan.split('\n'):
        line = line.strip()
        if line and (line[0].isdigit() or line[0] == '-'):
            topic = line.lstrip('0123456789.- ').strip()
            if topic:
                topics.append(topic)
    return topics




# Function to retrieve relevant chunks
def retrieve_relevant_chunks(model, index, chunks, query, k=5):
    query_embedding = model.encode(query)
    distances, indices = index.search(np.array([query_embedding]), k)
    return [chunks[i] for i in indices[0]]

# Main function to generate detailed educational content
def generate_detailed_content(subject, directory , model_name = "llama3.1"):
    # Generate the educational plan
    plan = generate_educational_plan(subject)
    print("Educational Plan:")
    print(plan)
    print("\n")

    # Load and split documents into chunks
    documents = load_documents(directory)
    all_chunks = []
    chunk_metadata = []
    for doc in documents:
        chunks = split_into_chunks(doc['text'])
        all_chunks.extend(chunks)
        chunk_metadata.extend([doc['metadata']] * len(chunks))

    # Create embeddings and index
    model, index, embeddings = create_embeddings_and_index(all_chunks)

    # Extract topics from the plan
    topics = extract_topics_from_plan(plan)
    print(topics)
    # Generate detailed content for each topic

    detailed_plan = {}
    for topic in topics:
        print(f"Processing topic: {topic}")
        # Retrieve relevant chunks
        chunks = retrieve_relevant_chunks(model, index, all_chunks, topic, k=5)
        # Combine the text from the chunks
        context = ' '.join(chunks)
        # Ensure context does not exceed token limit
        max_context_length = 9000  # Adjust as needed
        if len(context) > max_context_length:
            context = context[:max_context_length]
        # Create a prompt for the LLM
        prompt = f"Provide a detailed explanation on the topic '{topic}' using the following context:\n\n{context}\n\nDetailed explanation:"
        # Generate detailed content
        #response = openai.chat.completions.create(
        #    model="gpt-4",
        #messages=[{"role": "user", "content": prompt}]
        #)
        # use ollama
        response = ollama.chat(model=model_name, messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ])


        detailed_content = response['message']['content']
        # Store the detailed content
        detailed_plan[topic] = detailed_content
        print(f"Completed topic: {topic}\n")

    return plan, detailed_plan
def save_results(plan, detailed_plan, plan_filename='educational_plan_ollama.txt', detailed_plan_filename='detailed_educational_plan_ollama.txt'):
    # Save the educational plan
    with open(plan_filename, 'w', encoding='utf-8') as f:
        f.write("Educational Plan:\n\n")
        f.write(plan)
    print(f"Educational plan saved to {plan_filename}")

    # Save the detailed educational plan
    with open(detailed_plan_filename, 'w', encoding='utf-8') as f:
        f.write("Detailed Educational Plan:\n\n")
        for topic, content in detailed_plan.items():
            f.write(f"Topic: {topic}\n\n")
            f.write(f"{content}\n\n{'-'*80}\n\n")
    print(f"Detailed educational plan saved to {detailed_plan_filename}")
# Example usage
if __name__ == "__main__":
    subject = "Machine Learning"
    directory = "datadir"  # Replace with the path to your documents
    plan, detailed_plan = generate_detailed_content(subject, directory)

    # Print the final detailed educational plan
    print("Final Detailed Educational Plan:")
    for topic, content in detailed_plan.items():
        print(f"\nTopic: {topic}\n")
        print(content)
    save_results(plan, detailed_plan)
