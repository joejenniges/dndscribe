from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain.chains import LLMChain

# Consider using a different LLM:
# Yi: https://github.com/01-ai/Yi
# Mixtral https://huggingface.co/mistralai/Mixtral-8x7B-v0.1

# Bugs:
# Determine what causes random crashes


class Summarizer:

    def __init__(self):
        self.llm = OllamaLLM(model="llama2-uncensored", base_url="http://localhost:11434")
        self.initial_prompt = "You are an expert in Dungeons and Dragons. You are tasked with taking a transcribed recording of a session and summarizing it to the best of your ability. Please provide an overall summary of major plot points, results of battle (plus any highlights for critical hits), as well as action items for future sessions and a list of any currency or items gained, who gained them, and how much. You are being given a partial section of the transcript so please provide as many details as possible with the knowledge that all summarized sections will then be combined into one and then summarized again into something smaller."


    def summarize_chunk(self,chunk, previous_summary=None):
        prompt = self.initial_prompt
        if previous_summary:
            # Append the previous summary for continuity
            prompt += f"\nPrevious Summary: {previous_summary}"
        
        prompt += f"\nCurrent Transcript: {chunk}"

        # Create the LangChain LLMChain with Ollama
        prompt_template = ChatPromptTemplate(input_variables=["transcript"], template=prompt)
        chain = LLMChain(llm=self.llm, prompt=prompt_template)

        # Get the summary for the chunk
        return chain.run(transcript=chunk)

    def chunk_transcript_with_overlap(self, transcript, max_tokens=3000, overlap_tokens=100):
        """
        Chunk the transcript while keeping overlap to maintain context.
        Splits at sentence boundaries to maintain readability.
        
        :param transcript: The full transcript to be chunked
        :param max_tokens: The maximum number of tokens per chunk (without overlap)
        :param overlap_tokens: The number of tokens to overlap between consecutive chunks
        :return: A list of chunks with overlap
        """
        import re
        
        # Split the transcript into sentences
        # This regex matches sentence endings (period, question mark, exclamation mark)
        # followed by a space or newline
        sentences = re.split(r'([.!?][\s\n])', transcript)
        
        # Recombine the sentences with their punctuation
        complete_sentences = []
        for i in range(0, len(sentences)-1, 2):
            if i+1 < len(sentences):
                complete_sentences.append(sentences[i] + sentences[i+1])
            else:
                complete_sentences.append(sentences[i])
        
        # If the last element doesn't have punctuation, add it
        if len(sentences) % 2 == 1:
            complete_sentences.append(sentences[-1])
        
        # Create chunks based on sentence boundaries
        chunks = []
        current_chunk = ""
        
        for sentence in complete_sentences:
            # If adding this sentence would exceed max_tokens, start a new chunk
            if len(current_chunk) + len(sentence) > max_tokens:
                chunks.append(current_chunk)
                # Start new chunk with some overlap from the previous chunk
                overlap_point = max(0, len(current_chunk) - overlap_tokens)
                current_chunk = current_chunk[overlap_point:] + sentence
            else:
                current_chunk += sentence
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks