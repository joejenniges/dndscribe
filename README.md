# dndscribe


## Todo List
1. Improve transcription
    1. Pre-process lines to remove repeated phrases or incoherent sentences
    2. Correct spelling using language-tool or an LLM
    3. Provide tokenized words for whisper
2. Reliability improvements
    1. Detect when bot crashes and was previously transcribing, make it rejoin the channel and continue transcribing
    2. Fallback audio processing when memory buffer fails (DONE)
    3. Provide docker containers with access to GPU, need to research cross platform compatibility for this
3. UX
    1. Ability to set whisper preamble prompt of common D&D phrases that might be used
    2. Allow discord users to set their PC name, allowing PC name to be who spoke in the transcript instead of their discord username
4. Summarizing
    1. Continue search for a decent LLM that can summarize
    2. Chunk transcripts into sections to be sent to LLM with larger summaries
    3. Combine larger summaries into an overall summary
5. Other
    1. Postgres for storage instead of text files