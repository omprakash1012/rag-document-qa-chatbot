"""
Pluggable embeddings provider.

Uses OpenAI embeddings when OPENAI_API_KEY is set (production path,
matches the resume's OpenAI API + LangChain stack). Falls back to a free
local sentence-transformers model when no key is present, so the project
can be cloned and run end-to-end without any paid API access.
"""
import os


def get_embeddings():
      if os.getenv("OPENAI_API_KEY"):
                from langchain_openai import OpenAIEmbeddings
                return OpenAIEmbeddings(model="text-embedding-3-small")

      from langchain_community.embeddings import HuggingFaceEmbeddings
      return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_llm(temperature: float = 0.0):
      if os.getenv("OPENAI_API_KEY"):
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)

      raise RuntimeError(
          "No OPENAI_API_KEY set. Set it in a .env file to enable answer generation. "
          "Retrieval (chunk search) works without it, but generation needs an LLM."
      )
