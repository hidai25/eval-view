#!/usr/bin/env python3
"""
Create an OpenAI Assistant for testing with EvalView.

Usage:
    python create_assistant.py
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env and .env.local
load_dotenv()
load_dotenv(".env.local", override=True)

def main():
    client = OpenAI()

    # Create an assistant with code interpreter
    assistant = client.beta.assistants.create(
        name="EvalView Test Assistant",
        instructions="""You are a helpful technical assistant.

When asked to analyze topics:
- Provide comprehensive, well-structured responses
- Use code interpreter when calculations or data analysis would help
- Include pros, cons, and practical considerations
- Be specific with examples when relevant""",
        model="gpt-4o-mini",  # Cost-effective for testing
        tools=[
            {"type": "code_interpreter"}
        ]
    )

    print(f"✅ Assistant created!")
    print(f"")
    print(f"   ID: {assistant.id}")
    print(f"   Name: {assistant.name}")
    print(f"   Model: {assistant.model}")
    print(f"")
    print(f"Next steps:")
    print(f"")
    print(f"   1. Set the assistant ID:")
    print(f"      export OPENAI_ASSISTANT_ID={assistant.id}")
    print(f"")
    print(f"   2. Run the test:")
    print(f"      evalview run --pattern examples/openai-assistants/test-case.yaml")
    print(f"")
    print(f"   Or add to your .env file:")
    print(f"      OPENAI_ASSISTANT_ID={assistant.id}")


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not set")
        print("   Run: export OPENAI_API_KEY=your-key")
        exit(1)

    main()
