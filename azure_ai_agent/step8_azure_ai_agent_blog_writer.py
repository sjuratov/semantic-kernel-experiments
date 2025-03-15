# Copyright (c) Microsoft. All rights reserved.

import asyncio

from azure.identity.aio import DefaultAzureCredential

from semantic_kernel.agents.azure_ai import AzureAIAgent
from semantic_kernel.agents.strategies import TerminationStrategy
from semantic_kernel.agents import AgentGroupChat
from semantic_kernel.contents import AuthorRole

"""
The following sample demonstrates how to use existing
Azure AI Agents within Semantic Kernel. This sample requires that you
have an existing agents created either previously in code or via the
Azure Portal (or CLI).
"""

class ApprovalTerminationStrategy(TerminationStrategy):
    """A strategy for determining when an agent should terminate."""

    async def should_agent_terminate(self, agent, history):
        """Check if the agent should terminate."""
        return "approved" in history[-1].content.lower()

# Give agents a task to create blog post
TASK = "Why is the sky blue?"

async def main() -> None:
    async with (
        DefaultAzureCredential() as creds,
        AzureAIAgent.create_client(credential=creds) as client,
    ):
        # 1. Retrieve the Web Search agent definition based on the `agent_id`
        web_search_agent_definition = await client.agents.get_agent(
            agent_id="asst_Q4qsv2SYDnydk2fgrus66ObQ",
        )

        # 2. Create a Semantic Kernel agent for the Web Search Azure AI agent
        agent_web_search = AzureAIAgent(
            client=client,
            definition=web_search_agent_definition,
        )

        # 3. Retrieve the Blog Writer agent definition based on the `agent_id`
        blog_writer_agent_definition = await client.agents.get_agent(
            agent_id="asst_Pm1QqlryTVIGaTBIB3I25d8b",
        )

        # 4. Create a Semantic Kernel agent for the Blog Writer Azure AI agent
        agent_blog_writer = AzureAIAgent(
            client=client,
            definition=blog_writer_agent_definition,
        )

        # 5. Retrieve the Blog Writer Critique agent definition based on the `agent_id`
        blog_writer_critique_agent_definition = await client.agents.get_agent(
            agent_id="asst_Pfoz0uTEetM6U6IpEAqMryT8",
        )

        # 6. Create a Semantic Kernel agent for the Blog Writer Critique Azure AI agent
        agent_blog_writer_critique = AzureAIAgent(
            client=client,
            definition=blog_writer_critique_agent_definition,
        )

        # 7. Place the agents in a group chat with a custom termination strategy
        chat = AgentGroupChat(
            agents=[agent_web_search, agent_blog_writer, agent_blog_writer_critique],
            termination_strategy=ApprovalTerminationStrategy(agents=[agent_blog_writer_critique], maximum_iterations=10),
        )

        try:
            # 6. Add the task as a message to the group chat
            await chat.add_chat_message(message=TASK)
            print(f"# {AuthorRole.USER}: '{TASK}'")
            # 7. Invoke the chat
            async for content in chat.invoke():
                print(f"# {content.role} - {content.name or '*'}: '{content.content}'")
        finally:
            # 8. Cleanup: Delete the agents
            await chat.reset()

if __name__ == "__main__":
    asyncio.run(main())
