import os
import asyncio
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError

base_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(base_dir, "zotero-mcp-server", ".env")
load_dotenv(env_path)

class ZoteroMCPClient:
    def __init__(self, mcp_server_path=None, api_key=None, user_id=None):
        if mcp_server_path is None:
            mcp_server_path = os.path.join(base_dir, "zotero-mcp-server", "dist", "index.js")
        self.server_path = mcp_server_path
        self.api_key = api_key or os.getenv("ZOTERO_API_KEY")
        self.user_id = user_id or os.getenv("ZOTERO_USER_ID")
        self._session = None
        self._exit_stack = None

    async def connect(self):
        if not self.api_key or not self.user_id:
            raise ValueError("Zotero API Key and User ID are required to connect.")
            
        import contextlib
        self._exit_stack = contextlib.AsyncExitStack()
        
        env = os.environ.copy()
        env["ZOTERO_API_KEY"] = self.api_key
        env["ZOTERO_USER_ID"] = self.user_id

        server_params = StdioServerParameters(
            command="node",
            args=[self.server_path],
            env=env
        )

        stdio_transport = await self._exit_stack.enter_async_context(stdio_client(server_params))
        self.read, self.write = stdio_transport
        
        self._session = await self._exit_stack.enter_async_context(ClientSession(self.read, self.write))
        await self._session.initialize()

    async def disconnect(self):
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._session = None

    async def search_items(self, query: str, limit: int = 5, **kwargs):
        if not self._session:
            await self.connect()
        try:
            params = {"query": query, "limit": limit}
            params.update(kwargs)
            print(f"DEBUG mcp_client.py: params being sent to search_items={params}")
            result = await self._session.call_tool("search_items", params)
            if result.content and len(result.content) > 0:
                return result.content[0].text
            return "[]"
        except McpError as e:
            return f'[{{"error": "{str(e)}"}}]'
        except Exception as e:
            return f'[{{"error": "Connection error: {e}"}}]'

    async def generate_citation(self, item_keys: list, style: str = "apa"):
        if not self._session:
            await self.connect()
        try:
            result = await self._session.call_tool("generate_citation", {"itemKeys": item_keys, "style": style})
            if result.content and len(result.content) > 0:
                return result.content[0].text
            return "Error formatting citation: no content returned"
        except McpError as e:
            return f"Error formatting citation: {e}"
        except Exception as e:
            return f"Connection error: {e}"

    async def extract_pdf_text(self, item_key: str):
        if not self._session:
            await self.connect()
        try:
            # We don't have to provide pages unless required.
            result = await self._session.call_tool("extract_pdf_text", {"itemKey": item_key})
            if result.content and len(result.content) > 0:
                return result.content[0].text
            return ""
        except McpError as e:
            return f"Error extracting PDF: {e}"
        except Exception as e:
            return f"Connection error: {e}"

    async def manage_collections(self, **kwargs):
        if not self._session:
            await self.connect()
        try:
            result = await self._session.call_tool("manage_collections", kwargs)
            if result.content and len(result.content) > 0:
                return result.content[0].text
            return "[]"
        except McpError as e:
            return f'[{{"error": "{str(e)}"}}]'
        except Exception as e:
            return f'[{{"error": "Connection error: {e}"}}]'
