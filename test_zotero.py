import asyncio
import json
import os
import sys
from mcp_client import ZoteroMCPClient

async def main():
    api_key = os.environ.get("ZOTERO_API_KEY")
    user_id = os.environ.get("ZOTERO_USER_ID")
    
    client = ZoteroMCPClient(api_key=api_key, user_id=user_id)
    await client.connect()
    
    print("Connected. Fetching attachments...", file=sys.stderr)
    try:
        res = await asyncio.wait_for(client.search_items("", limit=5, itemType="attachment"), timeout=15.0)
        data = json.loads(res)
        items = data.get("items", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
        print(f"Attachments found: {len(items)}", file=sys.stderr)
        
        for item in items:
            key = item.get("key")
            title = item.get("title") or item.get("data", {}).get("title", "PDF")
            print(f"Trying to extract PDF for {key}: {title}", file=sys.stderr)
            text_json = await client.extract_pdf_text(key)
            if not text_json:
                print("No text received", file=sys.stderr)
                continue
            parsed = json.loads(text_json)
            print(f"PDF extract success, Extracted characters: {len(parsed.get('content', ''))}", file=sys.stderr)
            break
            
    except Exception as e:
        print(f"Search failed: {e}", file=sys.stderr)
        
    await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
