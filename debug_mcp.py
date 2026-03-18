import asyncio
from mcp_client import ZoteroMCPClient

async def run_test():
    zc = ZoteroMCPClient()
    await zc.connect()
    print("Testing search_items parameter formatting...")
    
    # 1. Test original bug (query passed as dictionary)
    try:
        res1 = await zc.search_items({"limit": 50})
        print(f"Test 1 (Dict query) result: {res1[:100]}")
    except Exception as e:
        print(f"Test 1 error: {e}")
        
    # 2. Test fix approach (passed as empty string)
    try:
        res2 = await zc.search_items(query="", limit=50)
        print(f"Test 2 (String query='') result: {res2[:100]}")
    except Exception as e:
        print(f"Test 2 error: {e}")
        
    await zc.disconnect()

if __name__ == "__main__":
    asyncio.run(run_test())
